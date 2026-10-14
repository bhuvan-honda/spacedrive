"""Gradient conflict between the occupancy auxiliary loss and the planning loss on the shared trainable weights.

For a trained checkpoint, on `--num-samples` random training frames (each preceded by its `--history` previous
frames of the same scene, run without grad, so the ego-status memory is as in training), the training forward
is run once (eval mode: no LoRA dropout, no gradient checkpointing) and the two losses are back-propagated
separately:
    g_plan = d(planning loss = every 'loss' term except loss_occ) / d theta,   g_occ = d(loss_occ) / d theta
(loss_occ already includes its loss_weight, so g_occ is the gradient the optimizer actually saw). For every
group of shared parameters (LoRA of each LLM layer, epipolar adapter, FiLM, other pre-LLM modules) it reports
  cos_sum   cosine of the gradients summed over all samples (the direction an epoch of updates takes)
  cos_mean  mean per-sample cosine, neg_frac = fraction of samples with cosine < 0
  ratio     mean |g_occ| / mean |g_plan| (how strongly the occupancy loss pulls on these weights)
cos < 0: the occupancy loss pushes these weights against planning; cos ~ 0 with a large ratio: it spends their
capacity on something unrelated to planning. Layers at or after the occupancy read-out layer get no g_occ.

    docker exec -i -e CUDA_VISIBLE_DEVICES=0 spacedrive_flash bash -lc 'cd /workspace && python \
        tools/occ_grad_cosine.py --name occ_l14 --config results/spacedrive_sd2_occ_l14/sd2_occ.py \
        --ckpt results/spacedrive_sd2_occ_l14/iter_5274.pth'
    python tools/occ_grad_cosine.py --plot occ_l14 occ_l28 full        # combined figure + table from the JSONs
"""
import argparse
import json
import os
import os.path as osp
import re
import sys
import time

import numpy as np

sys.path.insert(0, '.')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name')
    p.add_argument('--config', help='config file (work-dir dumps with `gpu_ids = [device(...)]` are fine)')
    p.add_argument('--ckpt', help='checkpoint; omitted = freshly initialised model (sanity checks only)')
    p.add_argument('--num-samples', type=int, default=48)
    p.add_argument('--history', type=int, default=2, help='previous frames run first to fill the ego memory')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='vis/grad_cosine')
    p.add_argument('--plot', nargs='+', metavar='NAME', help='only combine existing <out>/<NAME>.json files')
    return p.parse_args()


def group_of(name):
    m = re.search(r'language_model\.layers\.(\d+)\.', name)
    if m:
        return f'llm.L{int(m.group(1)):02d}'
    if '.visual.' in name:
        return 'vision'
    if name.startswith('lm_head.'):
        return 'llm.other'
    return 'pre-llm.' + name.split('.')[0]


def load_cfg(path):
    from mmcv import Config
    text = open(path).read()
    if '\ngpu_ids = [' not in text:
        return Config.fromfile(path)  # source config (may use relative _base_ paths)
    # work-dir dump (flattened): `gpu_ids = [device(...), ...]` is not valid python
    return Config.fromstring(re.sub(r'\ngpu_ids = \[.*?\]\n', '\n', text, flags=re.S), '.py')


def run(args):
    import torch
    from mmcv.parallel import collate, scatter
    from mmcv.runner import wrap_fp16_model
    import projects.mmdet3d_plugin  # noqa: F401
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    torch.manual_seed(args.seed)
    cfg = load_cfg(args.config)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location='cpu')
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.get('state_dict', sd).items()}
        missing, _ = model.load_state_dict(sd, strict=False)
        del sd
        assert not missing, f'missing keys: {missing[:5]}'
    else:
        model.init_weights()
    model = model.cuda()
    wrap_fp16_model(model)
    model.eval()  # no LoRA dropout; HF disables gradient checkpointing outside training -> plain autograd
    assert model.bev_occ_head is not None, 'config has no occupancy head'

    shared = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not n.startswith('bev_occ_head.')]
    groups = sorted({group_of(n) for n, _ in shared})
    print(f'[{args.name}] {len(shared)} shared trainable tensors, '
          f'{sum(p.numel() for _, p in shared) / 1e6:.1f}M params, groups: {groups}', flush=True)

    tcfg = cfg.data.train.copy()
    tcfg['pipeline'] = [t for t in tcfg['pipeline'] if not t['type'].startswith('LoadPoints')]
    ds = build_dataset(tcfg)
    infos = ds.data_infos
    H = args.history
    cand = [i for i in range(H, len(infos)) if 'gt_planning' in infos[i]
            and infos[i]['gt_planning_mask'].any(axis=-1).sum() == 6
            and all(infos[i - k]['scene_token'] == infos[i]['scene_token'] for k in range(1, H + 1))]
    rng = np.random.RandomState(args.seed)
    samples = sorted(rng.choice(cand, args.num_samples, replace=False).tolist())

    def batch(i):
        return scatter(collate([ds[i]], samples_per_gpu=1), [torch.cuda.current_device()])[0]

    def grads():
        return {n: (p.grad.detach().float().flatten() if p.grad is not None else None) for n, p in shared}

    sum_p = {n: torch.zeros(p.numel(), device='cuda') for n, p in shared}
    sum_o = {n: torch.zeros(p.numel(), device='cuda') for n, p in shared}
    per = {g: dict(dot=[], pp=[], oo=[]) for g in groups + ['ALL']}
    t0 = time.time()
    for s, i in enumerate(samples):
        model.reset_memory()
        with torch.no_grad():
            for k in range(H, 0, -1):
                model(return_loss=True, **batch(i - k))
        model.zero_grad(set_to_none=True)
        losses = model(return_loss=True, **batch(i))
        plan = sum(v for k, v in losses.items() if 'loss' in k and k != 'loss_occ')
        plan.backward(retain_graph=True)
        gp = grads()
        model.zero_grad(set_to_none=True)
        losses['loss_occ'].backward()
        go = grads()
        model.zero_grad(set_to_none=True)
        acc = {g: [0.0, 0.0, 0.0] for g in groups + ['ALL']}
        for n, _ in shared:
            a, b = gp[n], go[n]
            if a is not None:
                sum_p[n] += a
            if b is not None:
                sum_o[n] += b
            d = float(a @ b) if a is not None and b is not None else 0.0
            pp = float(a @ a) if a is not None else 0.0
            oo = float(b @ b) if b is not None else 0.0
            for g in (group_of(n), 'ALL'):
                acc[g][0] += d
                acc[g][1] += pp
                acc[g][2] += oo
        for g, (d, pp, oo) in acc.items():
            per[g]['dot'].append(d)
            per[g]['pp'].append(pp)
            per[g]['oo'].append(oo)
        c = acc['ALL'][0] / max(np.sqrt(acc['ALL'][1] * acc['ALL'][2]), 1e-30)
        print(f'[{args.name}] {s + 1}/{len(samples)} frame {i}: plan {float(plan):.4f} occ {float(losses["loss_occ"]):.4f} '
              f'cos(ALL) {c:+.3f} |g_occ|/|g_plan| {np.sqrt(acc["ALL"][2] / max(acc["ALL"][1], 1e-30)):.2f} '
              f'({time.time() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.0f} GiB)', flush=True)
        del losses, plan, gp, go

    res = {}
    for g in groups + ['ALL']:
        names = [n for n, _ in shared if g == 'ALL' or group_of(n) == g]
        d = sum(float(sum_p[n] @ sum_o[n]) for n in names)
        pp = sum(float(sum_p[n] @ sum_p[n]) for n in names)
        oo = sum(float(sum_o[n] @ sum_o[n]) for n in names)
        dot, ppa, ooa = (np.array(per[g][k]) for k in ('dot', 'pp', 'oo'))
        ok = (ppa > 0) & (ooa > 0)
        cos = dot[ok] / np.sqrt(ppa[ok] * ooa[ok])
        res[g] = dict(
            cos_sum=d / np.sqrt(pp * oo) if pp > 0 and oo > 0 else None,
            cos_mean=float(cos.mean()) if ok.any() else None,
            neg_frac=float((cos < 0).mean()) if ok.any() else None,
            ratio=float(np.sqrt(ooa).mean() / max(np.sqrt(ppa).mean(), 1e-30)),
            plan_norm=float(np.sqrt(ppa).mean()), occ_norm=float(np.sqrt(ooa).mean()))
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(name=args.name, config=args.config, ckpt=args.ckpt, samples=samples,
                   occ_layer=model.bev_occ_head.layer, detach_input=getattr(model.bev_occ_head, 'detach_input', False),
                   groups=res), open(osp.join(args.out, f'{args.name}.json'), 'w'), indent=1)
    table([args.name], args.out)


def fmt(v, f='{:+.3f}'):
    return '-' if v is None else f.format(v)


def table(names, out):
    lines = []
    for name in names:
        r = json.load(open(osp.join(out, f'{name}.json')))
        lines += [f'\n### {name} (occupancy read-out at LLM layer {r["occ_layer"]}, {len(r["samples"])} train frames)\n',
                  '| params | cos_sum | cos_mean | neg_frac | \\|g_occ\\|/\\|g_plan\\| |', '|---|---|---|---|---|']
        for g, v in r['groups'].items():
            if v['occ_norm'] == 0 and g != 'ALL':
                continue  # layers after the read-out: no occupancy gradient
            lines.append(f'| {g} | {fmt(v["cos_sum"])} | {fmt(v["cos_mean"])} | {fmt(v["neg_frac"], "{:.2f}")} | '
                         f'{fmt(v["ratio"], "{:.2f}")} |')
    text = '\n'.join(lines)
    print(text)
    return text


def plot(names, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name in names:
        r = json.load(open(osp.join(out, f'{name}.json')))['groups']
        layers = sorted(g for g in r if g.startswith('llm.L') and r[g]['occ_norm'] > 0)
        x = [int(g[5:]) for g in layers]
        axes[0].plot(x, [r[g]['cos_sum'] for g in layers], 'o-', label=name)
        axes[1].semilogy(x, [r[g]['ratio'] for g in layers], 'o-', label=name)
    axes[0].axhline(0, color='k', lw=0.8)
    axes[0].set(xlabel='LLM layer (LoRA)', ylabel='cos(g_occ, g_plan), summed grads', title='gradient alignment')
    axes[1].axhline(1, color='k', lw=0.8)
    axes[1].set(xlabel='LLM layer (LoRA)', ylabel='|g_occ| / |g_plan|', title='relative pull of the occupancy loss')
    for a in axes:
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(osp.join(out, 'grad_cosine_by_layer.png'), dpi=110)
    open(osp.join(out, 'summary.md'), 'w').write('# Occupancy vs planning gradient alignment\n' + table(names, out) + '\n')


if __name__ == '__main__':
    a = parse_args()
    if a.plot:
        plot(a.plot, a.out)
    else:
        run(a)
