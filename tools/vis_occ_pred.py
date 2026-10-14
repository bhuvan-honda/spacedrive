"""Visualise the BEV occupancy predictions of SD2 occupancy-head runs side by side (e.g. occ_l14 vs occ_l28).

Step 1 (GPU, once per run, skipped if the logits exist): load the run's config dump + checkpoint, run the
teacher-forced training forward (eval mode, no grad, batch 1) on frames of the ValLossHook val scenes and
save the occupancy-head logits -> <out>/<run>/<token>.npy (float16, 200x200, grid[i, j] i = forward, j = left).
Step 2 (CPU): per-frame PNGs (cameras, 3-class LiDAR GT, P(occupied) and an error map per run) and
<out>/summary.md with IoU overall / by range (known cells only, same definition as the training occ_iou).

    docker exec -i spacedrive_flash bash -lc 'cd /workspace && CUDA_VISIBLE_DEVICES=7 python tools/vis_occ_pred.py \
        --run occ_l14=results/spacedrive_sd2_occ_l14 --run occ_l28=results/spacedrive_sd2_occ_l28'
"""
import argparse
import gc
import glob
import json
import os
import os.path as osp
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, '.')

FREE, OCC, UNKNOWN = 0, 1, 2
R, RES, N = 50.0, 0.5, 200
EGO_X, EGO_HW = (-1.2, 3.8), 1.2  # tools/gen_bev_occ.py defaults
RANGE_BINS = [0, 10, 20, 30, 50]
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True,
                   help='NAME=WORK_DIR; config = the *.py dump in WORK_DIR, ckpt = its highest iter_*.pth')
    p.add_argument('--ckpt-iter', type=int, help='use iter_<N>.pth instead of the highest one')
    p.add_argument('--out', default='vis/occ_pred')
    p.add_argument('--num-scenes', type=int, default=16, help='val scenes (same choice as ValLossHook)')
    p.add_argument('--frames-per-scene', type=int, default=3)
    p.add_argument('--max-gib', type=float, default=26.0,
                   help='cap on this process GPU memory, so it OOMs itself rather than a job sharing the GPU')
    p.add_argument('--render-only', action='store_true')
    return p.parse_args()


# ----------------------------------------------------------------------------------------------- inference
def select_frames(ds, num_scenes, per_scene):
    """Scenes as in ValLossHook (num_scenes spread evenly over val, frames with a full 3 s GT trajectory),
    then `per_scene` frames spread evenly inside each scene."""
    scene_of = ds.flag
    scenes = np.unique(scene_of)
    chosen = scenes[::max(1, len(scenes) // num_scenes)][:num_scenes]
    out = []
    for s in chosen:
        idx = [i for i, info in enumerate(ds.data_infos) if scene_of[i] == s and 'gt_planning' in info
               and info['gt_planning_mask'].any(axis=-1).sum() == 6]
        out += [idx[len(idx) * (k + 1) // (per_scene + 1)] for k in range(per_scene)] if idx else []
    return out


def infer(name, work_dir, args):
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import wrap_fp16_model
    import projects.mmdet3d_plugin  # noqa: F401  (registers the model / dataset / pipeline)
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    cfg_path, = glob.glob(osp.join(work_dir, '*.py'))
    ckpts = {int(re.search(r'iter_(\d+)\.pth', p).group(1)): p for p in glob.glob(osp.join(work_dir, 'iter_*.pth'))}
    it = args.ckpt_iter or max(ckpts)
    ckpt = ckpts[it]
    # the dump's `gpu_ids = [device(...), ...]` is not valid python; drop it before loading
    text = re.sub(r'\ngpu_ids = \[.*?\]\n', '\n', open(cfg_path).read(), flags=re.S)
    cfg = Config.fromstring(text, '.py')
    print(f'[{name}] config {cfg_path}, ckpt {ckpt}, bev_occ {cfg.model.bev_occ}', flush=True)

    torch.cuda.set_per_process_memory_fraction(
        min(1.0, args.max_gib * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    sd = torch.load(ckpt, map_location='cpu')
    sd = sd.get('state_dict', sd)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd
    gc.collect()
    print(f'[{name}] missing {len(missing)} {missing[:3]}, unexpected {len(unexpected)} {unexpected[:3]}', flush=True)
    assert not [k for k in missing if k.startswith('bev_occ_head.')], 'occupancy head weights missing'
    model = model.cuda().eval()
    if cfg.get('fp16'):
        wrap_fp16_model(model)

    vcfg, = [h['dataset'] for h in cfg.custom_hooks if h['type'] == 'ValLossHook']
    ds = build_dataset(vcfg)
    frames = select_frames(ds, args.num_scenes, args.frames_per_scene)

    captured = {}
    model.bev_occ_head.register_forward_hook(lambda m, i, o: captured.__setitem__('logits', o.detach()))
    os.makedirs(osp.join(args.out, name), exist_ok=True)
    meta = []
    with torch.no_grad():
        for n, i in enumerate(frames):
            data = scatter(collate([ds[i]], samples_per_gpu=1), [torch.cuda.current_device()])[0]
            if hasattr(model, 'reset_memory'):
                model.reset_memory()  # vision tokens precede the ego tokens (causal LM), so memory does not matter
            losses = model(return_loss=True, **data)
            token = ds.data_infos[i]['token']
            np.save(osp.join(args.out, name, token + '.npy'), captured['logits'][0].half().cpu().numpy())
            meta.append(dict(token=token, occ_iou=float(losses['occ_iou']), tf_ade=float(losses['tf_ade'])))
            print(f'[{name}] {n + 1}/{len(frames)} {token[:8]} occ_iou {meta[-1]["occ_iou"]:.3f} '
                  f'peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB', flush=True)
    json.dump(dict(config=cfg_path, ckpt=ckpt, frames=meta), open(osp.join(args.out, name, 'meta.json'), 'w'), indent=1)
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------- rendering
def load_gt(token):
    bits = np.unpackbits(np.load(osp.join('data/nuscenes/bev_occ3', token + '.npy')))[:2 * N * N]
    occ, known = bits[:N * N].reshape(N, N).astype(bool), bits[N * N:].reshape(N, N).astype(bool)
    return occ, known


def cell_range():
    x = -R + (np.arange(N) + 0.5) * RES
    return np.hypot(x[:, None], x[None, :])


def counts(pred, occ, known, rng):
    """(len(RANGE_BINS), 3) TP / FP / FN on known cells; row 0 = all ranges, then one row per range bin."""
    out = []
    for lo, hi in [(0, np.inf)] + list(zip(RANGE_BINS[:-1], RANGE_BINS[1:])):
        m = known & (rng >= lo) & (rng < hi)
        out.append([(pred & occ & m).sum(), (pred & ~occ & m).sum(), (~pred & occ & m).sum()])
    return np.array(out, np.int64)


def iou(c):
    return c[..., 0] / np.maximum(c.sum(-1), 1)


def bev_show(ax, img, **kw):
    # grid[i, j] (i forward, j left) -> image with forward up and left on the left; axes in metres (y, x)
    ax.imshow(img[::-1, ::-1], extent=[R, -R, -R, R], interpolation='nearest', **kw)
    ax.set_xlim(R, -R)
    ax.set_ylim(-R, R)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_boxes(ax, info, color='lime'):
    b = info['gt_boxes']
    for x, y, l, w, yaw in zip(b[:, 0], b[:, 1], b[:, 3], b[:, 4], b[:, 6]):
        c = np.array([[l, w], [l, -w], [-l, -w], [-l, w], [l, w]]) / 2
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        c = c @ rot.T + [x, y]
        ax.plot(c[:, 1], c[:, 0], color=color, lw=0.6)
    ego = np.array([[EGO_X[1], EGO_HW], [EGO_X[1], -EGO_HW], [EGO_X[0], -EGO_HW], [EGO_X[0], EGO_HW], [EGO_X[1], EGO_HW]])
    ax.plot(ego[:, 1], ego[:, 0], color='red', lw=1.0)


def render(args, names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    infos = pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl', 'rb'))
    infos = {i['token']: i for i in (infos['infos'] if isinstance(infos, dict) else infos)}
    metas = {n: {f['token']: f for f in json.load(open(osp.join(args.out, n, 'meta.json')))['frames']} for n in names}
    tokens = [t for t in metas[names[0]] if all(t in metas[n] for n in names)]
    rng = cell_range()
    gt_cmap = ListedColormap([(0.75, 0.75, 0.75), (0.05, 0.05, 0.05), (0.35, 0.35, 0.55)])  # free, occ, unknown
    # error map: 0 unknown, 1 TN, 2 TP, 3 FP, 4 FN
    err_cmap = ListedColormap([(0.35, 0.35, 0.55), (0.85, 0.85, 0.85), (0.05, 0.05, 0.05), (0.9, 0.15, 0.15), (0.1, 0.45, 0.95)])
    tot = {n: np.zeros((len(RANGE_BINS), 3), np.int64) for n in names}
    per_frame = {n: [] for n in names}
    vis_dir = osp.join(args.out, 'frames')
    os.makedirs(vis_dir, exist_ok=True)

    for k, t in enumerate(tokens):
        info = infos[t]
        occ, known = load_gt(t)
        label = np.where(occ, OCC, np.where(known, FREE, UNKNOWN))
        nc = 2 + len(names)
        fig = plt.figure(figsize=(4.2 * nc, 8.6))
        gs = fig.add_gridspec(3, nc, height_ratios=[1.0, 4.0, 4.0])
        cam_ax = fig.add_subplot(gs[0, :])
        imgs = []
        for cam in CAMS:
            im = plt.imread(info['cams'][cam]['data_path'])
            imgs.append(im[::4, ::4])
        cam_ax.imshow(np.concatenate([np.concatenate(imgs[:3], 1), np.concatenate(imgs[3:], 1)], 1))
        cam_ax.set_axis_off()
        cam_ax.set_title(f'{info["scene_name"]}  {t[:8]}  cmd: {info.get("gt_planning_command_desc", "")}'[:140], fontsize=9)

        ax = fig.add_subplot(gs[1, 0])
        bev_show(ax, label, cmap=gt_cmap, vmin=0, vmax=2)
        draw_boxes(ax, info)
        wp = info['gt_planning'][0, :, :2]
        ax.plot(wp[:, 1], wp[:, 0], 'o', color='orange', ms=3)
        ax.set_title('LiDAR GT (3-class)', fontsize=10)
        ax = fig.add_subplot(gs[2, 0])
        ax.set_axis_off()
        ax.legend(handles=[Patch(color=gt_cmap(0), label='GT free'), Patch(color=gt_cmap(1), label='GT occupied'),
                           Patch(color=gt_cmap(2), label='GT unknown (ignored)'),
                           Patch(color=err_cmap(2), label='TP'), Patch(color=err_cmap(3), label='FP'),
                           Patch(color=err_cmap(4), label='FN'), Patch(color=err_cmap(1), label='TN'),
                           Patch(edgecolor='lime', fill=False, label='GT boxes'),
                           Patch(edgecolor='red', fill=False, label='ego')], loc='center', fontsize=9, frameon=False)

        lines = []
        for c, n in enumerate(names):
            logit = np.load(osp.join(args.out, n, t + '.npy')).astype(np.float32)
            prob = 1 / (1 + np.exp(-logit))
            pred = logit > 0
            cnt = counts(pred, occ, known, rng)
            tot[n] += cnt
            per_frame[n].append(iou(cnt[0]))
            ax = fig.add_subplot(gs[1, 2 + c])
            bev_show(ax, prob, cmap='magma', vmin=0, vmax=1)
            draw_boxes(ax, info, color='cyan')
            ax.set_title(f'{n}: P(occupied)', fontsize=10)
            err = np.where(~known, 0, np.where(pred & occ, 2, np.where(pred, 3, np.where(occ, 4, 1))))
            ax = fig.add_subplot(gs[2, 2 + c])
            bev_show(ax, err, cmap=err_cmap, vmin=0, vmax=4)
            draw_boxes(ax, info, color='lime')
            ious = iou(cnt)
            ax.set_title(f'{n}: errors  IoU {ious[0]:.3f}', fontsize=10)
            lines.append(f'{n}\n  IoU {ious[0]:.3f}\n  ' + '\n  '.join(
                f'{lo}-{hi} m: {v:.3f}' for lo, hi, v in zip(RANGE_BINS[:-1], RANGE_BINS[1:], ious[1:])))
        tax = fig.add_subplot(gs[1:, 1])
        tax.set_axis_off()
        tax.text(0.05, 0.95, 'IoU on known cells\n(range = distance\nfrom ego centre)\n\n' + '\n\n'.join(lines),
                 va='top', fontsize=9, family='monospace', transform=tax.transAxes)
        fig.tight_layout()
        fig.savefig(osp.join(vis_dir, f'{k:03d}_{info["scene_name"]}_{t[:8]}.png'), dpi=90)
        plt.close(fig)

    # summary
    hdr = ['run', 'frames', 'IoU (pooled)', 'IoU mean/frame', 'precision', 'recall'] + \
          [f'IoU {lo}-{hi} m' for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:])]
    rows = []
    for n in names:
        c = tot[n]
        tp, fp, fn = c[0]
        rows.append([n, len(tokens), f'{iou(c[0]):.3f}', f'{np.mean(per_frame[n]):.3f}', f'{tp / max(tp + fp, 1):.3f}',
                     f'{tp / max(tp + fn, 1):.3f}'] + [f'{v:.3f}' for v in iou(c[1:])])
    md = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]
    open(osp.join(args.out, 'summary.md'), 'w').write(
        f'# BEV occupancy predictions ({len(tokens)} val frames, known cells only, threshold P > 0.5)\n\n'
        + '\n'.join(md) + '\n')
    print('\n'.join(md))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    w = 0.8 / len(names)
    labels = [f'{lo}-{hi}' for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:])]
    for c, n in enumerate(names):
        ax.bar(np.arange(len(labels)) + c * w, iou(tot[n][1:]), w, label=n)
    ax.set_xticks(np.arange(len(labels)) + w * (len(names) - 1) / 2, labels)
    ax.set_xlabel('distance from ego (m)')
    ax.set_ylabel('IoU (known cells)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(args.out, 'iou_by_range.png'), dpi=110)


def main():
    args = parse_args()
    runs = dict(r.split('=', 1) for r in args.run)
    if not args.render_only:
        for name, work_dir in runs.items():
            if osp.exists(osp.join(args.out, name, 'meta.json')):
                print(f'[{name}] logits exist, skipping inference')
                continue
            infer(name, work_dir, args)
    render(args, list(runs))


if __name__ == '__main__':
    main()
