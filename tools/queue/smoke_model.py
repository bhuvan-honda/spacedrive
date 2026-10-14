"""Single-GPU sanity checks for the SD2 geometry modules (run inside the spacedrive_flash container).

    python tools/queue/smoke_model.py projects/configs/spacedrive/sd2/sd2_full.py [--batch 2] [--test-frames 2]

Checks (exit code 1 on any failure):
  1. epipolar geometry on real camera poses (synthetic cylinder world): two views of the same 3D point agree
  2. token ray directions: FRONT camera centre ray points to +x (ego frame)
  3. zero-init identity: with FiLM / epipolar enabled the training losses equal those with them bypassed
  4. one training forward/backward: all losses finite, every new parameter gets a finite non-zero gradient
     (FiLM / epipolar output layers are zero-initialised so only their last layer gets gradient at step 0),
     FiLM hooks fire, occupancy targets load
  5. test-time generation on a few frames: FiLM fires during prefill, output contains a parsable trajectory
"""
import argparse
import importlib
import math
import sys
from functools import partial

import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter
from mmcv.runner import wrap_fp16_model

sys.path.insert(0, '.')

FAILS = []


def check(cond, msg):
    print(('[ OK ] ' if cond else '[FAIL] ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def batch_from(dataset, indices):
    data = collate([dataset[i] for i in indices], samples_per_gpu=len(indices))
    return scatter(data, [torch.cuda.current_device()])[0]


def check_geometry(lidar2img, H, W):
    from projects.mmdet3d_plugin.models.vlm_utils.epipolar_adapter import DepthEpipolarAdapter
    from projects.mmdet3d_plugin.models.vlm_utils.geometry import project, token_rays
    import torch.nn.functional as F

    l2i = lidar2img[:1].float()
    h = w = 23
    dirs, centres = token_rays(l2i, h, w, H, W)  # (1, N, h, w, 3)
    check(dirs[0, 0, h // 2, w // 2, 0] > 0.9, f'FRONT centre ray ~ +x: {dirs[0, 0, h // 2, w // 2].tolist()}')
    # synthetic world: vertical cylinder of radius R around the ego origin
    R = 15.0
    c = centres[:, :, None, None, :2].expand_as(dirs[..., :2])
    d = dirs[..., :2]
    a = (d * d).sum(-1)
    b = 2 * (c * d).sum(-1)
    cc = (c * c).sum(-1) - R * R
    t = (-b + torch.sqrt((b * b - 4 * a * cc).clamp(min=0))) / (2 * a)
    pts = centres[:, :, None, None] + t[..., None] * dirs
    _, _, z = project(l2i[:, :, None, None], pts)
    depth_tok = z.clamp(min=0.5)
    adapter = DepthEpipolarAdapter(16, inner=16, heads=2, scales=(1.0,)).cuda()
    grid, pz, valid = adapter._geometry(depth_tok, l2i, H, W)
    B, N = depth_tok.shape[:2]
    M = grid.shape[-2]
    g = grid.permute(0, 5, 1, 2, 3, 4, 6).reshape(B * M, 1, -1, 2)
    d_tgt = F.grid_sample(depth_tok.reshape(B * N, 1, h, w), g, mode='nearest', padding_mode='border',
                          align_corners=False).view(B, M, N, h, w, 1).permute(0, 2, 3, 4, 5, 1)
    err = (pz.clamp(min=0.5).log() - d_tgt.clamp(min=0.5).log()).abs()[valid]
    shuffled = depth_tok[:, torch.randperm(N)]
    d_shuf = F.grid_sample(shuffled.reshape(B * N, 1, h, w), g, mode='nearest', padding_mode='border',
                           align_corners=False).view(B, M, N, h, w, 1).permute(0, 2, 3, 4, 5, 1)
    err_shuf = (pz.clamp(min=0.5).log() - d_shuf.clamp(min=0.5).log()).abs()[valid]
    frac = valid.any(-1).float().mean().item()
    check(valid.sum() > 0 and err.median() < 0.05,
          f'epipolar consistency on cylinder: median |dlog z| {err.median():.4f} '
          f'(shuffled cams {err_shuf.median():.3f}), tokens with a valid view {frac:.3f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--test-frames', type=int, default=2)
    p.add_argument('--load', help='checkpoint to load (non-strict) before the checks')
    p.add_argument('--expect-traj', action='store_true')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = p.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    importlib.import_module('projects.mmdet3d_plugin')
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.core.hook.gen_l2_hook import parse_traj

    torch.manual_seed(0)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    if args.load:
        from mmcv.runner import load_checkpoint
        load_checkpoint(model, args.load, map_location='cpu', strict=False)
    model = model.cuda()
    wrap_fp16_model(model)
    new_mods = {n: m for n, m in [('bev_occ_head', model.bev_occ_head), ('depth_film', model.depth_film),
                                  ('epipolar_adapter', model.epipolar_adapter)] if m is not None}
    for n, m in new_mods.items():
        print(f'{n}: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params')

    train_cfg = cfg.data.train.copy()
    train_cfg['pipeline'] = [t for t in train_cfg['pipeline'] if not t['type'].startswith('LoadPoints')]
    ds = build_dataset(train_cfg)
    idx = [i for i, info in enumerate(ds.data_infos) if 'gt_planning' in info
           and info['gt_planning_mask'].any(axis=-1).sum() == 6][100:100 + args.batch]
    data = batch_from(ds, idx)
    H, W = data['img'].shape[-2:]

    check_geometry(data['lidar2img'], H, W)

    mem_attrs = ('memory_canbus', 'memory_egopose', 'sample_time', 'memory_count')

    def snapshot():
        return {k: getattr(model, k).clone() if torch.is_tensor(getattr(model, k)) else getattr(model, k)
                for k in mem_attrs if hasattr(model, k)}

    def restore(s):
        for k, v in s.items():
            setattr(model, k, v)

    # 3. zero-init identity (eval mode, no grad)
    if model.depth_film is not None or model.epipolar_adapter is not None:
        model.eval()
        s = snapshot()
        with torch.no_grad():
            l_on = model(return_loss=True, **data)
            restore(s)
            orig = model._setup_geo_modules
            model._setup_geo_modules = lambda *a, **k: None
            l_off = model(return_loss=True, **data)
            model._setup_geo_modules = orig
        restore(s)
        for k in ('vlm_loss', 'loss_pos'):
            a, b = float(l_on[k]), float(l_off[k])
            check(abs(a - b) <= 1e-3 * max(1.0, abs(b)), f'identity at init {k}: on {a:.5f} off {b:.5f}')

    # 4. training step
    model.train()
    torch.cuda.reset_peak_memory_stats()
    hits0 = model.depth_film.hits if model.depth_film is not None else 0
    losses = model(return_loss=True, **data)
    total = sum(v for k, v in losses.items() if 'loss' in k)
    print({k: round(float(v), 5) for k, v in losses.items()}, flush=True)
    check(all(torch.isfinite(torch.as_tensor(v)).all() for v in losses.values()), 'all logged values finite')
    check('tf_ade' in losses and 'tf_fde' in losses, f"tf_ade {float(losses.get('tf_ade', -1)):.3f} m")
    if model.bev_occ_head is not None:
        check('loss_occ' in losses, f"loss_occ {float(losses['loss_occ']):.4f} occ_iou {float(losses['occ_iou']):.4f}")
    if model.epipolar_adapter is not None:
        check(float(losses['epi_valid_frac']) > 0.1, f"epi_valid_frac {float(losses['epi_valid_frac']):.3f}")
    total.backward()
    if model.depth_film is not None:
        hits = model.depth_film.hits - hits0
        # forward + recompute in backward (reentrant gradient checkpointing)
        check(hits == 2 * len(model.depth_film.layers), f'FiLM hooks fired {hits}x in fwd+bwd')
    print(f'peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB (batch {args.batch})', flush=True)
    for n, m in new_mods.items():
        no_grad, zero = [], []
        for pn, prm in m.named_parameters():
            if prm.grad is None:
                no_grad.append(pn)
            elif not torch.isfinite(prm.grad).all():
                check(False, f'{n}.{pn} non-finite grad')
            elif prm.grad.abs().sum() == 0:
                zero.append(pn)
        if n == 'bev_occ_head':
            check(not no_grad and not zero, f'{n}: all params get grad (none={no_grad[:4]}, zero={zero[:4]})')
        else:
            out_layer = [pn for pn in [x for x, _ in m.named_parameters()] if pn.startswith(('out.', 'nets.'))]
            ok_last = all(pn not in no_grad for pn in out_layer)
            check(ok_last and len(no_grad) < len(list(m.parameters())),
                  f'{n}: grads reach params (no grad: {len(no_grad)}, zero grad at init: {len(zero)})')
    lora = [prm for pn, prm in model.named_parameters() if 'lora' in pn and prm.requires_grad]
    check(len(lora) > 0 and any(prm.grad is not None and prm.grad.abs().sum() > 0 for prm in lora), 'LoRA gets grad')
    model.zero_grad(set_to_none=True)
    del losses, total
    torch.cuda.empty_cache()

    # 5. test-time generation
    test_cfg = cfg.data.test.copy()
    test_cfg['test_mode'] = True
    test_cfg['pipeline'] = [t for t in test_cfg['pipeline'] if not t['type'].startswith('LoadPoints')]
    tds = build_dataset(test_cfg)
    model.eval()
    model.save_path = '/tmp/sd2_smoke_gen/'
    if hasattr(model, 'reset_memory'):
        model.reset_memory()
    n_ok = 0
    with torch.no_grad():
        for i in range(args.test_frames):
            hits0 = model.depth_film.hits if model.depth_film is not None else 0
            out = model(return_loss=False, rescale=True, **batch_from(tds, [i]))
            ans = out[0]['A'] if out else None
            traj = parse_traj(ans)
            print(f'test frame {i}: {str(ans)[:160]}', flush=True)
            n_ok += traj is not None and len(traj) >= 6
            if model.depth_film is not None:
                check(model.depth_film.hits - hits0 == len(model.depth_film.layers),
                      f'FiLM fired once per layer in generation ({model.depth_film.hits - hits0})')
    if args.expect_traj:  # only meaningful for a trained checkpoint (--load); untrained LoRA answers in prose
        check(n_ok == args.test_frames, f'parsable 6-waypoint trajectories {n_ok}/{args.test_frames}')
    else:
        print(f'parsable 6-waypoint trajectories {n_ok}/{args.test_frames} (not checked)')

    print('SMOKE ' + ('PASSED' if not FAILS else f'FAILED: {FAILS}'), flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
