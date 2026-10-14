#!/usr/bin/env python3
# -------------------------------------------------------------------------
# SpaceDrive+ — inference on ONE nuScenes val scene -> annotated CAM_FRONT MP4
# -------------------------------------------------------------------------
# Sanity-check run on the data the model was trained/evaluated on. Uses the
# repo's own val dataset + test pipeline (so ego_pose / can_bus / command /
# command_desc / intrinsics / extrinsics are exactly what dist_test feeds the
# model), runs consecutive keyframes of a single scene in order on one model
# instance (ego-status memory is temporally meaningful), and draws the
# predicted trajectory (and the GT trajectory) on the native CAM_FRONT image.
#
# nuScenes ego frame origin is at road level, so waypoints are projected at z=0.
#
# Usage:
#   python tools/infer_nuscenes_video.py --scene-index 0 --device cuda:4
# -------------------------------------------------------------------------

import os
import sys
import re
import json
import argparse
from functools import partial

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from nuscenes.eval.common.utils import Quaternion

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'projects'))
import projects.mmdet3d_plugin  # noqa: F401

from infer_yui_no_mori import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='projects/configs/spacedrive/spacedrive_plus_qwen_depth_orth.py')
    p.add_argument('--ckpt', default='results/spacedrive_plus_qwen_depth_orth_hard/latest.pth')
    p.add_argument('--depth-skip', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=1)
    p.add_argument('--depth-skip-mode', default='add', choices=['concat', 'add'])
    p.add_argument('--depth-orth-mode', default=None, choices=['none', 'hard', 'soft'])
    p.add_argument('--scene-index', type=int, default=0, help="Nth scene (in val pkl order).")
    p.add_argument('--start-frame', type=int, default=0, help="Offset within the scene.")
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--fps', type=float, default=10.0, help="nuScenes keyframes are 2 Hz.")
    p.add_argument('--output-dir', default='./infer_nuscenes_output')
    p.add_argument('--device', default='cuda:7')
    p.add_argument('--workers', type=int, default=2)
    return p.parse_args()


def cam_front_calib(info):
    cam = info['cams']['CAM_FRONT']
    R = Quaternion(cam['sensor2ego_rotation']).rotation_matrix
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = R
    cam2ego[:3, 3] = cam['sensor2ego_translation']
    ego2cam = np.linalg.inv(cam2ego)
    K = np.array(cam['cam_intrinsic'], dtype=np.float64)
    return K, ego2cam, cam['data_path']


def project(pts_xy, K, E, z=0.0):
    out = []
    for x, y in pts_xy:
        pc = E @ np.array([x, y, z, 1.0])
        if pc[2] <= 0.1:
            out.append(None)
            continue
        uv = K @ pc[:3]
        out.append((uv[0] / uv[2], uv[1] / uv[2]))
    return out


def draw(img, pred_uv, pred_xy, gt_uv, text):
    h, w = img.shape[:2]
    r = max(6, w // 160)
    for uv in gt_uv:
        if uv is not None:
            cv2.circle(img, (int(uv[0]), int(uv[1])), r, (255, 128, 0), 2)  # GT: blue ring (BGR)
    n = len(pred_uv)
    for i, (uv, (x, y)) in enumerate(zip(pred_uv, pred_xy)):
        if uv is None:
            continue
        t = i / max(n - 1, 1)
        col = (0, int(255 * (1 - t)), int(255 * t))  # BGR: green -> red
        p = (int(uv[0]), int(uv[1]))
        cv2.circle(img, p, r, col, -1)
        cv2.circle(img, p, r, (255, 255, 255), 1)
        cv2.putText(img, f"({x:.1f},{y:.1f})", (p[0] + r + 2, p[1] - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    cv2.rectangle(img, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(img, text + "   [filled=pred, blue ring=GT]", (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return img


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device_id = int(args.device.split(':')[-1]) if ':' in args.device else 0
    torch.cuda.set_device(device_id)

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    cfg = load_config(cfg_path)
    for k, v in (('depth_skip', args.depth_skip), ('depth_skip_mode', args.depth_skip_mode),
                 ('depth_orth_mode', args.depth_orth_mode)):
        if v is not None:
            cfg.model[k] = v
    cfg.model.save_path = os.path.join(args.output_dir, '_model_json/')
    os.makedirs(cfg.model.save_path, exist_ok=True)

    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    dataset = build_dataset(cfg.data.test)

    # pick the scene
    scenes = []
    for i, info in enumerate(dataset.data_infos):
        if not scenes or scenes[-1][0] != info['scene_token']:
            scenes.append((info['scene_token'], []))
        scenes[-1][1].append(i)
    token, idxs = scenes[args.scene_index]
    idxs = idxs[args.start_frame:]
    if args.max_frames:
        idxs = idxs[:args.max_frames]
    print(f"[INFO] {len(scenes)} val scenes; using scene {args.scene_index} ({token}), {len(idxs)} frames")

    loader = DataLoader(Subset(dataset, idxs), batch_size=1, shuffle=False,
                        num_workers=args.workers, collate_fn=partial(collate, samples_per_gpu=1))

    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    ck = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PROJECT_ROOT, args.ckpt)
    print(f"[INFO] Loading checkpoint {ck}")
    load_checkpoint(model, ck, map_location='cpu', strict=False)
    model = MMDataParallel(model.cuda(device_id), device_ids=[device_id])
    model.eval()
    if getattr(model.module, 'ego_status', None) is not None:
        model.module.reset_memory()

    frames_dir = os.path.join(args.output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    results, writer = [], None
    video_path = os.path.join(args.output_dir, 'waypoints_video.mp4')

    for step, (data, ds_idx) in enumerate(zip(loader, idxs)):
        with torch.no_grad():
            out = model(return_loss=False, rescale=True, **data)
        r = out[0] if isinstance(out, list) and out else out
        ans = r.get('A', '') if isinstance(r, dict) else str(r)
        if isinstance(ans, list):
            ans = ans[0] if ans else ''
        pred = [(float(a), float(b)) for a, b in re.findall(
            r'\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)', ans)]

        info = dataset.data_infos[ds_idx]
        K, E, path = cam_front_calib(info)
        img = cv2.imread(path)
        gt = info.get('gt_planning')
        gt_xy = [tuple(map(float, p[:2])) for p in gt[0]] if gt is not None else []
        text = f"scene {args.scene_index} frame {step}  cmd: {info['gt_planning_command_desc'].strip()}"
        img = draw(img, project(pred, K, E), pred, project(gt_xy, K, E), text)
        cv2.imwrite(os.path.join(frames_dir, f"{step:06d}.jpg"), img)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
        writer.write(img)
        l2 = float(np.mean([np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(pred, gt_xy)])) if pred and gt_xy else None
        results.append(dict(step=step, token=info['token'], answer=ans, pred=pred, gt=gt_xy, mean_l2=l2))
        print(f"[{step + 1}/{len(idxs)}] {info['token']} pred={len(pred)} mean_l2={l2}")

    if writer is not None:
        writer.release()
        print(f"[INFO] Video: {video_path}")
    with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    l2s = [x['mean_l2'] for x in results if x['mean_l2'] is not None]
    if l2s:
        print(f"[INFO] mean L2 over scene: {np.mean(l2s):.3f} m")


if __name__ == '__main__':
    main()
