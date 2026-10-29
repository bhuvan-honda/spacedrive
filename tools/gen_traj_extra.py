"""Dense waypoints and driving-path labels for the trajectory-format experiments (sd2_traj_e12.py).

Writes data/nuscenes/traj_extra.pkl: {sample token: dict(
    wp12       (12, 2) float32  ego positions every 0.25 s in the current ego frame; the 0.5 s points (odd indices)
                                are copied from gt_planning, the others interpolated from the 20 Hz LIDAR_TOP ego poses
                                at the midpoint between consecutive keyframes,
    wp12_valid (12,) bool       from gt_planning_mask,
    path       (80, 2) float32  future ego positions up to the end of the scene, resampled every 0.5 m of arc length
                                (0.5 .. 40 m); poses within 0.2 m of the previous kept one are skipped (pose jitter
                                while stopped); beyond the recorded future the path continues along the last heading,
    path_real  (80,) bool       False for extrapolated points)}
for every sample of the train and val info files. Run inside the container from the repo root:
    python tools/gen_traj_extra.py
"""
import argparse
import importlib.util
import os
import pickle

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    'traj_format', os.path.join(ROOT, 'projects/mmdet3d_plugin/core/traj_format.py'))
tf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tf)

PATH_POINTS = 80
MIN_STEP = 0.2


def scene_lidar_poses(nusc, scene):
    """Timestamps (s) and ego poses (R, t) of all LIDAR_TOP sample_data (keyframes + sweeps) of a scene."""
    sample = nusc.get('sample', scene['first_sample_token'])
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ts, Rs, ts_xyz = [], [], []
    while True:
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        ts.append(sd['timestamp'] / 1e6)
        Rs.append(Quaternion(pose['rotation']).rotation_matrix)
        ts_xyz.append(pose['translation'])
        if not sd['next']:
            break
        sd = nusc.get('sample_data', sd['next'])
    return np.array(ts), np.array(Rs), np.array(ts_xyz)


def build(info, nusc, poses, by_token):
    R0 = Quaternion(info['ego2global_rotation']).rotation_matrix
    t0 = np.array(info['ego2global_translation'])
    to_ego = lambda p: (np.asarray(p) - t0) @ R0  # rows: R0^T (p - t0)
    ts, Rs, xyz = poses
    t_cur = info['timestamp'] / 1e6

    # keyframe timestamps: current + up to 6 future keyframes
    key_t = [t_cur]
    cur = info
    for _ in range(6):
        cur = by_token.get(cur['next']) if cur['next'] else None
        if cur is None:
            break
        key_t.append(cur['timestamp'] / 1e6)
    mask = info['gt_planning_mask'][0].all(-1)
    wp12 = np.zeros((12, 2), dtype=np.float32)
    valid = np.zeros(12, dtype=bool)
    for k in range(6):
        if not mask[k] or k + 1 >= len(key_t):
            continue
        tm = 0.5 * (key_t[k] + key_t[k + 1])
        p = np.array([np.interp(tm, ts, xyz[:, j]) for j in range(3)])
        wp12[2 * k] = to_ego(p)[:2]
        wp12[2 * k + 1] = info['gt_planning'][0, k, :2]
        valid[2 * k:2 * k + 2] = True

    fut = ts >= t_cur - 1e-3
    pts = to_ego(xyz[fut])[:, :2]
    kept = [np.zeros(2)]
    for p in pts:
        if np.linalg.norm(p - kept[-1]) >= MIN_STEP:
            kept.append(p)
    kept = np.array(kept[1:]).reshape(-1, 2)
    heading = (R0.T @ Rs[fut][-1])[:2, 0]
    s = tf.PATH_RES * np.arange(1, PATH_POINTS + 1)
    path = tf.resample_polyline(kept, s, extrap_dir=heading).astype(np.float32)
    total = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), kept]), axis=0), axis=1).sum() if len(kept) else 0.0
    return dict(wp12=wp12, wp12_valid=valid, path=path, path_real=s <= total + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/nuscenes')
    ap.add_argument('--out', default='data/nuscenes/traj_extra.pkl')
    args = ap.parse_args()
    nusc = NuScenes('v1.0-trainval', args.data_root, verbose=False)
    pose_cache = {}
    out = {}
    for split in ('train', 'val'):
        infos = pickle.load(open(f'{args.data_root}/nuscenes2d_ego_temporal_infos_{split}_with_command_desc.pkl', 'rb'))['infos']
        by_token = {d['token']: d for d in infos}
        proj_err, full20, full40 = [], 0, 0
        for d in infos:
            st = d['scene_token']
            if st not in pose_cache:
                pose_cache[st] = scene_lidar_poses(nusc, nusc.get('scene', st))
            e = build(d, nusc, pose_cache[st], by_token)
            out[d['token']] = e
            full20 += e['path_real'][tf.path_index(1.0, 20)].all()
            full40 += e['path_real'][tf.path_index(2.0, 20)].all()
            if e['wp12_valid'].all() and e['path_real'][tf.path_index(1.0, 20)].all():
                gt6 = d['gt_planning'][0, :, :2]
                proj = tf.project_on_path(e['wp12'], e['path'], tf.half_second_index(12))
                proj_err.append(np.linalg.norm(proj - gt6, axis=1).max())
        n = len(infos)
        print(f'{split}: {n} samples, path real to 20 m: {full20 / n:.1%}, to 40 m: {full40 / n:.1%}; '
              f'GT projection error (max per sample) median {np.median(proj_err):.3f} m, '
              f'p95 {np.percentile(proj_err, 95):.3f} m, max {np.max(proj_err):.3f} m over {len(proj_err)} samples')
    pickle.dump(out, open(args.out, 'wb'))
    print('wrote', args.out, len(out))


if __name__ == '__main__':
    main()
