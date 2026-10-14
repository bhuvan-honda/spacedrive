# No-learning planning baselines. Writes prediction files in the format parsed by
# eval_planning.py.
#  - extrap_h{N}: constant velocity from a least-squares fit (anchored at the
#    current position) to the last N historical keyframe positions.
#  - ctrv: constant turn rate and velocity from the current CAN bus state
#    (can_bus[10] = forward speed m/s, can_bus[9] = yaw rate rad/s; verified
#    against pose-derived values on nuScenes val, corr 0.998 / 0.983).
import argparse
import json
import os
import pickle

import numpy as np
from pyquaternion import Quaternion

FUT_T = 0.5 * np.arange(1, 7)


def ego2global(info):
    mat = np.eye(4)
    mat[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
    mat[:3, 3] = info['ego2global_translation']
    return mat


def write_pred(out_dir, token, traj, tag):
    pts = ', '.join(f'<POS_INDICATOR>({x:.4f}, {y:.4f})' for x, y in traj)
    with open(os.path.join(out_dir, token), 'w') as f:
        json.dump([{'Q': f'{tag} baseline', 'A': f'Here is the planning trajectory [{pts}].'}], f)


def extrapolate_history(info, infos, token2idx, n_hist):
    g2e = np.linalg.inv(ego2global(info))
    ts, xy = [], []
    cur = info
    for _ in range(n_hist):
        if not cur['prev'] or cur['prev'] not in token2idx:
            break
        cur = infos[token2idx[cur['prev']]]
        if cur['scene_token'] != info['scene_token']:
            break
        ts.append((cur['timestamp'] - info['timestamp']) * 1e-6)
        xy.append((g2e @ ego2global(cur))[:2, 3])
    if ts:
        ts, xy = np.array(ts), np.array(xy)
        vel = (ts[:, None] * xy).sum(0) / (ts ** 2).sum()
    else:
        vel = np.zeros(2)
    return FUT_T[:, None] * vel[None], len(ts)


def ctrv(info):
    v, w = info['can_bus'][10], info['can_bus'][9]
    if abs(w) < 1e-4:
        return np.stack([v * FUT_T, np.zeros_like(FUT_T)], -1)
    return np.stack([v / w * np.sin(w * FUT_T), v / w * (1 - np.cos(w * FUT_T))], -1)


def run(name, infos, predict, out_root):
    out_dir = os.path.join(out_root, f'spacedrive_baseline_{name}', '_results_planning_only')
    os.makedirs(out_dir, exist_ok=True)
    for info in infos:
        write_pred(out_dir, info['token'], predict(info), name)
    print(f'{name}: wrote {len(infos)} preds to {out_dir}')


def main(args):
    infos = pickle.load(open(os.path.join(args.base_path, args.anno_path), 'rb'))['infos']
    token2idx = {info['token']: i for i, info in enumerate(infos)}

    for n_hist in args.history:
        run(f'extrap_h{n_hist}', infos,
            lambda info: extrapolate_history(info, infos, token2idx, n_hist)[0], args.out_root)
    if args.ctrv:
        run('ctrv', infos, ctrv, args.out_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default='data/nuscenes/')
    parser.add_argument('--anno_path', default='nuscenes2d_ego_temporal_infos_val.pkl')
    parser.add_argument('--out_root', default='results/')
    parser.add_argument('--history', type=int, nargs='*', default=[3, 6])
    parser.add_argument('--ctrv', action='store_true', help='also write the CAN-bus CTRV baseline')
    main(parser.parse_args())
