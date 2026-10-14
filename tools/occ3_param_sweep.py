"""Parameter sweep for tools/gen_bev_occ.py free-space labelling (diagnostic, not used by training).

    python tools/occ3_param_sweep.py --frames 200 --variants "free_hmax=1.0" "free_hmax=1.5" ...

Per variant: unknown / free / occupied fractions, GT waypoints (outside the ego footprint) on unknown, and of
those the share NOT occluded by an occupied cell (coverage gaps); cells inside GT boxes labelled free (lower is
better), and box interiors more than 0.5 m from the box edge labelled free.
"""
import argparse
import pickle
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, 'tools')
import gen_bev_occ as g  # noqa: E402


def stats(item):
    info, kv = item
    sys.argv = ['x']
    a = g.parse_args()
    for k, v in kv.items():
        setattr(a, k, type(getattr(a, k))(v))
    li = g.light(info)
    occ, known = g.occupancy(*g.ego_points(li, a.sweep_window), a)
    lab = g.decode(occ, known)
    R, res, N = a.range, a.res, lab.shape[0]
    r = np.zeros(8)
    r[:3] = np.bincount(lab.ravel(), minlength=3) / lab.size
    wp = info['gt_planning'][0, :, :2][info['gt_planning_mask'][0].any(-1).astype(bool)]
    o = np.asarray(info['lidar2ego_translation'])[:2]
    for x, y in wp:
        i, j = int((x + R) / res), int((y + R) / res)
        if g.ego_body(np.array(x), np.array(y), a) or not (0 <= i < N and 0 <= j < N):
            continue
        r[3] += 1
        if lab[i, j] == g.UNKNOWN:
            r[4] += 1
            t = np.linspace(0, 1, 400)[:-4]
            ii = ((o[0] + t * (x - o[0]) + R) / res).astype(int)
            jj = ((o[1] + t * (y - o[1]) + R) / res).astype(int)
            r[5] += not (lab[ii, jj] == g.OCC).any()
    _, _, cells, free = g.box_stats(lab, li, a)
    r[6:8] = cells, free
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=int, default=200)
    p.add_argument('--split', default='train')
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--variants', nargs='+', default=['free_hmax=1.0'])
    args = p.parse_args()
    infos = pickle.load(open(f'data/nuscenes/nuscenes2d_ego_temporal_infos_{args.split}.pkl', 'rb'))['infos']
    infos = [infos[i] for i in np.linspace(0, len(infos) - 1, args.frames).astype(int)]
    print(f'{"variant":45s} unk   free  occ  | wp_unk  wp_gap | box_free')
    with Pool(args.workers) as pool:
        for var in args.variants:
            kv = dict(s.split('=') for s in var.split(',') if s)
            r = np.array(pool.map(stats, [(info, kv) for info in infos]))
            s = r.sum(0)
            print(f'{var:45s} {r[:, 2].mean():.3f} {r[:, 0].mean():.3f} {r[:, 1].mean():.3f} | '
                  f'{s[4] / s[3]:.3f}  {s[5] / s[3]:.3f} | {s[7] / s[6]:.3f}', flush=True)


if __name__ == '__main__':
    main()
