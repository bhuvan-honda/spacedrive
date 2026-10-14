"""Offline 3-class 2D BEV occupancy ground truth (free / occupied / unknown) from nuScenes LiDAR (CPU only).

For every keyframe in the info pkls, the keyframe LiDAR scan plus the sweeps within `--sweep-window`
seconds are moved into the keyframe EGO frame (x forward, y left, z up). This is the same frame as
`lidar2img` / the 3D PE (the dataset builds lidar2img from the camera sensor2ego) and `gt_planning`.
Ego-body returns are dropped and a local ground height is estimated. Then
  - OCCUPIED: cells with >= `--min-pts` points at height in [hmin, hmax] above the local ground;
  - FREE: cells crossed by a LiDAR ray (from the scan's own sensor origin to its return) at a height of at
    most `--free-hmax` above the local ground (a low ray proves there is no obstacle there; a ray passing
    high above a car does not), excluding the last `res` metres before a non-ground return;
  - UNKNOWN: everything else (occluded, out of LiDAR reach, and the ego footprint).
Occupied wins over free.

Output: <out>/<token>.npy = np.packbits(np.concatenate([occ.ravel(), known.ravel()])), occ / known are
(N, N) {0, 1} with grid[i, j], i = floor((x + R) / res) (forward axis), j = floor((y + R) / res) (left axis).
Label = 1 (occupied) if occ, else 0 (free) if known, else 2 (unknown); see `decode`. Plus <out>/meta.json.
Tokens are unique across splits, so one flat directory. Resumable: existing files are skipped unless
--overwrite.

    python tools/gen_bev_occ.py --splits train val --workers 32 --vis 12
    python tools/gen_bev_occ.py --splits val --limit 200 --vis 12 --out /tmp/bev_occ_test   # sanity run
"""

FREE, OCC, UNKNOWN = 0, 1, 2
import argparse
import json
import os
import os.path as osp
import pickle
import time
from multiprocessing import Pool

import numpy as np
from pyquaternion import Quaternion

ARGS = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data/nuscenes')
    p.add_argument('--splits', nargs='+', default=['train', 'val'])
    p.add_argument('--out', default='data/nuscenes/bev_occ3')
    p.add_argument('--range', type=float, default=50.0, help='grid covers [-R, R) on x and y (m)')
    p.add_argument('--res', type=float, default=0.5, help='cell size (m)')
    p.add_argument('--sweep-window', type=float, default=0.15, help='max age of sweeps (s); 0 = keyframe only')
    p.add_argument('--hmin', type=float, default=0.3, help='min height above local ground (m)')
    p.add_argument('--hmax', type=float, default=3.0, help='max height above local ground (m)')
    p.add_argument('--ground-cell', type=float, default=2.0, help='ground estimation cell size (m)')
    p.add_argument('--ground-nb', type=int, default=1, help='ground = min over (2k+1)^2 ground cells')
    p.add_argument('--ground-q', type=float, default=0.2, help='per-cell height quantile used as ground')
    p.add_argument('--min-pts', type=int, default=1)
    p.add_argument('--free-hmax', type=float, default=1.0,
                   help='a ray marks a cell free only where it is at most this high above the local ground (m)')
    p.add_argument('--free-hmin', type=float, default=0.3,
                   help='... and at least this high (a ray under a vehicle chassis proves nothing) (m)')
    p.add_argument('--ray-step', type=float, default=0.25, help='ray sampling step (m, horizontal)')
    p.add_argument('--ego-x', type=float, nargs=2, default=[-1.2, 3.8], help='ego body x extent (ego frame)')
    p.add_argument('--ego-hw', type=float, default=1.2, help='ego body half width (m)')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--limit', type=int, default=0, help='only the first N infos per split (testing)')
    p.add_argument('--vis', type=int, default=0, help='write N PNG overlays per split to <out>/vis')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def load_xyz(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)


def ego_points(info, window):
    """Returns points (P, 3) and the sensor origin of each point's scan (P, 3), both in the keyframe ego frame."""
    pts = [load_xyz(info['lidar_path'])]
    org = [np.zeros((1, 3))]
    for sw in info['sweeps']:
        if (info['timestamp'] - sw['timestamp']) / 1e6 > window + 1e-3:
            continue
        p = load_xyz(sw['data_path'])
        t = np.asarray(sw['sensor2lidar_translation'])
        pts.append(p @ np.asarray(sw['sensor2lidar_rotation']).T + t)
        org.append(t[None])
    org = np.concatenate([np.repeat(o, len(p), 0) for o, p in zip(org, pts)], 0)
    pts = np.concatenate(pts, 0)
    return pts @ info['l2e_R'].T + info['l2e_t'], org @ info['l2e_R'].T + info['l2e_t']


def ground_map(x, y, z, a):
    """Local ground height per ground cell (G, G): a low height quantile per cell, then the min over a
    (2k+1)^2 neighbourhood (cells fully covered by an object borrow the ground of their neighbours).
    Cells without any nearby return fall back to z = 0 (ego-frame ground plane)."""
    R, g = a.range, a.ground_cell
    G = int(round(2 * R / g))
    gi = np.clip(((x + R) / g).astype(np.int64), 0, G - 1)
    gj = np.clip(((y + R) / g).astype(np.int64), 0, G - 1)
    cell = gi * G + gj
    order = np.lexsort((z, cell))
    counts = np.bincount(cell, minlength=G * G)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    q = np.full(G * G, np.inf)
    nz = counts > 0
    q[nz] = z[order][starts[nz] + (a.ground_q * (counts[nz] - 1)).astype(np.int64)]
    q = q.reshape(G, G)
    k = a.ground_nb
    pad = np.pad(q, k, constant_values=np.inf)
    nb = np.full_like(q, np.inf)
    for di in range(2 * k + 1):
        for dj in range(2 * k + 1):
            nb = np.minimum(nb, pad[di:di + G, dj:dj + G])
    nb[np.isinf(nb)] = 0.0
    return nb


def ground_at(gm, x, y, a):
    G = gm.shape[0]
    gi = np.clip(((x + a.range) / a.ground_cell).astype(np.int64), 0, G - 1)
    gj = np.clip(((y + a.range) / a.ground_cell).astype(np.int64), 0, G - 1)
    return gm[gi, gj]


def ego_body(x, y, a):
    return (x > a.ego_x[0]) & (x < a.ego_x[1]) & (np.abs(y) < a.ego_hw)


def free_space(pts, org, gm, a, N, max_samples=4_000_000):
    """(N*N,) bool: cells crossed by a ray at <= free_hmax above the local ground."""
    R, res, step = a.range, a.res, a.ray_step
    d = pts - org
    L = np.hypot(d[:, 0], d[:, 1])
    h_end = pts[:, 2] - ground_at(gm, pts[:, 0], pts[:, 1], a)
    stop = np.where(h_end < a.hmin, L, L - res)          # ground return: its own cell is observed free
    stop = np.minimum(stop, 1.5 * R)                      # beyond this every sample is outside the grid
    n = np.where(stop >= 0, np.floor(stop / step).astype(np.int64) + 1, 0)
    known = np.zeros(N * N, bool)
    cum = np.cumsum(n)
    r0 = 0
    while r0 < len(n):
        base = cum[r0 - 1] if r0 else 0
        r1 = max(int(np.searchsorted(cum, base + max_samples, side='right')), r0 + 1)
        nn = n[r0:r1]
        ray = np.repeat(np.arange(r0, r1), nn)
        k = np.arange(nn.sum()) - np.repeat(cum[r0:r1] - nn - base, nn)
        t = k * step / np.maximum(L[ray], 1e-6)
        sx = org[ray, 0] + t * d[ray, 0]
        sy = org[ray, 1] + t * d[ray, 1]
        sz = org[ray, 2] + t * d[ray, 2]
        ok = (sx >= -R) & (sx < R) & (sy >= -R) & (sy < R)
        sx, sy, sz = sx[ok], sy[ok], sz[ok]
        hz = sz - ground_at(gm, sx, sy, a)
        ok = (hz <= a.free_hmax) & (hz >= a.free_hmin)
        i = ((sx[ok] + R) / res).astype(np.int64).clip(0, N - 1)
        j = ((sy[ok] + R) / res).astype(np.int64).clip(0, N - 1)
        known[i * N + j] = True
        r0 = r1
    return known


def occupancy(pts, org, a):
    """Returns occ, known: (N, N) uint8 each."""
    R, res = a.range, a.res
    N = int(round(2 * R / res))
    keep = ~ego_body(pts[:, 0], pts[:, 1], a)
    pts, org = pts[keep], org[keep]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    inr = (x >= -R) & (x < R) & (y >= -R) & (y < R)
    gm = ground_map(x[inr], y[inr], z[inr], a)
    x, y, z = x[inr], y[inr], z[inr]
    h = z - ground_at(gm, x, y, a)
    sel = (h > a.hmin) & (h < a.hmax)
    i = ((x[sel] + R) / res).astype(np.int64).clip(0, N - 1)
    j = ((y[sel] + R) / res).astype(np.int64).clip(0, N - 1)
    occ = np.bincount(i * N + j, minlength=N * N) >= a.min_pts
    known = free_space(pts, org, gm, a, N) | occ
    c = (np.arange(N) + 0.5) * res - R
    known[ego_body(c[:, None], c[None, :], a).ravel()] = False
    return occ.astype(np.uint8).reshape(N, N), known.astype(np.uint8).reshape(N, N)


def decode(occ, known):
    return np.where(occ > 0, OCC, np.where(known > 0, FREE, UNKNOWN)).astype(np.uint8)


def boxes_ego(info):
    """gt boxes (x y z l w h yaw) are stored in the EGO frame in these infos (verified: centre coverage
    0.84 as-is vs 0.32 after an extra lidar->ego transform) -> (cx, cy, l, w, yaw)."""
    b = info['gt_boxes']
    return b[:, 0], b[:, 1], b[:, 3], b[:, 4], b[:, 6]


def box_stats(label, info, a, margin=0.25):
    """Boxes with >= 5 lidar pts and centre within 0.9 R: (#boxes, #boxes with >= 1 occupied cell centre inside
    the footprint inflated by `margin`, #cells strictly inside the footprints, #of those labelled free)."""
    if len(info['gt_boxes']) == 0:
        return 0, 0, 0, 0
    R, res = a.range, a.res
    N = label.shape[0]
    cx, cy, ls, ws, yaws = boxes_ego(info)
    ok = (info['num_lidar_pts'] >= 5) & (np.abs(cx) < 0.9 * R) & (np.abs(cy) < 0.9 * R)
    hit = cells = free = 0
    for x, y, l, w, yaw in zip(cx[ok], cy[ok], ls[ok], ws[ok], yaws[ok]):
        r = int(np.ceil((np.hypot(l, w) / 2 + margin) / res))
        i, j = int((x + R) / res), int((y + R) / res)
        i0, i1, j0, j1 = max(i - r, 0), min(i + r + 1, N), max(j - r, 0), min(j + r + 1, N)
        ii, jj = np.mgrid[i0:i1, j0:j1]
        dx = (ii + 0.5) * res - R - x
        dy = (jj + 0.5) * res - R - y
        u = dx * np.cos(yaw) + dy * np.sin(yaw)
        v = -dx * np.sin(yaw) + dy * np.cos(yaw)
        lab = label[i0:i1, j0:j1]
        hit += bool(((np.abs(u) < l / 2 + margin) & (np.abs(v) < w / 2 + margin) & (lab == OCC)).any())
        inside = (np.abs(u) < l / 2) & (np.abs(v) < w / 2)
        cells += int(inside.sum())
        free += int((inside & (lab == FREE)).sum())
    return int(ok.sum()), int(hit), cells, free


def waypoint_labels(label, info, a):
    """Counts of the labels under the valid GT future waypoints outside the ego footprint (should be almost
    all free; waypoints inside the footprint of slow / stopped frames are unknown by construction)."""
    out = np.zeros(3, np.int64)
    if info.get('gt_planning') is None:
        return out
    R, res, N = a.range, a.res, label.shape[0]
    wp = info['gt_planning'][0, :, :2][info['gt_planning_mask'][0].any(-1).astype(bool)]
    wp = wp[~ego_body(wp[:, 0], wp[:, 1], a)]
    i = ((wp[:, 0] + R) / res).astype(np.int64)
    j = ((wp[:, 1] + R) / res).astype(np.int64)
    ok = (i >= 0) & (i < N) & (j >= 0) & (j < N)
    np.add.at(out, label[i[ok], j[ok]], 1)
    return out


COLORS = {FREE: (110, 110, 110), OCC: (255, 255, 255), UNKNOWN: (0, 0, 0)}  # BGR


def draw(label, info, a, path):
    import cv2
    N, s, R = label.shape[0], 4, a.range
    img = np.zeros((N * s, N * s, 3), np.uint8)
    # image row = forward (up), image col = left (left): row = N-1-i, col = N-1-j
    lab = label[::-1, ::-1].repeat(s, 0).repeat(s, 1)
    for k, c in COLORS.items():
        img[lab == k] = c

    def to_px(x, y):
        return ((R - y) / a.res * s, (R - x) / a.res * s)  # (col, row)

    if len(info['gt_boxes']):
        for x, y, l, w, yaw in zip(*boxes_ego(info)):
            corners = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]) / 2
            rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            cw = corners @ rot.T + [x, y]
            px = np.array([to_px(*p) for p in cw], np.int32)
            cv2.polylines(img, [px], True, (0, 200, 0), 1)
    ego = np.array([to_px(x, y) for x, y in [(a.ego_x[1], a.ego_hw), (a.ego_x[1], -a.ego_hw),
                                             (a.ego_x[0], -a.ego_hw), (a.ego_x[0], a.ego_hw)]], np.int32)
    cv2.polylines(img, [ego], True, (0, 0, 255), 2)
    if info.get('gt_planning') is not None:
        for x, y in info['gt_planning'][0, :, :2]:
            cv2.circle(img, tuple(int(v) for v in to_px(x, y)), 4, (255, 128, 0), -1)
    cv2.putText(img, info['token'][:8], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    legend = [('occupied', COLORS[OCC]), ('free', COLORS[FREE]), ('unknown', (60, 60, 60)),
              ('GT box', (0, 200, 0)), ('ego', (0, 0, 255)), ('GT future', (255, 128, 0))]
    for k, (txt, c) in enumerate(legend):
        y0 = N * s - 20 - 22 * k
        cv2.rectangle(img, (5, y0 - 12), (19, y0 + 2), c, -1)
        cv2.rectangle(img, (5, y0 - 12), (19, y0 + 2), (0, 255, 255), 1)
        cv2.putText(img, txt, (25, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(path, img)


def work(item):
    info, vis_path = item
    a = ARGS
    out = osp.join(a.out, info['token'] + '.npy')
    if osp.exists(out) and not a.overwrite and not vis_path:
        return None
    occ, known = occupancy(*ego_points(info, a.sweep_window), a)
    np.save(out, np.packbits(np.concatenate([occ.ravel(), known.ravel()])))
    label = decode(occ, known)
    if vis_path:
        draw(label, info, a, vis_path)
    fracs = np.bincount(label.ravel(), minlength=3) / label.size
    return fracs, np.array(box_stats(label, info, a)), waypoint_labels(label, info, a)


def light(info):
    return dict(
        token=info['token'], lidar_path=info['lidar_path'], timestamp=info['timestamp'],
        sweeps=[dict(data_path=s['data_path'], timestamp=s['timestamp'],
                     sensor2lidar_rotation=s['sensor2lidar_rotation'],
                     sensor2lidar_translation=s['sensor2lidar_translation']) for s in info['sweeps']],
        l2e_R=Quaternion(info['lidar2ego_rotation']).rotation_matrix,
        l2e_t=np.asarray(info['lidar2ego_translation']),
        gt_boxes=info['gt_boxes'], num_lidar_pts=info['num_lidar_pts'],
        gt_planning=info.get('gt_planning'), gt_planning_mask=info.get('gt_planning_mask'))


def main():
    global ARGS
    ARGS = a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = {k: v for k, v in vars(a).items() if k not in ('workers', 'limit', 'vis', 'overwrite', 'splits')}
    meta.update(frame='ego (x forward, y left)', layout='grid[i, j], i=floor((x+R)/res), j=floor((y+R)/res)',
                shape=[int(round(2 * a.range / a.res))] * 2,
                storage='np.packbits(np.concatenate([occ.ravel(), known.ravel()]))',
                labels=dict(free=FREE, occupied=OCC, unknown=UNKNOWN))
    with open(osp.join(a.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    if a.vis:
        os.makedirs(osp.join(a.out, 'vis'), exist_ok=True)

    for split in a.splits:
        infos = pickle.load(open(osp.join(a.data_root, f'nuscenes2d_ego_temporal_infos_{split}.pkl'), 'rb'))['infos']
        if a.limit:
            infos = infos[:a.limit]
        vis_idx = set(np.linspace(0, len(infos) - 1, a.vis).astype(int).tolist()) if a.vis else set()
        items = [(light(info), osp.join(a.out, 'vis', f'{split}_{k:05d}_{info["token"][:8]}.png') if k in vis_idx else None)
                 for k, info in enumerate(infos)]
        t0 = time.time()
        fracs, boxes, wps, done = [], np.zeros(4, np.int64), np.zeros(3, np.int64), 0
        with Pool(a.workers) as pool:
            for r in pool.imap_unordered(work, items, chunksize=16):
                done += 1
                if r is not None:
                    fracs.append(r[0]); boxes += r[1]; wps += r[2]
                if done % 2000 == 0:
                    print(f'[{split}] {done}/{len(items)} {time.time() - t0:.0f}s', flush=True)
        fr = np.array(fracs) if fracs else np.zeros((1, 3))
        n_box, n_hit, cells, free = boxes
        print(f'[{split}] {len(items)} infos, {len(fracs)} written in {time.time() - t0:.0f}s', flush=True)
        for k, name in ((FREE, 'free'), (OCC, 'occupied'), (UNKNOWN, 'unknown')):
            print(f'[{split}]   {name:8s} fraction mean {fr[:, k].mean():.4f} std {fr[:, k].std():.4f} '
                  f'min {fr[:, k].min():.4f} max {fr[:, k].max():.4f}', flush=True)
        print(f'[{split}]   occupied among known cells {fr[:, OCC].sum() / max(fr[:, :2].sum(), 1e-9):.4f}', flush=True)
        print(f'[{split}]   boxes with an occupied cell {n_hit}/{n_box} = {n_hit / max(n_box, 1):.3f}; '
              f'cells inside boxes labelled free {free}/{cells} = {free / max(cells, 1):.3f}', flush=True)
        print(f'[{split}]   GT waypoints (outside ego footprint) on free/occupied/unknown {wps.tolist()} = '
              f'{np.round(wps / max(wps.sum(), 1), 4).tolist()}', flush=True)


if __name__ == '__main__':
    main()
