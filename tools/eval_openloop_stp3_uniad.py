#!/usr/bin/env python
"""ST-P3 (VAD-style) and UniAD-style open-loop planning metrics for every run in results.md.

Same definitions as the official scripts (SpaceDrive supplementary Tab. G; docs/train_test.md):
  ST-P3 = scripts/evaluation/eval_planning_vad.py   : L2 averaged over the waypoints up to t,
          collision = fraction of the frames up to t in collision (obj_box_coll.mean()).
  UniAD = scripts/evaluation/eval_planning_uniad.py : L2 of the waypoint at t,
          collision = any collision up to t (obj_box_coll.max()).
  Both: 200x200 BEV @ 0.5 m (planning_utils_vad / _uniad are functionally identical), samples with full 3 s GT.
Boundary (out-of-drivable) is not part of these tables; the VAD script's boundary overwrites instead of
accumulating, so it is not reproduced here.

Predictions are taken from the eval log (eval_planning.py prints every raw answer in val-info order),
i.e. exactly what produced the results.md row, even if the prediction dir was later deleted/overwritten.
If the log does not hold one answer line per val sample, the prediction dir is used instead.
Check: the ST-P3 L2 equals the default eval's L2 by definition, so it is compared with the l2_Ns values in
the eval log; a mismatch means the predictions are not those of the results.md row.

Usage (inside the container, from the repo root):
  python tools/eval_openloop_stp3_uniad.py                    # all results.md runs -> results.md section
  python tools/eval_openloop_stp3_uniad.py --run NAME=EVAL_LOG ...   # extra / override runs
Per-run metrics are cached in results/_openloop_cache/ (keyed by eval-log size+mtime); --force recomputes.
"""
import argparse
import glob
import importlib.util
import json
import os
import pickle
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'evaluation'))
_spec = importlib.util.spec_from_file_location(
    'traj_format', os.path.join(ROOT, 'projects/mmdet3d_plugin/core/traj_format.py'))
traj_format = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(traj_format)

BEGIN, END = '<!-- OPENLOOP:BEGIN -->', '<!-- OPENLOOP:END -->'
CACHE_DIR = os.path.join(ROOT, 'results', '_openloop_cache')
MAIN_BEGIN, MAIN_END = '<!-- RESULTS:BEGIN -->', '<!-- RESULTS:END -->'

# Reported numbers (SpaceDrive paper main table) for reference rows.
PAPER = {
    'SpaceDrive (paper)': ([1.06, 1.79, 2.55], [0.35, 0.61, 1.31], [1.41, 2.88, 4.51], [0.59, 1.72, 4.53]),
    'SpaceDrive+ (paper)': ([0.15, 0.29, 0.51], [0.05, 0.08, 0.16], [0.20, 0.53, 1.13], [0.10, 0.31, 0.80]),
}


def rel(p):
    p = p.strip()
    return p[len('/workspace/'):] if p.startswith('/workspace/') else p


def md_run_names(md):
    text = open(md).read()
    table = text.split(MAIN_BEGIN, 1)[1].split(MAIN_END, 1)[0]
    rows = [l for l in table.strip().splitlines() if l.startswith('|')][2:]
    return [r.split('|')[1].strip() for r in rows]


def eval_log_for(name):
    backfill = open(os.path.join(ROOT, 'tools', 'results_md_backfill.sh')).read()
    m = re.search(rf'--name {re.escape(name)}\s+--eval-log (\S+)', backfill)
    cands = [m.group(1).replace('$L/', 'logs/')] if m else []
    if name.startswith('sd2_'):
        cands.append(f'logs/queue/{name[4:]}_eval.txt')
    cands += [f'logs/spacedrive_plus_qwen_{name}_eval_log.txt', f'logs/spacedrive_{name}_eval_log.txt']
    return next((c for c in cands if os.path.exists(c)), None)


def parse_answer(traj, project=False):
    if traj_format.is_new_format(traj):
        wp, path, dense = traj_format.split_answer(traj)
        half = traj_format.to_half_seconds(wp, dense)
        if half is not None and project and path is not None and len(path):
            half = traj_format.project_on_path(wp, path, traj_format.half_second_index(12 if dense else 6))
        return half
    # Same regexes as eval_planning.py ('spacedrive' branch).
    full = re.search(r'\[<POS_INDICATOR>\(([\d\.-]+, [\d\.-]+)\)\s*([\s\S]*<POS_INDICATOR>\([\d\.-]+, [\d\.-]+\)\s*)*',
                     traj)
    if not full:
        return None
    coords = re.findall(r'\(\+?[\d\.-]+, \+?[\d\.-]+\)', full.group(0))
    return np.array([tuple(map(float, re.findall(r'-?\d+\.\d+', c))) for c in coords])


def load_predictions(eval_log, infos):
    # newline='' keeps tqdm's '\r' inside its line (universal newlines would add an empty "answer" line).
    lines = open(eval_log, errors='replace', newline='').read().split('\n')
    i0 = next(i for i, l in enumerate(lines) if l.startswith('The current pred_path'))
    pred_dir = rel(lines[i0].split('pred_path', 1)[1])
    # eval_planning.py --path_project prints this summary right after the answers
    project = any(l.startswith('path projection applied') for l in lines)
    answers = []
    for l in lines[i0 + 1:]:
        if (l.startswith('Processing sample') or 'it/s]' in l or '%|' in l or l.startswith('number of samples')
                or l.startswith('path projection applied')):
            break
        answers.append(l)
    tokens = [d['token'] for d in infos]
    if len(answers) == len(tokens):
        raw, src = dict(zip(tokens, answers)), 'eval log'
    elif os.path.isdir(pred_dir) and len(os.listdir(pred_dir)) > 0:
        raw = {}
        for t in tokens:
            f = os.path.join(pred_dir, t)
            if os.path.exists(f):
                raw[t] = json.load(open(f))[0]['A']
        src = 'pred dir'
        print(f'  {len(answers)} answer lines in log != {len(tokens)} samples; using {pred_dir}')
    else:
        return None, f'{len(answers)} answer lines in log, pred dir {pred_dir} missing'
    preds = {}
    for t, a in raw.items():
        c = parse_answer(a, project)
        if c is not None:
            preds[t] = c
    return preds, src + (', path-projected' if project else '')


def logged_l2(eval_log):
    text = open(eval_log, errors='replace').read()
    return [float(re.findall(rf'^l2_{s}s: ([\d.]+)', text, re.M)[-1]) for s in (1, 2, 3)]


class GT:
    """Per-sample ground truth shared by all runs (BEV occupancy at 0.5 m + GT trajectory)."""

    def __init__(self, base_path, infos, metric):
        self.items = []
        for d in infos:
            if not d['gt_planning_mask'][0].all():
                continue
            boxes = np.concatenate([d['gt_boxes'], d['gt_velocity']], -1)
            feats = np.concatenate([d['gt_fut_traj'][:, :6].reshape(-1, 12), d['gt_fut_traj_mask'][:, :6],
                                    d['gt_fut_yaw'][:, :6], d['gt_fut_idx']], -1)
            seg = metric.get_birds_eye_view_label(boxes, feats, add_rec=True)
            self.items.append((d['token'], torch.from_numpy(d['gt_planning'][..., :2]),
                               torch.from_numpy(seg[1:]).unsqueeze(0)))


def run_metrics(preds, gt, metric):
    acc = {k: np.zeros(3) for k in ('stp3_l2', 'stp3_col', 'uniad_l2', 'uniad_col', 'stp3_col_round', 'uniad_col_round')}
    n = short = 0
    for tok, gt_traj, seg in gt.items:
        if tok not in preds:
            continue
        p = preds[tok][:6, :]
        if p.ndim != 2 or p.shape != (6, 2):
            short += 1  # the official scripts raise on these inside a try/except
            continue
        pred = torch.from_numpy(p).unsqueeze(0)
        n += 1
        err = torch.linalg.norm(pred[0] - gt_traj[0], dim=-1).numpy()
        # Collision is evaluated per frame (yaw from consecutive deltas), so the 6-frame result sliced to
        # the first t frames equals evaluating the first t frames as the official scripts do.
        _, box_coll = metric.evaluate_coll(pred, gt_traj, seg)
        box_coll = box_coll.numpy()
        _, box_coll_r = metric.evaluate_coll(pred, gt_traj, seg, round_cells=True)
        box_coll_r = box_coll_r.numpy()
        for i in range(3):
            t = (i + 1) * 2
            acc['stp3_l2'][i] += err[:t].mean()
            acc['uniad_l2'][i] += err[t - 1]
            acc['stp3_col'][i] += box_coll[:t].mean()
            acc['uniad_col'][i] += box_coll[:t].max()
            acc['stp3_col_round'][i] += box_coll_r[:t].mean()
            acc['uniad_col_round'][i] += box_coll_r[:t].max()
    out = {k: (v / n * (100 if 'col' in k else 1)).tolist() for k, v in acc.items()}
    out.update(samples=n, short=short)
    return out


def fmt_row(name, s_l2, s_col, u_l2, u_col, note='', d=3):
    def block(v):
        return [f'{x:.{d}f}' for x in v] + [f'**{np.mean(v):.{d}f}**']
    return '| ' + ' | '.join([name] + block(s_l2) + block(s_col) + block(u_l2) + block(u_col) + [note]) + ' |'


def write_section(md, rows, n_samples, skipped, round_rows=()):
    head = ('| Experiment | L2 1s | L2 2s | L2 3s | **L2 avg** | Col 1s | Col 2s | Col 3s | **Col avg** '
            '| L2 1s | L2 2s | L2 3s | **L2 avg** | Col 1s | Col 2s | Col 3s | **Col avg** | Notes |\n'
            '|' + '|'.join(['---'] * 18) + '|')
    body = '\n'.join(rows)
    skip = '\n'.join(f'- `{n}`: {why}' for n, why in skipped.items())
    skip = f'\nNot in the table:\n{skip}\n' if skip else ''
    round_table = ''
    if round_rows:
        round_table = (
            '\n### Collision with nearest-cell rounding (not the official protocol)\n\n'
            'The official box-collision check places the ego box with `astype(int)` (truncation toward zero), so a\n'
            'near-zero coordinate of -0.004 m and +0.004 m lands in cells 0.5 m apart: stationary frames can flip into a\n'
            'collision with a neighbouring vehicle. Same metrics with `np.rint` instead (`evaluate_coll(round_cells=True)`):\n\n'
            '| Experiment | ST-P3 Col 1s | 2s | 3s | **avg** | UniAD Col 1s | 2s | 3s | **avg** |\n'
            '|' + '|'.join(['---'] * 9) + '|\n' + '\n'.join(round_rows) + '\n')
    section = f"""{BEGIN}
## Open-loop metrics: ST-P3 vs UniAD protocol (all runs)

Generated by `python tools/eval_openloop_stp3_uniad.py` (inside the container, repo root); re-run after new
evals. Columns 2-9 = **ST-P3** protocol (`eval_planning_vad.py`: L2 averaged over waypoints up to t, collision =
fraction of frames up to t), columns 10-17 = **UniAD** protocol (`eval_planning_uniad.py`: L2 at t, collision =
any collision up to t). Both on a 200x200 @ 0.5 m BEV, no ego yaw, {n_samples} val samples. Col in %.
ST-P3 L2 is identical to the main-table L2 (checked per run against the eval log). The main-table Col uses a
finer 0.1 m grid with tangent yaw and "any collision up to t", so it is not directly comparable to either.
Verified on sd2_epi_dlm_e12: identical to `eval_planning_vad.py` / `eval_planning_uniad.py` (all digits). Note:
`eval_planning_vad.py` with its default 4 threads under-counts ST-P3 Col at 2s/3s: an undefined `out_of_frame`
in its debug print raises for colliding samples until each thread has seen a boundary violation, and the
remaining horizons of those samples are dropped (sd2_epi_dlm_e12: 0.083/0.160 instead of 0.088/0.173 with 8
threads). The values here are the bug-free (single-thread) ones. Paper rows are the reported numbers (2 decimals).

|  | ST-P3 | | | | | | | | UniAD | | | | | | | | |
{head}
{body}
{round_table}{skip}{END}"""
    text = open(md).read()
    if BEGIN in text:
        pre, rest = text.split(BEGIN, 1)
        text = pre + section + rest.split(END, 1)[1]
    else:
        text = text.rstrip('\n') + '\n\n' + section + '\n'
    open(md, 'w').write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--md', default='results.md')
    ap.add_argument('--base-path', default='data/nuscenes/')
    ap.add_argument('--anno', default='nuscenes2d_ego_temporal_infos_val.pkl')
    ap.add_argument('--run', action='append', default=[], help='NAME=EVAL_LOG (extra or override)')
    ap.add_argument('--force', action='store_true', help='ignore the cache')
    args = ap.parse_args()
    os.chdir(ROOT)

    runs = {n: eval_log_for(n) for n in md_run_names(args.md)}
    for r in args.run:
        n, p = r.split('=', 1)
        runs[n] = p

    os.makedirs(CACHE_DIR, exist_ok=True)
    infos = pickle.load(open(os.path.join(args.base_path, args.anno), 'rb'))['infos']
    metric, gt, results, skipped = None, None, {}, {}
    for name, log in runs.items():
        if not log:
            print(f'[{name}] no eval log found, skipped')
            skipped[name] = 'no eval log'
            continue
        key = f'v2_{os.path.getsize(log)}_{int(os.path.getmtime(log))}'  # v2: + rounded collisions
        cache = os.path.join(CACHE_DIR, f'{name}.json')
        if not args.force and os.path.exists(cache):
            c = json.load(open(cache))
            if c.get('key') == key:
                results[name] = c
                print(f'[{name}] cached')
                continue
        preds, src = load_predictions(log, infos)
        if preds is None:
            print(f'[{name}] skipped: {src}')
            skipped[name] = 'predictions not available (not in eval log, pred dir deleted)'
            continue
        if gt is None:
            from planning_utils_vad import PlanningMetric
            print('building GT BEV (200x200 @ 0.5 m) ...')
            metric = PlanningMetric(args.base_path)
            gt = GT(args.base_path, infos, metric)
        m = run_metrics(preds, gt, metric)
        ref = logged_l2(log)
        m.update(key=key, eval_log=log, source=src,
                 l2_check=bool(np.allclose(m['stp3_l2'], ref, atol=1e-6)), logged_l2=ref)
        json.dump(m, open(cache, 'w'))
        results[name] = m
        print(f'[{name}] {src}, {m["samples"]} samples, short {m["short"]}, '
              f'STP3 L2 {np.round(m["stp3_l2"], 4)} vs log {ref} -> {"OK" if m["l2_check"] else "MISMATCH"}')

    rows = [fmt_row(n, *v, note='reported', d=2) for n, v in PAPER.items()]
    round_rows = []
    for name, m in results.items():
        if not m['l2_check']:
            skipped[name] = 'pred dir was overwritten by a later eval (L2 does not match its eval log)'
            continue
        notes = []
        if m['short']:
            notes.append(f'{m["short"]} preds with != 6 waypoints skipped')
        src = m['source'].split(', ')
        if src[0] != 'eval log':
            notes.append(f'preds from {src[0]}')
        if 'path-projected' in src:
            notes.append('path-projected')
        rows.append(fmt_row(name, m['stp3_l2'], m['stp3_col'], m['uniad_l2'], m['uniad_col'], '; '.join(notes)))
        if 'stp3_col_round' in m:
            round_rows.append('| ' + ' | '.join(
                [name] + [f'{x:.3f}' for x in m['stp3_col_round']] + [f'**{np.mean(m["stp3_col_round"]):.3f}**']
                + [f'{x:.3f}' for x in m['uniad_col_round']] + [f'**{np.mean(m["uniad_col_round"]):.3f}**']) + ' |')
    n_samples = sorted({m['samples'] for m in results.values()})
    write_section(args.md, rows, '/'.join(map(str, n_samples)), skipped, round_rows)
    print(f'wrote {len(rows)} rows to {args.md}')


if __name__ == '__main__':
    main()
