#!/usr/bin/env python
"""Collect planning-eval metrics + train/val losses into a markdown table (results.md).

Usage (inside the container, from the repo root):
  python tools/results_md.py add --name NAME --eval-log logs/X_eval_log.txt \
      [--inf-log logs/X_inf.txt] [--work-dir results/RUN] [--ckpt-iter N] [--notes "..."] [--md results.md]

  - Metrics are the last values of l2_Ns / plan_obj_box_col_Ns / plan_boundary_Ns in the eval log.
  - Checkpoint: --ckpt-iter, else the last "CHECKPOINT:" line of the inference log (guessed from the
    eval-log name if --inf-log is not given). latest.pth is resolved via "Saving checkpoint" timestamps
    in the training text log vs the inference start time.
  - Work dir: --work-dir, else the directory of the checkpoint.
  - Losses come from the work dir *.log.json (resumed runs are stitched, newest log wins):
      train @ckpt = mean of logged train `loss` over the last 500 iters up to the ckpt;
      train best  = min of a 10-entry rolling mean up to the ckpt;
      val         = ValLossHook val_loss_pos + val_vlm_loss (16 val scenes) at / best up to the ckpt.
  - Re-adding an existing name replaces its row.
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEGIN, END = '<!-- RESULTS:BEGIN -->', '<!-- RESULTS:END -->'
COLUMNS = ['Experiment', 'Ckpt iter', 'L2 1s/2s/3s (m)', 'L2 avg', 'Col 1s/2s/3s (%)', 'Col avg',
           'Bnd 1s/2s/3s (%)', 'Bnd avg', 'Train loss @ckpt', 'Train loss best (iter)',
           'Val loss @ckpt (iter)', 'Val loss best (iter)', 'Notes']
HEADER = f"""# SpaceDrive results

nuScenes open-loop planning (val, 5119 samples with full 3 s GT). Lower is better everywhere.
- **L2**: ADE over the first 1/2/3 s (m). **Col**: object box collision rate (%). **Bnd**: out-of-drivable-area rate (%).
  `avg` = mean of the 1s/2s/3s values.
- **Train loss** (total `loss` = vlm_loss + loss_pos, rank 0): @ckpt = mean over the last 500 iters before the
  evaluated checkpoint; best = min 10-log-entry rolling mean up to that checkpoint.
- **Val loss** (ValLossHook, teacher-forced, 16 val scenes, val_loss_pos + val_vlm_loss): only for runs trained
  with the hook. Monitor for overfitting only; it does not rank checkpoints by L2 reliably.
- `-` = not available (released weights / no-learning baselines / run without ValLossHook).

Add or update a row (inside the container, from the repo root):
```
python tools/results_md.py add --name <name> --eval-log logs/<x>_eval_log.txt [--inf-log logs/<x>_inf.txt] \\
    [--work-dir results/<run>] [--ckpt-iter N] [--notes "..."]
```

{BEGIN}
| {' | '.join(COLUMNS)} |
|{'|'.join(['---'] * len(COLUMNS))}|
{END}
"""


def repo_path(p):
    if p is None:
        return None
    if p.startswith('/workspace/'):
        p = p[len('/workspace/'):]
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def read(path):
    with open(path, 'r', errors='replace') as f:
        return f.read()


def parse_eval(path):
    text = read(path)
    out = {}
    for key in ['l2', 'plan_obj_box_col', 'plan_boundary']:
        vals = []
        for t in (1, 2, 3):
            m = re.findall(rf'^{key}_{t}s: ([-\d.eE]+)', text, re.M)
            if not m:
                sys.exit(f'{key}_{t}s not found in {path}')
            vals.append(float(m[-1]))
        out[key] = vals
    pred = re.findall(r'The current pred_path (\S+)', text)
    out['pred_path'] = pred[-1] if pred else None
    n = re.findall(r'number of samples: (\d+)', text)
    out['samples'] = int(n[-1]) if n else None
    return out


def guess_inf_log(eval_log):
    base = eval_log.replace('_eval_log.txt', '')
    for cand in (base + '_inf_log.txt', base + '_inf.txt', base + '.txt'):
        if os.path.exists(cand):
            return cand
    return None


def ckpt_from_inf(inf_log):
    text = read(inf_log)
    ck = re.findall(r'CHECKPOINT: (\S+)', text) or re.findall(r'load checkpoint from local path: (\S+)', text)
    if not ck:
        return None, None, None
    elapsed = re.findall(r'elapsed: (\d+)s', text)
    end = os.path.getmtime(inf_log)
    start = end - int(elapsed[-1]) if elapsed else end
    return ck[-1], start, end


def work_dir_of(ckpt):
    d = repo_path(os.path.dirname(ckpt))
    if not os.path.isdir(d):
        alt = os.path.join(ROOT, 'results', os.path.basename(d))
        d = alt if os.path.isdir(alt) else d
    return d


def resolve_latest(work_dir, start_time):
    """Last checkpoint saved before start_time (epoch secs), from the training text logs."""
    best = None
    for log in sorted(glob.glob(os.path.join(work_dir, '*.log'))):
        lines = read(log).splitlines()
        stamps = [l for l in lines if re.match(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d', l)]
        if not stamps:
            continue
        # log timestamps are container-local; align them with the file mtime of the last line
        last = datetime.datetime.strptime(stamps[-1][:19], '%Y-%m-%d %H:%M:%S')
        offset = os.path.getmtime(log) - last.replace(tzinfo=datetime.timezone.utc).timestamp()
        for l in stamps:
            m = re.search(r'Saving checkpoint at (\d+) iterations', l)
            if m:
                t = datetime.datetime.strptime(l[:19], '%Y-%m-%d %H:%M:%S')
                t = t.replace(tzinfo=datetime.timezone.utc).timestamp() + offset
                if t <= start_time and (best is None or t > best[1]):
                    best = (int(m.group(1)), t)
    return best[0] if best else None


def load_json_logs(work_dir, ckpt_iter):
    files = []
    for f in sorted(glob.glob(os.path.join(work_dir, '*.log.json'))):
        train, val = [], []
        for line in read(f).splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get('mode') == 'train' and 'loss' in e:
                train.append((e['iter'], e['loss']))
            elif e.get('mode') == 'val' and 'val_loss_pos' in e:
                val.append((e['iter'], e['val_loss_pos'] + e.get('val_vlm_loss', 0.0)))
        if train:
            files.append((f, train, val))
    if not files:
        return [], []
    # anchor = newest log covering the checkpoint; older logs fill earlier iters (resumed runs)
    anchor = len(files) - 1
    if ckpt_iter is not None:
        for i in range(len(files) - 1, -1, -1):
            its = [it for it, _ in files[i][1]]
            if min(its) <= ckpt_iter and max(its) >= ckpt_iter - 200:
                anchor = i
                break
    train, val, lo = [], [], float('inf')
    for _, t, v in reversed(files[:anchor + 1]):
        train = [x for x in t if x[0] < lo] + train
        val = [x for x in v if x[0] < lo] + val
        lo = min(lo, min(it for it, _ in t))
    return train, val


def loss_stats(train, val, ckpt_iter):
    if ckpt_iter is not None:
        train = [x for x in train if x[0] <= ckpt_iter]
        val = [x for x in val if x[0] <= ckpt_iter]
    res = {}
    if train:
        last_it = train[-1][0]
        tail = [l for it, l in train if it > last_it - 500]
        res['train_ckpt'] = sum(tail) / len(tail)
        w = min(10, len(train))
        roll = [(train[i + w - 1][0], sum(l for _, l in train[i:i + w]) / w) for i in range(len(train) - w + 1)]
        res['train_best'] = min(roll, key=lambda x: x[1])
    if val:
        res['val_ckpt'] = val[-1]
        res['val_best'] = min(val, key=lambda x: x[1])
    return res


def fmt3(v, nd):
    return ' / '.join(f'{x:.{nd}f}' for x in v)


def build_row(args):
    ev = parse_eval(repo_path(args.eval_log))
    ckpt_iter, work_dir, notes = args.ckpt_iter, repo_path(args.work_dir), [args.notes] if args.notes else []
    inf_log = repo_path(args.inf_log) if args.inf_log else guess_inf_log(repo_path(args.eval_log))
    if inf_log and (ckpt_iter is None or work_dir is None):
        ckpt, start, _ = ckpt_from_inf(inf_log)
        if ckpt:
            if work_dir is None:
                work_dir = work_dir_of(ckpt)
            if ckpt_iter is None:
                m = re.search(r'iter_(\d+)\.pth', ckpt)
                if m:
                    ckpt_iter = int(m.group(1))
                elif ckpt.endswith('latest.pth') and work_dir and os.path.isdir(work_dir):
                    pred = repo_path(ev['pred_path']) if ev['pred_path'] else None
                    if pred and os.path.isdir(pred):
                        mt = [os.path.getmtime(os.path.join(pred, f)) for f in os.listdir(pred)]
                        if mt:
                            start = min(start, min(mt))
                    ckpt_iter = resolve_latest(work_dir, start)
                    print(f'[{args.name}] latest.pth resolved to iter {ckpt_iter} '
                          f'(inference started ~{datetime.datetime.fromtimestamp(start, datetime.timezone.utc):%Y-%m-%d %H:%M} UTC); '
                          f'pass --ckpt-iter to override')
    stats = {}
    if work_dir and os.path.isdir(work_dir):
        stats = loss_stats(*load_json_logs(work_dir, ckpt_iter), ckpt_iter)
    if ev['samples'] not in (None, 5119):
        notes.append(f"{ev['samples']} samples")

    avg = {k: sum(v) / 3 for k, v in ev.items() if isinstance(v, list)}
    dash = '-'
    tb = stats.get('train_best')
    vc, vb = stats.get('val_ckpt'), stats.get('val_best')
    cells = [
        args.name,
        str(ckpt_iter) if ckpt_iter is not None else dash,
        fmt3(ev['l2'], 3), f"**{avg['l2']:.4f}**",
        fmt3(ev['plan_obj_box_col'], 3), f"**{avg['plan_obj_box_col']:.3f}**",
        fmt3(ev['plan_boundary'], 3), f"**{avg['plan_boundary']:.3f}**",
        f"{stats['train_ckpt']:.4f}" if 'train_ckpt' in stats else dash,
        f'{tb[1]:.4f} ({tb[0]})' if tb else dash,
        f'{vc[1]:.4f} ({vc[0]})' if vc else dash,
        f'{vb[1]:.4f} ({vb[0]})' if vb else dash,
        '; '.join(notes) or dash,
    ]
    return '| ' + ' | '.join(c.replace('|', '/') for c in cells) + ' |'


def upsert(md_path, name, row):
    text = read(md_path) if os.path.exists(md_path) else HEADER
    if BEGIN not in text or END not in text:
        text = text.rstrip('\n') + '\n\n' + HEADER[HEADER.index(BEGIN):]
    pre, rest = text.split(BEGIN, 1)
    table, post = rest.split(END, 1)
    lines = [l for l in table.strip('\n').splitlines() if l.strip()]
    for i, l in enumerate(lines[2:], start=2):
        if l.split('|')[1].strip() == name:
            lines[i] = row
            break
    else:
        lines.append(row)
    with open(md_path, 'w') as f:
        f.write(pre + BEGIN + '\n' + '\n'.join(lines) + '\n' + END + post)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('add', help='add or replace one row')
    a.add_argument('--name', required=True)
    a.add_argument('--eval-log', required=True)
    a.add_argument('--inf-log')
    a.add_argument('--work-dir')
    a.add_argument('--ckpt-iter', type=int)
    a.add_argument('--notes', default='')
    a.add_argument('--md', default='results.md')
    args = p.parse_args()
    row = build_row(args)
    upsert(repo_path(args.md), args.name, row)
    print(row)


if __name__ == '__main__':
    main()
