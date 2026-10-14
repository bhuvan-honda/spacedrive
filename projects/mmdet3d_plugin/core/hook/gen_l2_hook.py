import os
import os.path as osp
import re
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import collate, scatter
from mmcv.runner import HOOKS, Hook, get_dist_info
from mmdet3d.datasets import build_dataset
from torch.utils.data import DataLoader, Subset

TRAJ_RE = re.compile(r'\[<POS_INDICATOR>\(([\d\.-]+, [\d\.-]+)\)\s*([\s\S]*<POS_INDICATOR>\([\d\.-]+, [\d\.-]+\)\s*)*')
COORD_RE = re.compile(r'\(\+?[\d\.-]+, \+?[\d\.-]+\)')
NUM_RE = re.compile(r'-?\d+\.\d+')


def parse_traj(answer):
    """Same parsing as scripts/evaluation/eval_planning.py (spacedrive branch). Returns (K, 2) or None."""
    if not isinstance(answer, str):
        return None
    m = TRAJ_RE.search(answer)
    if not m:
        return None
    coords = [tuple(map(float, NUM_RE.findall(c))) for c in COORD_RE.findall(m.group(0))]
    coords = [c for c in coords if len(c) == 2]
    return np.array(coords) if coords else None


@HOOKS.register_module()
class GenL2Hook(Hook):
    """Every `interval` iterations, run the real (free-running) generation on a fixed val subset and log the
    open-loop L2 exactly like scripts/evaluation/eval_planning.py: L2 at {1,2,3}s = mean displacement over
    the first {2,4,6} waypoints, averaged over frames with a full GT future.

    The subset is `num_scenes` whole scenes spread evenly over the val set (same rule as ValLossHook).
    Scenes are split over ranks; each rank runs its frames in temporal order with batch size 1, so the
    ego-status memory sees the test-time history. Generated jsons go to <work_dir>/_gen_l2_tmp instead of
    model.save_path. Logged: val_gen_l2_{1,2,3}s, val_gen_l2_avg, val_gen_fail (frames without a parsable
    6-waypoint trajectory, excluded from the L2). interval <= 0 disables the hook.
    """

    MEMORY_ATTRS = ('memory_canbus', 'memory_egopose', 'sample_time', 'memory_count')

    def __init__(self, dataset, interval=0, num_scenes=16, workers=2):
        self.dataset_cfg = dataset
        self.interval = interval
        self.num_scenes = num_scenes
        self.workers = workers

    def before_run(self, runner):
        if self.interval <= 0:
            return
        cfg = dict(self.dataset_cfg)
        cfg['pipeline'] = [t for t in cfg['pipeline'] if not t['type'].startswith('LoadPoints')]  # unused by the model
        self.dataset = build_dataset(cfg)
        infos = self.dataset.data_infos
        scenes = list(dict.fromkeys(info['scene_token'] for info in infos))  # first-appearance order
        chosen = scenes[::max(1, len(scenes) // self.num_scenes)][:self.num_scenes]
        rank, world_size = get_dist_info()
        mine = set(chosen[rank::world_size])
        self.indices = [i for i, info in enumerate(infos) if info['scene_token'] in mine]
        n_scored = sum(1 for i in self.indices if 'gt_planning' in infos[i] and infos[i]['gt_planning_mask'][0].all())
        self.tmp_dir = osp.join(runner.work_dir, '_gen_l2_tmp') + '/'
        os.makedirs(self.tmp_dir, exist_ok=True)
        runner.logger.info(f'GenL2Hook: {len(chosen)} val scenes, {len(self.indices)} frames ({n_scored} scored) '
                           f'on this rank, every {self.interval} iters')

    def after_train_iter(self, runner):
        if self.interval <= 0 or not self.every_n_iters(runner, self.interval):
            return
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        loader = DataLoader(
            Subset(self.dataset, self.indices),
            batch_size=1,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=partial(collate, samples_per_gpu=1),
            generator=torch.Generator(),
        )
        infos = self.dataset.data_infos
        saved_memory = {k: getattr(model, k) for k in self.MEMORY_ATTRS if hasattr(model, k)}
        saved_modes = {m: m.training for m in model.modules()}
        saved_path = model.save_path
        sums, n_ok, n_fail = np.zeros(3), 0, 0
        try:
            if hasattr(model, 'reset_memory'):
                model.reset_memory()
            model.save_path = self.tmp_dir
            model.eval()
            with torch.no_grad():
                for idx, data in zip(self.indices, loader):
                    data = scatter(data, [torch.cuda.current_device()])[0]
                    out = model(return_loss=False, rescale=True, **data)
                    info = infos[idx]
                    if 'gt_planning' not in info or not info['gt_planning_mask'][0].all():
                        continue
                    traj = parse_traj(out[0]['A']) if out else None
                    if traj is None or len(traj) < 6:
                        n_fail += 1
                        continue
                    err = np.linalg.norm(traj[:6, :2] - info['gt_planning'][0, :6, :2], axis=-1)
                    sums += [err[:2].mean(), err[:4].mean(), err[:6].mean()]
                    n_ok += 1
        finally:
            model.save_path = saved_path
            for m, training in saved_modes.items():
                m.training = training
            for k, v in saved_memory.items():
                setattr(model, k, v)

        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, (sums, n_ok, n_fail))
        else:
            gathered = [(sums, n_ok, n_fail)]
        total = sum(g[1] for g in gathered)
        fails = sum(g[2] for g in gathered)
        out = runner.log_buffer.output
        if total > 0:
            l2 = sum(g[0] for g in gathered) / total
            for t in range(3):
                out[f'val_gen_l2_{t + 1}s'] = float(l2[t])
            out['val_gen_l2_avg'] = float(l2.mean())
        out['val_gen_fail'] = fails
        runner.log_buffer.ready = True
        runner.logger.info(f'GenL2Hook: {total} frames scored, {fails} failed')
