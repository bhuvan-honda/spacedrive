from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import collate, scatter
from mmcv.runner import HOOKS, Hook, get_dist_info
from mmdet3d.datasets import build_dataset
from torch.utils.data import DataLoader, Subset


@HOOKS.register_module()
class ValLossHook(Hook):
    """Every `interval` iterations, run the training forward (teacher forcing, eval mode, no grad) on a
    fixed val subset and log the mean of every loss term as `val_<name>`.

    The subset is `num_scenes` whole scenes spread evenly over the val set, restricted to frames with a
    GT planning trajectory (the same frames eval_planning.py scores). Scenes are split across ranks and
    each rank runs its frames in temporal order with batch size 1, so the ego-status memory sees the same
    history as at test time. The training memory and per-module train/eval flags are restored afterwards.
    interval <= 0 disables the hook.
    """

    MEMORY_ATTRS = ('memory_canbus', 'memory_egopose', 'sample_time', 'memory_count')
    # non-loss scalars that are also averaged over the subset (teacher-forced ADE/FDE, occupancy IoU, ...)
    METRIC_PREFIXES = ('tf_', 'occ_', 'epi_')

    def __init__(self, dataset, interval=0, num_scenes=16, workers=2):
        self.dataset_cfg = dataset
        self.interval = interval
        self.num_scenes = num_scenes
        self.workers = workers

    def before_run(self, runner):
        if self.interval <= 0:
            return
        self.dataset = build_dataset(self.dataset_cfg)
        scene_of = self.dataset.flag  # one group per scene (seq_mode=True, seq_split_num=1)
        scenes = np.unique(scene_of)
        chosen = scenes[::max(1, len(scenes) // self.num_scenes)][:self.num_scenes]
        rank, world_size = get_dist_info()
        mine = set(chosen[rank::world_size].tolist())
        self.indices = [
            i for i, info in enumerate(self.dataset.data_infos)
            if scene_of[i] in mine and 'gt_planning' in info
            and info['gt_planning_mask'].any(axis=-1).sum() == 6
        ]
        runner.logger.info(f'ValLossHook: {len(chosen)} val scenes, {len(self.indices)} frames on this rank, '
                           f'every {self.interval} iters')

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
            generator=torch.Generator(),  # keep the global torch RNG untouched
        )

        saved_memory = {k: getattr(model, k) for k in self.MEMORY_ATTRS if hasattr(model, k)}
        saved_modes = {m: m.training for m in model.modules()}
        sums, count = {}, 0
        try:
            if hasattr(model, 'reset_memory'):
                model.reset_memory()
            model.eval()
            with torch.no_grad():
                for data in loader:
                    data = scatter(data, [torch.cuda.current_device()])[0]
                    losses = model(return_loss=True, **data)
                    for k, v in losses.items():
                        if 'loss' in k or k.startswith(self.METRIC_PREFIXES):
                            sums[k] = sums.get(k, 0.0) + float(v)
                    count += 1
        finally:
            for m, training in saved_modes.items():
                m.training = training
            for k, v in saved_memory.items():
                setattr(model, k, v)

        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, (sums, count))
        else:
            gathered = [(sums, count)]
        total = sum(c for _, c in gathered)
        if total == 0:
            return
        for k in sorted({k for s, _ in gathered for k in s}):
            runner.log_buffer.output[f'val_{k}'] = sum(s.get(k, 0.0) for s, _ in gathered) / total
        runner.log_buffer.ready = True
