# SpaceDrive+ with the ORTHOGONAL depth-fusion experiment.
#
# Builds on the depth skip connection ('add' mode): the predicted depth map is
# encoded by the Qwen vision encoder, passed through a small trainable adapter, and
# added element-wise onto the RGB vision tokens. Before adding, the depth token is
# made orthogonal to the appearance token (per-token Gram-Schmidt against a
# stop-grad appearance reference), so it injects only the geometry component that
# appearance cannot already explain and does not corrupt the visual tokens. A single
# learnable scalar controls the overall depth magnitude.
_base_ = ['./spacedrive_plus_qwen_depth_skip.py']

# depth skip in 'add' mode + orthogonal fusion
depth_skip = True
depth_skip_mode = 'add'
# 'none' -> plain add ; 'hard' -> project depth orthogonal to appearance ; 'soft' -> cos^2 penalty
depth_orth_mode = 'hard'
depth_orth_lambda = 1.0
depth_scale_init = 1.0

# separate work dir / result paths for this experiment
base_path = '/workspace/results/spacedrive_plus_qwen_depth_orth_hard/'
work_dir = base_path
results_path = base_path + '_results_planning_only/'
wb_run_name = base_path + 'depth_orth_run'

# ---- multi-batch (true batch_size > 1 per GPU) ----
# Temporal streaming for B>1 is handled by InfiniteGroupEachSampleInBatchSampler
# (one continuous sequence per batch slot), which the base config already uses.
# All batch-dependent values are recomputed here so changing batch_size cascades.
num_gpus = 4
batch_size = 2
num_epochs = 6
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
lr = 1e-4 / 2 * batch_size / 8 * num_gpus

data = dict(samples_per_gpu=batch_size)
optimizer = dict(lr=lr)
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
checkpoint_config = dict(interval=num_iters_per_epoch // 2, max_keep_ckpts=12)
evaluation = dict(interval=num_iters_per_epoch * num_epochs)

model = dict(
    depth_skip=depth_skip,
    depth_skip_mode=depth_skip_mode,
    depth_orth_mode=depth_orth_mode,
    depth_orth_lambda=depth_orth_lambda,
    depth_scale_init=depth_scale_init,
    save_path=results_path,
)
