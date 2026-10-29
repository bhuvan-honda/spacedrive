# SD2 + depth-guided epipolar adapter (sd2_epi), standard init (no DepthLM), trained for 12 epochs instead of 6
# (cosine LR follows runner.max_iters). Control for sd2_epi_depthlm_e12: isolates the DepthLM-init effect.
_base_ = ['./sd2_epi.py']

num_epochs = 12
num_iters_per_epoch = 28130 // (4 * 8)

runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
