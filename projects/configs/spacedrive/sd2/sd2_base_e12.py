# SD2 base (= depth_skip_add_v2_fulllearning setup, no occ/FiLM/epipolar modules) trained for 12 epochs
# instead of 6 (cosine LR follows runner.max_iters). Used with VQA stage-1 init via load_from.
_base_ = ['./sd2_base.py']

num_epochs = 12
num_iters_per_epoch = 28130 // (4 * 8)

runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
