# SD2 + depth-guided epipolar adapter (sd2_epi), initialised from the DepthLM-v2 merged weights (as depthlm_v4),
# trained for 12 epochs instead of 6 (cosine LR follows runner.max_iters).
_base_ = ['./sd2_epi.py']

num_epochs = 12
num_iters_per_epoch = 28130 // (4 * 8)

model = dict(lm_head='ckpts/Qwen2.5-VL-7B-Instruct-with-new-special-tokens-DepthLM-v2/')

runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
