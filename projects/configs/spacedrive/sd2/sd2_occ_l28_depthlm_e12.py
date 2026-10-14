# SD2 + BEV occupancy aux head at LLM layer 28 (sd2_occ, layer=28), initialised from the DepthLM-v2 merged
# weights (as depthlm_v4), trained for 12 epochs instead of 6 (cosine LR follows runner.max_iters).
_base_ = ['./sd2_occ.py']

num_epochs = 12
num_iters_per_epoch = 28130 // (4 * 8)

model = dict(
    lm_head='ckpts/Qwen2.5-VL-7B-Instruct-with-new-special-tokens-DepthLM-v2/',
    bev_occ=dict(layer=28),
)

runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
