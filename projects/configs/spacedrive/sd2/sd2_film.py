# SD2 + FiLM of the vision tokens before LLM layers 7/14/21, conditioned on frozen UniDepth decoder latents,
# log depth and log confidence (zero-initialised: identical to sd2_base at iteration 0).
_base_ = ['./sd2_base.py']

model = dict(depth_film=dict(layers=(7, 14, 21), mlp_dim=256))
