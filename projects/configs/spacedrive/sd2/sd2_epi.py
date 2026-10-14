# SD2 + depth-guided epipolar cross-view adapter on the camera vision tokens before the LLM
# (zero-initialised output projection: identical to sd2_base at iteration 0).
_base_ = ['./sd2_base.py']

model = dict(epipolar=dict(inner=256, heads=8))
