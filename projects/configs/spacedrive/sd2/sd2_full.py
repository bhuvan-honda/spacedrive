# SD2 + all three: BEV occupancy head (layer 14) + UniDepth FiLM (7/14/21) + depth-guided epipolar adapter.
_base_ = ['./sd2_base.py']

model = dict(
    bev_occ=dict(layer=14, loss_weight=0.5),
    depth_film=dict(layers=(7, 14, 21), mlp_dim=256),
    epipolar=dict(inner=256, heads=8),
)
