# SD2 + BEV occupancy head (layer 14) + UniDepth FiLM (layers 7/14/21).
_base_ = ['./sd2_base.py']

model = dict(
    bev_occ=dict(layer=14, loss_weight=0.5),
    depth_film=dict(layers=(7, 14, 21), mlp_dim=256),
)
