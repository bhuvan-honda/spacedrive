# SD2 + auxiliary BEV occupancy head on the vision-token hidden states entering LLM layer 14
# (LiDAR 3-class GT free / occupied / unknown from tools/gen_bev_occ.py in data/nuscenes/bev_occ3, 200x200 @ 0.5 m;
# occupied-vs-free loss, unknown cells ignored; training only).
_base_ = ['./sd2_base.py']

model = dict(bev_occ=dict(layer=14, loss_weight=0.5))
