# SD2 + depth-guided epipolar adapter + BEV occupancy aux head at LLM layer 28, initialised from the
# DepthLM-v2 merged weights, 12 epochs (= sd2_epi_depthlm_e12 + the sd2_occ_l28_depthlm_e12 head; no FiLM).
_base_ = ['./sd2_epi_depthlm_e12.py']

model = dict(bev_occ=dict(layer=28, loss_weight=0.5))
