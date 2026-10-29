# sd2_occ_l28_depthlm_e12 with a lower occupancy weight (0.5 -> 0.1: the occ gradient into the LLM is
# ~orthogonal to planning and the occ runs overfit faster) and a lower positive weight (4.0 -> 1.5: the head
# over-predicted occupied cells, 28% vs 14% GT).
_base_ = ['./sd2_occ_l28_depthlm_e12.py']

model = dict(bev_occ=dict(loss_weight=0.1, pos_weight=1.5))
