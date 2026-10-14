# sd2_occ (layer 14) with the occupancy head input detached: the head still trains on the LLM hidden states, but
# its loss cannot change the LLM / adapters. If this recovers sd2_base, the occ_l14 regression came from the
# occupancy gradient competing with planning in the shared weights.
_base_ = ['./sd2_occ.py']

model = dict(bev_occ=dict(detach_input=True))
