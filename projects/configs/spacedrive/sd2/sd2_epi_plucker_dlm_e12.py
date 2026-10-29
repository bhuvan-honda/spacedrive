# sd2_epi_depthlm_e12 + Plücker ray embedding (d, c x d per vision token, ego frame) added to the
# depth-based 3D PE of the vision tokens (zero-initialised output: identical to sd2_epi_depthlm_e12 at iter 0).
_base_ = ['./sd2_epi_depthlm_e12.py']

model = dict(plucker_pe=dict(num_freqs=6, hidden=512, moment_scale=2.0))
