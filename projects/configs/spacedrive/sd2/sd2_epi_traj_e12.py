# Epipolar adapter (sd2_epi_depthlm_e12, DepthLM-v2, 12 epochs) + trajectory/path answer format (see sd2_traj_e12.py).
# Variant via cfg-options: traj_format.path_spacing=1.0|2.0, traj_format.num_wp=12.
_base_ = ['./sd2_epi_depthlm_e12.py']

traj_format = dict(num_wp=6, path_spacing=0.0, path_points=20, extra_file='data/nuscenes/traj_extra.pkl')

model = dict(max_new_tokens=192)  # 12 waypoints + 20 path points need ~110 generated tokens
