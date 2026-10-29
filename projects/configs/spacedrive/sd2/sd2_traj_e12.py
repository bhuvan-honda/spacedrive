# Trajectory-format experiments: sd2_base_e12 (no SD2 modules, 12 epochs) with a different planning answer.
# The variant is chosen with cfg-options on the top-level traj_format (scripts/train/train.py copies it into the
# model, the train data and the hook datasets):
#   traj_format.num_wp=12          -> 12 waypoints every 0.25 s ("dense planning trajectory"), scored at 0.5 s
#   traj_format.path_spacing=1.0   -> + 20 path points every 1 m ("driving path"); 2.0 -> every 2 m
# Labels: data/nuscenes/traj_extra.pkl (tools/gen_traj_extra.py).
_base_ = ['./sd2_base_e12.py']

traj_format = dict(num_wp=6, path_spacing=0.0, path_points=20, extra_file='data/nuscenes/traj_extra.pkl')

model = dict(max_new_tokens=192)  # 12 waypoints + 20 path points need ~110 generated tokens
