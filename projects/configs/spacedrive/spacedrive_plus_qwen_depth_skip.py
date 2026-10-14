# SpaceDrive+ with the depth skip-connection experiment.
#
# Inherits everything from the SpaceDrive+ config (ego_status='feature+PE') and
# only turns on the depth skip connection: the predicted depth map is encoded by
# the Qwen vision encoder, passed through a small trainable adapter, gets the SAME
# 3D positional encoding as the RGB vision tokens, and is appended as extra vision
# tokens.
_base_ = ['./spacedrive_plus_qwen.py']

# enable the depth skip connection
depth_skip = True
# 'concat' -> append depth tokens as extra vision tokens
# 'add'    -> add depth tokens element-wise onto the RGB vision tokens (no extra tokens)
depth_skip_mode = 'add'
# condition UniDepth on the GT camera intrinsics, on an aspect-restored (square-pixel)
# image (otherwise it self-estimates K while the 3D PE unprojection uses the GT K).
# nuScenes val vs LiDAR: AbsRel 0.175 -> 0.135, median scale 0.950 -> 1.006
depth_real_intrinsics = True

# separate work dir / result paths for this experiment
base_path = '/workspace/results/spacedrive_plus_qwen_depth_skip_add_real_intrinsics/'
work_dir = base_path
results_path = base_path + '_results_planning_only/'
wb_run_name = base_path + 'depth_skip_run'

model = dict(
    depth_skip=depth_skip,
    depth_skip_mode=depth_skip_mode,
    depth_real_intrinsics=depth_real_intrinsics,
    save_path=results_path,
)
