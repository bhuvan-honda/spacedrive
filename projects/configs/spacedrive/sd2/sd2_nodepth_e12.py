# No-depth ablation: sd2_base_e12 without any depth input (no UniDepth / vision-token 3D PE, no depth skip).
# Answer coordinates keep their 3D PE (io_3d_pos does not use depth). Used with the DepthLM-v2 weights.
_base_ = ['./sd2_base_e12.py']

model = dict(
    vis_3d_pos=False,
    depth_skip=False,
    depth_skip_gpu_preprocess=False,
    # GPUs 0-3 are capped at 80 GB (nvidia-smi); without UniDepth one more layer fits there
    grad_ckpt_skip_layers=5 if {'0', '1', '2', '3'} & set(__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) else 14,
)
