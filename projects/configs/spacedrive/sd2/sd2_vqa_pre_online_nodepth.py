# VQA stage 1 (online grounding QA) without any depth input, for the no-depth ablation (sd2_nodepth_e12.py).
_base_ = ['./sd2_vqa_pre_online.py']

model = dict(
    vis_3d_pos=False,
    depth_skip=False,
    depth_skip_gpu_preprocess=False,
    # online-VQA batches peak at ~65 GB fully checkpointed (~23 GB per extra layer): none fit the 80 GB cap of GPUs 0-3
    grad_ckpt_skip_layers=0 if {'0', '1', '2', '3'} & set(__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) else 2,
)
