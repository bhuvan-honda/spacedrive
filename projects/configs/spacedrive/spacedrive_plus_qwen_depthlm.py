# SpaceDrive+ depth skip-connection experiment initialised from DepthLM: the DepthLM LoRA
# (LLM attn+MLP and vision MLP, r16, alpha32) merged into Qwen2.5-VL-7B-Instruct-with-new-special-tokens
# with tools/merge_lora.py. Only the model weights change; tokenizer/processor files are identical to the base.
# The merged folder name must keep 'Qwen2.5' and 'with-new-special-tokens' (checked in load_model).
#
# Inherits depth_skip=True, depth_skip_mode='add', depth_real_intrinsics=True from the depth_skip config.
_base_ = ['./spacedrive_plus_qwen_depth_skip.py']

llm_path = 'ckpts/Qwen2.5-VL-7B-Instruct-with-new-special-tokens-DepthLM-v2/'
base_path = '/workspace/results/spacedrive_plus_qwen_depth_skip_add_real_intrinsics_depthlm_v4/'

work_dir = base_path
results_path = base_path + '_results_planning_only/'
wb_run_name = base_path + 'depthlm_run'

model = dict(
    lm_head=llm_path,
    save_path=results_path,
)
