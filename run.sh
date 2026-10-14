# scripts/train/dist_train.sh projects/configs/spacedrive/spacedrive_plus_qwen_depth_skip.py 6 --resume-from results/spacedrive_plus_qwen_depth_skip_add_real_intrinsics/latest.pth
scripts/train/dist_train.sh projects/configs/spacedrive/spacedrive_plus_qwen_depthlm.py 4 --cfg-options model.attn_implementation=flash_attention_2 model.depth_skip_gpu_preprocess=True

scripts/test/dist_test.sh projects/configs/spacedrive/spacedrive_plus_qwen_depthlm.py 4   --cfg-options model.attn_implementation=flash_attention_2 model.depth_skip_gpu_preprocess=True --format-only >> logs/spacedrive_plus_qwen_depthlm_v4_inf.txt

python scripts/evaluation/eval_planning.py --pred_path results/spacedrive_plus_qwen_depth_skip_add_real_intrinsics_depthlm_v4/_results_planning_only/ --base_path data/nuscenes/  >> logs/spacedrive_plus_qwen_depthlm_v4_eval_log.txt

