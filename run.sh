scripts/train/dist_train.sh projects/configs/spacedrive/spacedrive_plus_qwen_depth_skip.py 4

scripts/test/dist_test.sh projects/configs/spacedrive/spacedrive_plus_qwen_depth_skip.py 4  --format-only >> logs/spacedrive_plus_qwen_depth_skip_concat_v2_inf_log.txt

python scripts/evaluation/eval_planning.py --pred_path results/spacedrive_plus_qwen_depth_skip_concat_v2/_results_planning_only/ --base_path data/nuscenes/  >> logs/spacedrive_plus_qwen_depth_skip_concat_v2_eval_log.txt

