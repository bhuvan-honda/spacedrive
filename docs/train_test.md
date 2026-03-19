# Train & Eval
## Train
```bash
# scripts/dist_train.sh <YOUR_CONFIG> <NUM_GPUS> 
scripts/train/dist_train.sh projects/configs/spacedrive/spacedrive_plus_qwen.py 8
```


## Evaluation
<!-- ### 1. OpenLoop Planning -->

Run the inference command which saves the predicted trajectories
```bash
# scripts/test/dist_test.sh <YOUR_CONFIG>  <NUM_GPUS>  --format-only
scripts/test/dist_test.sh projects/configs/spacedrive/spacedrive_plus_qwen.py 8  --format-only
```

Then you can get inference outputs under save_path ended with 'results_planning_only'.

Run the following commands to get L2 error, collision rate and intersection rate under different criteria. 
```bash
# Default
# python scripts/evaluation/eval_planning.py --pred_path <TRAJ_SAVE_PATH> --base_path data/nuscenes/ 
python scripts/evaluation/eval_planning.py --pred_path workspace/spacedrive_plus_qwen/_results_planning_only --base_path data/nuscenes/ 


# VAD metrics, see Tab. G in Supplementary
# python scripts/evaluation/eval_planning_vad.py --pred_path <TRAJ_SAVE_PATH> --base_path data/nuscenes/ 
python scripts/evaluation/eval_planning_vad.py --pred_path workspace/spacedrive_plus_qwen/_results_planning_only --base_path data/nuscenes/ 

# UniAD metrics, see Tab. G in Supplementary
# python scripts/evaluation/eval_planning_uniad.py --pred_path <TRAJ_SAVE_PATH> --base_path data/nuscenes/ 
python scripts/evaluation/eval_planning_uniad.py --pred_path workspace/spacedrive_plus_qwen/_results_planning_only --base_path data/nuscenes/ 
```
