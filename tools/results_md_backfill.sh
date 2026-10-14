#!/usr/bin/env bash
# Rebuild results.md from all existing eval logs (run inside the container from the repo root).
# New experiments: just run `python tools/results_md.py add ...` once; this file is only for the backfill.
set -e
A="python tools/results_md.py add"
L=logs

$A --name baseline_extrap_h3   --eval-log $L/spacedrive_baseline_extrap_h3_eval_log.txt   --notes "no learning: const. velocity fit to last 3 poses (1.5 s)"
$A --name baseline_extrap_h6   --eval-log $L/spacedrive_baseline_extrap_h6_eval_log.txt   --notes "no learning: const. velocity fit to last 6 poses (3 s)"
$A --name baseline_ctrv        --eval-log $L/spacedrive_baseline_ctrv_eval_log.txt        --notes "no learning: CTRV from current CAN-bus speed + yaw rate"
$A --name spacedrive_qwen      --eval-log $L/spacedrive_qwen_eval_log.txt                 --notes "released SpaceDrive weights, NO ego status"
$A --name spacedrive_plus_qwen --eval-log $L/spacedrive_plus_qwen_eval_log.txt            --notes "released SpaceDrive+ weights"
$A --name plus_new_train       --eval-log $L/spacedrive_plus_qwen_new_train_eval_log.txt  --notes "re-train of plus cfg (run-to-run noise reference)"
$A --name depth_skip           --eval-log $L/spacedrive_plus_qwen_depth_skip_eval_log.txt --notes "depth skip concat v1, 8 GPU"
$A --name depth_skip_new       --eval-log $L/spacedrive_plus_qwen_depth_skip_new_eval_log.txt --notes "depth skip (new code)"
$A --name depth_skip_add       --eval-log $L/spacedrive_plus_qwen_depth_skip_add_eval_log.txt --notes "depth skip add v1 (pre-adapter, bc884cb)"
$A --name depth_skip_add_v2_fulllearning --eval-log $L/spacedrive_plus_qwen_depth_skip_add_v2_fulllearning_eval_log.txt --notes "depth skip add v2 (zero-init adapter)"
$A --name depth_skip_concat_v2 --eval-log $L/spacedrive_plus_qwen_depth_skip_concat_v2_eval_log.txt --notes "4 GPU, lr 2.5e-5; regression"
$A --name depth_orth_hard      --eval-log $L/spacedrive_plus_qwen_depth_orth_hard_eval_log.txt --notes "4 GPU bs2, hard orthogonal depth fusion"
$A --name codebook_add         --eval-log $L/spacedrive_plus_qwen_codebook_add_eval_log.txt --notes "codebook add; stopped at half schedule; dropped"
$A --name realK                --eval-log $L/spacedrive_plus_qwen_depth_skip_add_real_intrinsics_eval_log.txt --ckpt-iter 7032 --notes "add + real intrinsics, 6x4, 6 ep, lr 1.5e-4"
$A --name depthlm_v1           --eval-log $L/spacedrive_plus_qwen_depthlm_eval_log.txt    --notes "DepthLM v1 LoRA merged init, realK cfg"
$A --name depthlm_v2_16243     --eval-log $L/spacedrive_plus_qwen_depthlm_v2_16243_eval_log.txt --notes "DepthLM v2 init, 8x4, 24 ep sched"
$A --name depthlm_v2           --eval-log $L/spacedrive_plus_qwen_depthlm_v2_eval_log.txt --notes "DepthLM v2 init, 8x4, 24 ep (final)"
$A --name depthlm_v3_12292     --eval-log $L/spacedrive_plus_qwen_depthlm_v3_12292_eval_log.txt --notes "v2 re-run (2nd seed-ish), 4x8, flash-attn, ValLossHook"
$A --name depthlm_v3_13609     --eval-log $L/spacedrive_plus_qwen_depthlm_v3_13609_eval_log.txt --notes "v3"
$A --name depthlm_v3_14048     --eval-log $L/spacedrive_plus_qwen_depthlm_v3_14048_eval_log.txt --notes "v3; ckpt kept in results/spacedrive_plus_qwen_depthlm_v3_best_iter14048"
$A --name depthlm_v3           --eval-log $L/spacedrive_plus_qwen_depthlm_v3_eval_log.txt --notes "v3 final"
