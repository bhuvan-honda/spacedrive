# SpaceDrive results

nuScenes open-loop planning (val, 5119 samples with full 3 s GT). Lower is better everywhere.
- **L2**: ADE over the first 1/2/3 s (m). **Col**: object box collision rate (%). **Bnd**: out-of-drivable-area rate (%).
  `avg` = mean of the 1s/2s/3s values.
- **Train loss** (total `loss` = vlm_loss + loss_pos, rank 0): @ckpt = mean over the last 500 iters before the
  evaluated checkpoint; best = min 10-log-entry rolling mean up to that checkpoint.
- **Val loss** (ValLossHook, teacher-forced, 16 val scenes, val_loss_pos + val_vlm_loss): only for runs trained
  with the hook. Monitor for overfitting only; it does not rank checkpoints by L2 reliably.
- `-` = not available (released weights / no-learning baselines / run without ValLossHook).

Add or update a row (inside the container, from the repo root):
```
python tools/results_md.py add --name <name> --eval-log logs/<x>_eval_log.txt [--inf-log logs/<x>_inf.txt] \
    [--work-dir results/<run>] [--ckpt-iter N] [--notes "..."]
```

<!-- RESULTS:BEGIN -->
| Experiment | Ckpt iter | L2 1s/2s/3s (m) | L2 avg | Col 1s/2s/3s (%) | Col avg | Bnd 1s/2s/3s (%) | Bnd avg | Train loss @ckpt | Train loss best (iter) | Val loss @ckpt (iter) | Val loss best (iter) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_extrap_h3 | - | 0.541 / 1.060 / 1.706 | **1.1023** | 0.273 / 1.113 / 4.376 | **1.921** | 2.110 / 6.994 / 12.639 | **7.248** | - | - | - | - | no learning: const. velocity fit to last 3 poses (1.5 s) |
| baseline_extrap_h6 | - | 0.694 / 1.305 / 2.033 | **1.3438** | 0.352 / 1.563 / 5.821 | **2.579** | 3.204 / 9.006 / 14.202 | **8.804** | - | - | - | - | no learning: const. velocity fit to last 6 poses (3 s) |
| baseline_ctrv | - | 0.235 / 0.526 / 0.954 | **0.5715** | 0.000 / 0.313 / 2.149 | **0.820** | 0.273 / 2.032 / 6.740 | **3.015** | - | - | - | - | no learning: CTRV from current CAN-bus speed + yaw rate |
| spacedrive_qwen | - | 1.067 / 1.817 / 2.578 | **1.8203** | 0.352 / 1.231 / 3.751 | **1.778** | 0.899 / 3.966 / 8.478 | **4.447** | - | - | - | - | released SpaceDrive weights, NO ego status |
| spacedrive_plus_qwen | 21096 | 0.152 / 0.294 / 0.516 | **0.3206** | 0.039 / 0.176 / 0.469 | **0.228** | 0.234 / 0.820 / 2.852 | **1.302** | - | - | - | - | released SpaceDrive+ weights |
| plus_new_train | 21096 | 0.155 / 0.296 / 0.519 | **0.3230** | 0.000 / 0.078 / 0.449 | **0.176** | 0.234 / 0.762 / 2.637 | **1.211** | 0.0779 | 0.0738 (19650) | - | - | re-train of plus cfg (run-to-run noise reference) |
| depth_skip | 21096 | 0.155 / 0.295 / 0.515 | **0.3216** | 0.039 / 0.176 / 0.645 | **0.287** | 0.215 / 0.684 / 2.520 | **1.140** | 0.0851 | 0.0816 (19150) | - | - | depth skip concat v1, 8 GPU |
| depth_skip_new | 21096 | 0.155 / 0.293 / 0.511 | **0.3197** | 0.039 / 0.078 / 0.371 | **0.163** | 0.215 / 0.860 / 2.911 | **1.328** | 0.1000 | 0.0837 (20100) | - | - | depth skip (new code) |
| depth_skip_add | 21096 | 0.151 / 0.293 / 0.519 | **0.3212** | 0.039 / 0.156 / 0.449 | **0.215** | 0.215 / 0.820 / 3.106 | **1.380** | 0.0771 | 0.0753 (19650) | - | - | depth skip add v1 (pre-adapter, bc884cb) |
| depth_skip_add_v2_fulllearning | 21096 | 0.155 / 0.297 / 0.520 | **0.3239** | 0.000 / 0.059 / 0.449 | **0.169** | 0.234 / 0.742 / 2.637 | **1.205** | 0.0715 | 0.0669 (20850) | - | - | depth skip add v2 (zero-init adapter) |
| depth_skip_concat_v2 | 21096 | 0.262 / 0.508 / 0.850 | **0.5401** | 0.000 / 0.098 / 0.391 | **0.163** | 0.215 / 0.938 / 2.637 | **1.263** | 0.1336 | 0.1213 (19600) | - | - | 4 GPU, lr 2.5e-5; regression |
| depth_orth_hard | 21096 | 0.155 / 0.295 / 0.515 | **0.3218** | 0.020 / 0.078 / 0.547 | **0.215** | 0.273 / 0.801 / 3.087 | **1.387** | 0.0800 | 0.0788 (21000) | - | - | 4 GPU bs2, hard orthogonal depth fusion |
| codebook_add | 10548 | 0.288 / 0.420 / 0.658 | **0.4552** | 0.039 / 0.254 / 0.801 | **0.365** | 1.544 / 3.068 / 6.175 | **3.596** | 4.0674 | 4.0674 (10500) | - | - | codebook add; stopped at half schedule; dropped; 5117 samples |
| realK | 7032 | 0.148 / 0.280 / 0.499 | **0.3090** | 0.000 / 0.039 / 0.488 | **0.176** | 0.273 / 0.879 / 2.989 | **1.380** | 0.1284 | 0.1269 (6450) | - | - | add + real intrinsics, 6x4, 6 ep, lr 1.5e-4 |
| depthlm_v1 | 7032 | 0.268 / 0.428 / 0.705 | **0.4671** | 0.078 / 0.313 / 0.996 | **0.462** | 0.234 / 1.074 / 3.477 | **1.595** | 0.2054 | 0.1837 (4650) | - | - | DepthLM v1 LoRA merged init, realK cfg |
| depthlm_v2_16243 | 16243 | 0.152 / 0.303 / 0.546 | **0.3334** | 0.000 / 0.117 / 0.684 | **0.267** | 0.215 / 0.567 / 2.149 | **0.977** | 0.0486 | 0.0486 (16200) | - | - | DepthLM v2 init, 8x4, 24 ep sched |
| depthlm_v2 | 21096 | 0.150 / 0.298 / 0.533 | **0.3271** | 0.000 / 0.137 / 0.664 | **0.267** | 0.215 / 0.547 / 2.129 | **0.964** | 0.0215 | 0.0209 (20200) | - | - | DepthLM v2 init, 8x4, 24 ep (final) |
| depthlm_v3_12292 | 12292 | 0.167 / 0.302 / 0.523 | **0.3306** | 0.000 / 0.117 / 0.488 | **0.202** | 0.195 / 0.547 / 1.895 | **0.879** | 0.1029 | 0.0983 (12150) | 0.1276 (12264) | 0.1276 (12264) | v2 re-run (2nd seed-ish), 4x8, flash-attn, ValLossHook |
| depthlm_v3_13609 | 13609 | 0.179 / 0.328 / 0.558 | **0.3549** | 0.059 / 0.137 / 0.469 | **0.221** | 0.234 / 0.645 / 2.149 | **1.009** | 0.0853 | 0.0835 (13500) | 0.1231 (13578) | 0.1231 (13578) | v3 |
| depthlm_v3_14048 | 14048 | 0.158 / 0.314 / 0.555 | **0.3422** | 0.000 / 0.078 / 0.469 | **0.182** | 0.254 / 0.820 / 2.325 | **1.133** | 0.0787 | 0.0787 (14000) | 0.1533 (14016) | 0.1231 (13578) | v3; ckpt kept in results/spacedrive_plus_qwen_depthlm_v3_best_iter14048 |
| depthlm_v3 | 21096 | 0.151 / 0.298 / 0.531 | **0.3266** | 0.000 / 0.078 / 0.527 | **0.202** | 0.215 / 0.606 / 1.856 | **0.892** | 0.0179 | 0.0168 (20750) | 0.1436 (21024) | 0.1231 (13578) | v3 final |
| depthlm_v4 | 5274 | 0.145 / 0.276 / 0.493 | **0.3046** | 0.020 / 0.117 / 0.488 | **0.208** | 0.234 / 0.742 / 2.813 | **1.263** | 0.1232 | 0.1204 (5050) | 0.1203 (5256) | 0.1203 (5256) | DepthLM v2 init, depthlm cfg, 4x8, 6 ep, flash-attn |
| sd2_base_s888 | 5274 | 0.149 / 0.288 / 0.508 | **0.3150** | 0.000 / 0.117 / 0.723 | **0.280** | 0.234 / 0.723 / 2.559 | **1.172** | 0.0924 | 0.0907 (5050) | 0.1354 (5256) | 0.1265 (4380) | SD2 base (realK cfg, 4x8, 6 ep), seed 888 |
| sd2_occ_l14 | 5274 | 0.150 / 0.293 / 0.520 | **0.3211** | 0.039 / 0.176 / 0.703 | **0.306** | 0.234 / 0.879 / 2.950 | **1.354** | 0.6729 | 0.6705 (5000) | 0.1517 (5256) | 0.1396 (4380) | SD2 + BEV occ aux head @ LLM layer 14 (3-class GT, unknown ignored); train loss incl. occ loss |
| sd2_film | 5274 | 0.152 / 0.295 / 0.522 | **0.3232** | 0.000 / 0.059 / 0.586 | **0.215** | 0.234 / 0.801 / 2.637 | **1.224** | 0.0715 | 0.0710 (5200) | 0.1497 (5256) | 0.1361 (3504) | SD2 + UniDepth FiLM into mid LLM layers |
| sd2_base_s889 | 5274 | 0.148 / 0.288 / 0.511 | **0.3159** | 0.059 / 0.117 / 0.449 | **0.208** | 0.215 / 0.723 / 2.559 | **1.166** | 0.0786 | 0.0778 (5200) | 0.1381 (5256) | 0.1335 (4599) | SD2 base, seed 889 (seed-noise reference) |
| sd2_epi | 5274 | 0.144 / 0.278 / 0.495 | **0.3057** | 0.020 / 0.098 / 0.430 | **0.182** | 0.234 / 0.899 / 2.989 | **1.374** | 0.1303 | 0.1258 (5050) | 0.1228 (5256) | 0.1220 (5037) | SD2 + depth-guided epipolar adapter |
| sd2_occ_l28 | 5274 | 0.149 / 0.288 / 0.507 | **0.3145** | 0.000 / 0.078 / 0.625 | **0.234** | 0.254 / 0.742 / 2.657 | **1.218** | 0.6796 | 0.6750 (5000) | 0.1328 (5256) | 0.1241 (4380) | SD2 + BEV occ aux head @ LLM layer 28; train loss incl. occ loss |
| sd2_full | 5274 | 0.151 / 0.288 / 0.511 | **0.3169** | 0.020 / 0.176 / 0.625 | **0.273** | 0.215 / 0.664 / 2.500 | **1.127** | 0.6632 | 0.6598 (5000) | 0.1388 (5256) | 0.1272 (1752) | queue: sd2_full.py seed 888  |
| sd2_epi_dlm_e12 | 10548 | 0.141 / 0.272 / 0.485 | **0.2992** | 0.020 / 0.156 / 0.547 | **0.241** | 0.176 / 0.664 / 2.129 | **0.990** | 0.0916 | 0.0882 (10400) | 0.1130 (10512) | 0.1130 (10512) | queue: sd2_epi_depthlm_e12.py seed 888  |
<!-- RESULTS:END -->

## SD2 ablation summary (ckpt 5274, 4x8, 6 ep)

Delta vs. SD2 base = mean of the two base seeds (L2 0.3155, Col 0.244, Bnd 1.169). Seed spread of the base
(|s888 - s889| / 2): L2 +/-0.0005, Col +/-0.036, Bnd +/-0.003. Negative = better.
Gen-L2 = GenL2Hook free-running L2 avg on 8 val scenes (273 frames) at the last epoch. Val loss = ValLossHook @ckpt.

| Experiment | L2 avg | dL2 | Col avg | dCol | Bnd avg | dBnd | Gen-L2 (val subset) | Val loss | Seeds |
|---|---|---|---|---|---|---|---|---|---|
| sd2_base_s888 | 0.3150 | - | 0.280 | - | 1.172 | - | 0.285 | 0.1354 | 888 |
| sd2_base_s889 | 0.3159 | - | 0.208 | - | 1.166 | - | 0.298 | 0.1381 | 889 |
| sd2_occ_l14 | 0.3211 | +0.0057 | 0.306 | +0.062 | 1.354 | +0.185 | 0.307 | 0.1517 | 888 |
| sd2_occ_l28 | 0.3145 | -0.0010 | 0.234 | -0.010 | 1.218 | +0.049 | 0.290 | 0.1328 | 888 |
| sd2_film | 0.3232 | +0.0078 | 0.215 | -0.029 | 1.224 | +0.055 | 0.299 | 0.1497 | 888 |
| **sd2_epi** | **0.3057** | **-0.0098** | **0.182** | **-0.062** | 1.374 | +0.205 | **0.282** | 0.1228 | 888 |
| depthlm_v4 (init change, not an SD2 module) | 0.3046 | -0.0109 | 0.208 | -0.036 | 1.263 | +0.094 | - | 0.1203 | 888 |
| sd2_full (occ + film + epi) | 0.3169 | +0.0014 | 0.273 | +0.029 | 1.127 | -0.042 | 0.296 | 0.1388 | 888 |
| **sd2_epi_dlm_e12** (epi + DepthLM init, 12 ep) | **0.2992** | **-0.0163** | 0.241 | -0.003 | **0.990** | **-0.179** | **0.268** | **0.1130** | 888 |
| sd2_occ_l28_dlm_e12 (occ L28 + DepthLM init, 12 ep) | running | | | | | | | | 888 |
