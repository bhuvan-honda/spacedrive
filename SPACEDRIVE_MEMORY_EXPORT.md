# SpaceDrive Research — Portable Memory Export

> HOW TO USE IN ANOTHER REPO:
> 1. Copy this file into the other repo (anywhere, e.g. repo root or docs/).
> 2. In that repo's VS Code, open Copilot chat and say:
>    "Read SPACEDRIVE_MEMORY_EXPORT.md and save its contents into repository memory."
>    (Copilot will write it into that workspace's /memories/repo/.)
> 3. Alternatively paste relevant sections into that repo's .github/copilot-instructions.md.
> NOTE: This is SpaceDrive-specific knowledge. Only import it into a repo where it's relevant.

====================================================================
## 0. ENVIRONMENT & HOW TO RUN (critical)
====================================================================
- HOST HAS NO PYTHON. Everything runs inside Docker container `spacedrive_local`,
  repo mounted at /workspace. Run via:
    docker exec spacedrive_local bash -lc 'cd /workspace && <cmd>'
- Host-side validation = static lint only. No python on host.
- Train:  scripts/train/dist_train.sh <config> <num_gpus>
- Test:   scripts/test/dist_test.sh <config> <num_gpus> --format-only >> <log>
- Eval:   python scripts/evaluation/eval_planning.py --pred_path <results>/_results_planning_only/ --base_path data/nuscenes/
- run.sh = user's scratch file for the above command lines.
- ckpts present: ckpts/Qwen2.5-VL-7B-Instruct (base),
  Qwen2.5-VL-7B-Instruct-with-new-special-tokens (2 tokens),
  Qwen2.5-VL-7B-Instruct-with-codebook (258 tokens, built this session).

====================================================================
## 1. ARCHITECTURE OVERVIEW
====================================================================
SpaceDrive = autonomous-driving VLM on mmdetection3d + Qwen2.5-VL-7B. Fuses 3D positional
encodings into a VLM for NuScenes ego planning (predict 6 future (x,y) BEV waypoints).

Directory map:
- projects/configs/spacedrive/spacedrive_plus_qwen.py  : main mmcv config
- projects/mmdet3d_plugin/models/vlm/spacedrive.py     : core model SpaceDrive(MVXTwoStageDetector)
- projects/mmdet3d_plugin/models/vlm_utils/custom_qwen.py : CustomQwen2_5_VLForConditionalGeneration + CustomQwen2_5_VLModel
- projects/mmdet3d_plugin/models/vlm_utils/positional_encoding.py : PositionalEncoding3D + RoPE3D
- projects/mmdet3d_plugin/models/vlm_utils/misc.py     : load_model() LoRA PEFT, locations()
- projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py : LoadAnnoatationVQA / ...Test
- projects/mmdet3d_plugin/datasets/qwen_utils/data_qwen.py   : preprocess_qwen_2_visual()
- projects/mmdet3d_plugin/datasets/utils/constants.py  : token IDs
- projects/mmdet3d_plugin/datasets/samplers/group_sampler.py : InfiniteGroupEachSampleInBatchSampler
- scripts/train|test, scripts/evaluation/eval_planning.py
- tools/token_additor.py : adds special tokens, resizes embeddings

Key facts:
- LLM Qwen2.5-VL-7B-Instruct, bf16, LoRA rank16 on q/k/v/o. llm_hidden_dim=3584.
- Depth: UniDepthV2-ViT-L14 frozen (depth_model_type='unidepth'). ViT frozen; only LoRA+adapters+heads train.
- Token IDs: IMAGE=151655, POS_INDICATOR=151665, POS_EMBEDDING=151666, COORD_TOKEN_START=151667.
- SpaceDrive vs SpaceDrive+ = ONLY config ego_status: ''=baseline, 'feature+PE'=plus. No separate class.
  plus injects: feature (MLP over canbus/egopose -> 1 extra image token) + PE (past ego xyz encoded).
- pos_embed shape (B, 6*23*23=3174, hidden). N_depth == N_rgb == 3174 (same grid).

The "semantic gap" (baseline coord handling):
- Pipeline replaces "(x,y)" text with "<POS_INDICATOR><POS_EMBEDDING>"; real xy in coords_pos_tensor.
- POS_INDICATOR: discrete CE-supervised token (weighted 3x), signals "coord next".
- POS_EMBEDDING: input-embed overwritten by continuous 3D PE; its LABEL is IGNORE.
- Output: hidden state at POS_EMBEDDING -> mlp_output_coords (2-layer MLP) -> (x,y), Huber loss.
  => continuous regression head, SEPARATE from discrete CE/language space.

Supervision fact: data_qwen.py preprocess_qwen_2_visual CE-supervises assistant answers, masking
only the first 3 answer tokens (target_mask[:3]=IGNORE). So tokens in answers auto-train.

====================================================================
## 2. KEY CODE LOCATIONS (fusion / forward)
====================================================================
- custom_qwen.py CustomQwen2_5_VLModel.forward (~L735-795): image_embeds=get_image_features;
  capture image_embeds_pure; +pos_emb add (~L771); depth 'add' fusion (~L775); ego_feature concat;
  masked_scatter into inputs_embeds.
- custom_qwen.py ForConditionalGeneration.forward: weighted_mask/io_coords_pos_mask only if
  io_coords_pos not None; logits new-token splice via index_copy(POS_INDICATOR..+num_new_tokens);
  returns CustomQwen2_5_VLCausalLMOutputWithPast (train variant w/ io mask, + test variant).
- spacedrive.py forward_train_vlm (~L656): depth_prediction -> position_embeding(pos_embed,coords3d)
  -> depth_skip build -> ego insert -> lm_head(...). generate() x2 (qwen/llava) in test_generation_pts.
- Ego memory: reset_memory/pre_update_memory/post_update_memory, buffers new_zeros(B,...).

====================================================================
## 3. EXPERIMENT 1 — Depth Skip Connection (IMPLEMENTED, active, small boost)
====================================================================
Encode predicted depth map with frozen Qwen ViT -> trainable adapter -> fuse with RGB vision tokens.
- spacedrive.py: params depth_skip(False), depth_skip_mode('concat'|'add').
  self.depth_adapter = bottleneck MLP (hidden->hidden/4->hidden), last layer zero-init (identity start).
- build_depth_skip_feature(depth,img,pos_embed,add_pe): resize depth->RGB res, per-view minmax->uint8
  3ch, processor.image_processor, inner get_image_features (self.lm_head.base_model.model.model),
  adapter, +pos_embed if add_pe. returns (B,N_tokens,hidden).
- 'concat' mode: depth tokens appended as extra vision tokens (ride ego_feature channel); bare
  IMAGE_TOKEN*n_depth + trailing VISION_END inserted after last RGB vision_end; labels IGNORE.
  Ordering invariant: placeholders L->R = rgb, depth, ego, ego-PE == feature concat order.
- 'add' mode: depth added element-wise onto RGB tokens inside custom_qwen (image_embeds += depth_feature),
  no extra tokens, add_pe=False. depth_feature threaded through custom_qwen ForCondGen -> model.forward,
  guarded `if depth_feature is not None` (prefill only).
- config: spacedrive_plus_qwen_depth_skip.py (inherits plus; depth_skip=True).

====================================================================
## 4. EXPERIMENT 2 — Codebook + Bidirectional (IMPLEMENTED, ran; DROPPED from paper)
====================================================================
Discretize coords into REAL vocab tokens <COORD_0..255> so trajectory+language share one vocab+CE.
Predictions diverse (339/400 unique) & sensible. Quantization=0.4m. USER DROPPED it (avoid "ensemble
of ideas"), focus on DEPTH. Code remains, gated behind use_codebook.
- constants.py: COORD_TOKEN_START=151667, COORD_NUM_BINS=256, COORD_MIN/MAX=-51.2/51.2, NUM_NEW_TOKENS=258.
- NEW datasets/utils/codebook.py: value<->bin<->token; replace_coords_with_tokens (fwd),
  replace_tokens_with_coords (decode). waypoint=2 tokens (x,y), shared codebook.
- tools/token_additor.py add_codebook_tokens(): built ckpts/Qwen2.5-VL-7B-Instruct-with-codebook/
  (vocab 151923; verified single-id POS_IND=151665,POS_EMB=151666,COORD_0=151667,COORD_255=151922).
- misc.py: finetune_new_token generalized 2->NUM_NEW_TOKENS when 'with-codebook' in path.
- custom_qwen.py: logits index_copy range = arange(POS_INDICATOR, +new_logits.shape[-1]);
  embed is_new_token = input_ids>=151665, rel id = input_id-151665.
- transform_3d.py LoadAnnoatationVQA: +use_codebook,+bidirectional. bidirectional appends reverse
  source (traj tokens in Q -> command_desc/action NL in A).
- spacedrive.py: +use_codebook; test decode skip_special_tokens=False -> replace_tokens_with_coords.
- config: spacedrive_plus_qwen_codebook.py (io_3d_pos=False, use_codebook/bidirectional=True, planning_only=True).
BUGS FIXED:
- custom_qwen.py unconditional `has_gt_planning.any()` crashed when io_3d_pos=False (None). Wrapped the
  output_pos/gt_pos/gt_coords_xy selection block in `if io_coords_pos_mask is not None:`.
- eval_planning.py: added `if 'codebook' in pred_path:` branch BEFORE 'spacedrive' branch (path has both);
  codebook answers are plain "[(x,y),...]" text -> plain regex. Else samples=0 -> ZeroDivisionError.
RISK: 0.4m quantization may raise L2 vs continuous MLP head. Mitigate: hybrid bin+residual, finer bins.

====================================================================
## 5. EXPERIMENT 3 — Orthogonal Depth Fusion (IMPLEMENTED, not trained) — CURRENT PAPER THREAD
====================================================================
User idea: depth tokens corrupt vision tokens when added (v_i+p_i+d_i); make depth orthogonal so model
separates geometry from semantics.
SANITY-CHECK CORRECTION: user said "depth orth to EACH OTHER" (d_i⊥d_j) = WRONG axis (doesn't protect
v_i; destroys depth smoothness). CORRECT = depth ⊥ APPEARANCE per token (d_i⊥v_i), like additive
positional encodings (linear separability). Caveat: ⊥ gives linear separability NOT zero-change
(transformer nonlinear) => soft prior. Also need MAGNITUDE control (post-LN renorm) => learnable scalar.
Framing = "get concat-quality separation at add-mode cost."
IMPLEMENTATION (all behind depth_skip & depth_skip_mode=='add'; baseline/concat untouched):
- spacedrive.py __init__: params depth_orth_mode('none'|'hard'|'soft'), depth_orth_lambda=1.0,
  depth_scale_init=1.0. self.depth_scale=nn.Parameter(tensor(init)) created ONLY in add mode.
- custom_qwen.py: +depth_orth_mode/depth_scale threaded ForCondGen.forward -> model.forward.
  CustomQwen2_5_VLCausalLMOutputWithPast +field depth_orth_loss (both return dataclasses set from
  getattr(self.model,'last_depth_orth_loss',None)).
  Inner model.forward: image_embeds_pure = image_embeds right after get_image_features (before pos_emb).
  Fusion (~L775): self.last_depth_orth_loss=None; if depth_feature is not None:
    hard: ref=image_embeds_pure.detach(); dot=(d*ref).sum(-1,kd); denom=(ref*ref).sum(-1,kd).clamp_min(1e-6);
          d = d-(dot/denom)*ref  (per-token Gram-Schmidt).
    soft: cos=cosine_similarity(d.float,ref.float,-1); last_depth_orth_loss=(cos**2).mean().
    scale = depth_scale.to(dtype) if not None else 1.0; image_embeds += scale*d.
- spacedrive.py: pass depth_orth_mode/depth_scale at lm_head() + BOTH generate() (guard: add mode else none/None).
  After losses.update(vlm_loss): if add+soft and orth not None: losses.update(depth_orth_loss=lambda*orth).
- NEW config spacedrive_plus_qwen_depth_orth.py: inherits depth_skip config; depth_skip_mode='add',
  depth_orth_mode='hard', lambda=1.0, scale_init=1.0.
EXPERIMENTS TO RUN: add vs add+orth-hard vs add+orth-soft vs concat (L2+collision). Analysis fig:
hist cos(d_i,v_i) before/after; token-norm inflation ||v+p+d||/||v+p||.

====================================================================
## 6. MULTI-BATCH (true B>1) — ENABLED
====================================================================
- Infra ALREADY supports streaming B>1: config uses InfiniteGroupEachSampleInBatchSampler
  (group_sampler.py L152/L268). __iter__ yields curr_batch of len samples_per_gpu; each slot keeps its
  own continuous temporal buffer (buffer_per_local_sample) -> B parallel temporal streams. builder.py:
  with this sampler + IterBasedRunner, batch_sampler=sampler, DataLoader batch_size=1.
- Ego memory, position_embeding, custom_qwen fusion/scatter, token insertion (input_ids[0] safe: uniform
  vision prefix) all already B-general.
- FIXES for B>1: (1) build_depth_skip_feature: removed `assert B==1`. (2) depth_prediction unidepth:
  intrinsics `.squeeze(0)` -> `.reshape(-1,3,3)` (camera var unused by infer(); avoids Pinhole shape error).
- CONFIG depth_orth: batch_size=2 + recomputed in CHILD: num_iters_per_epoch=28130//(num_gpus*bs),
  lr=1e-4/2*bs/8*num_gpus, data=dict(samples_per_gpu=bs), optimizer=dict(lr=lr), runner max_iters,
  checkpoint/eval intervals. num_gpus=4, num_epochs=6.
- WATCH: B=2 ~2x activation memory -> may OOM; fall back to 1. Alt: GradientCumulativeFp16OptimizerHook.

====================================================================
## 7. mmcv CONFIG GOTCHAS
====================================================================
- Child config CANNOT see parent's raw python vars. Re-declare point_cloud_range, class_names,
  data_root, collect_keys etc. if needed. Derived values (num_iters, lr) must be recomputed in child.
- dict fields deep-merge by key (data=dict(samples_per_gpu=..) merges, keeps rest; optimizer=dict(lr=..)
  keeps constructor/paramwise). runner must be fully re-specified (type+max_iters).
- base batch-dependent: num_iters_per_epoch=28130//(num_gpus*batch_size), lr=1e-4/2*bs/8*num_gpus.

====================================================================
## 8. USER PREFERENCES / DIRECTION
====================================================================
- PAPER THREAD = DEPTH ONLY. Drop codebook + coarse-to-fine (avoid "ensemble of ideas" look).
- Thesis: "monocular depth should GROUND vlm visual tokens in metric 3D geometry, not just add appearance."
  Idea menu (ranked): #1 depth-derived per-token 3D PE (unproject patch centers via depth+intrinsics ->
  position_encoder); #2 depth representation ablation (raw z, 1/z, log, HEIGHT, normals, xyz); #3 fusion
  ablation concat/add/gated-FiLM; #4 confidence-weighted depth (UniDepth conf); #5 aux geometric supervision.
  Orthogonal fusion (EXP3) is the current concrete step.
- Do NOT create markdown summary files unless asked. Prefer batched edits. Keep changes flag-gated so
  baseline paths stay untouched.
