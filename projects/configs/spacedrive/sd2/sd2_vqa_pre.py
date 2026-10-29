# Stage-1 VQA pre-training for the SD2 backbones: VQA + planning multitask on the OmniDrive-nuScenes QA
# (keywords, desc, vqa incl. counterfactuals, conv), 4 epochs. The planning turn always comes first; all the
# others are shuffled. Model planning_only=False -> loss_pos on ALL answer coordinates (planning + VQA).
# Backbone: base Qwen weights here; DepthLM via --cfg-options model.lm_head=<DepthLM-v2 dir>.
# The final checkpoint (iter_3516.pth) is the load_from init of the stage-2 runs.
# ValLossHook keeps the planning-only val pipeline of sd2_base (comparable to the other runs).
_base_ = ['./sd2_base.py']

num_epochs = 4
num_iters_per_epoch = 28130 // (4 * 8)
enable_online_vqa = False

model = dict(planning_only=False, grad_ckpt_skip_layers=0)  # ~5k-token VQA samples: keep full checkpointing

# --- train pipeline: copy of spacedrive_plus_qwen.py with planning_only=False (lists are replaced, not merged)
data_root = './data/nuscenes/'
llm_path = 'ckpts/Qwen2.5-VL-7B-Instruct-with-new-special-tokens/'
point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv',
                'command', 'can_bus']

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0, 1, 2, 3, 4]),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, use_dim=[0, 1, 2, 3, 4]),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
         with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type='LoadAnnoatationVQA',
         llm_type='qwenvl25',
         load_3d_pos=True,
         load_ego_command_in_question=True,
         load_high_level_command=False,
         single_token_output=False,
         pseudo_coords=False,
         planning_only=False,
         enable_online_vqa=enable_online_vqa,
         n_gen=2,
         base_vqa_path=data_root + 'vqa/train/',
         base_desc_path=data_root + 'desc/train/',
         base_conv_path=data_root + 'conv/train/',
         base_key_path=data_root + 'keywords/train/',
         tokenizer=llm_path,
         processor=llm_path,
         max_length=131072,
         ignore_type=[],
         num_commands=-1,
         lane_objs_info=data_root + 'lane_obj_train.pkl',
         classes=class_names),  # gt_labels_3d order (online QA class names)
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['has_gt_planning', 'img', 'coords_pos_tensor', 'input_ids', 'vlm_labels',
                                 'pixel_values', 'image_grid_thw', 'prev_exists'] + collect_keys,
         meta_keys=('sample_idx', 'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'img_norm_cfg', 'scene_token'))
]

data = dict(train=dict(pipeline=train_pipeline))

runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
