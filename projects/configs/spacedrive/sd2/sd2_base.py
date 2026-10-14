# SD2 experiment series: common base for the geometry improvements (BEV occupancy head, UniDepth FiLM,
# depth-guided epipolar adapter). Identical to spacedrive_plus_qwen_depth_skip.py (depth skip 'add' + UniDepth
# on GT intrinsics, base Qwen2.5-VL weights) plus
#   - flash attention + GPU depth-skip preprocessing (same as run.sh)
#   - one checkpoint per epoch, keep 2 (18.5 GB each)
#   - ValLossHook every 1/4 epoch (also averages tf_*/occ_*/epi_* metrics), GenL2Hook (free-running
#     generation L2 on 8 val scenes) every epoch
# The work dir is set by the queue (tools/queue/run_job.sh --work-dir); base_path below is only a fallback.
_base_ = ['../spacedrive_plus_qwen_depth_skip.py']

num_iters_per_epoch = 28130 // (4 * 8)

base_path = '/workspace/results/spacedrive_sd2_base/'
work_dir = base_path
results_path = base_path + '_results_planning_only/'

model = dict(
    save_path=results_path,
    attn_implementation='flash_attention_2',
    depth_skip_gpu_preprocess=True,
)

checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=2)

custom_hooks = [
    dict(
        type='ValLossHook',
        interval=num_iters_per_epoch // 4,
        num_scenes=16,
        workers=2,
        dataset=dict(
            type='CustomNuScenesDataset',
            data_root='./data/nuscenes/',
            ann_file='./data/nuscenes/nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl',
            seq_mode=True,
            seq_split_num=1,
            pipeline={{_base_.val_loss_pipeline}},
            classes={{_base_.class_names}},
            modality={{_base_.input_modality}},
            test_mode=False,
            use_valid_flag=True,
            filter_empty_gt=False,
            box_type_3d='LiDAR')),
    dict(
        type='GenL2Hook',
        interval=num_iters_per_epoch,
        num_scenes=8,
        workers=2,
        dataset=dict(
            type='CustomNuScenesDataset',
            data_root='./data/nuscenes/',
            ann_file='./data/nuscenes/nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl',
            pipeline={{_base_.test_pipeline}},
            classes={{_base_.class_names}},
            modality={{_base_.input_modality}},
            test_mode=True,
            box_type_3d='LiDAR')),
]
