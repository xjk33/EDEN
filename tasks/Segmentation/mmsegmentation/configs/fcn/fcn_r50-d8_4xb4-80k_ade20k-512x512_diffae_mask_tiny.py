randomness = dict(seed=0, deterministic=False)

_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

custom_imports = dict(
    imports=['mydiffae.dt_monitor_hook', 'mydiffae.mask_monitor_hook'],
    allow_failed_imports=False)

# ===== mask v3：软蒙（均值色混合）+ 动态幅度 U[0,0.3] + hot 选址（x0 预测）=====
train_cfg = dict(max_iters=36000, type='IterBasedTrainLoop', val_interval=12000)
param_scheduler = [
    dict(type='PolyLR', eta_min=0.0001, power=0.9, begin=0, end=36000, by_epoch=False),
]

model = dict(
    type='FCNWithDiffAEPlusPCGrad',
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150),
    image_size=128,
    enc_output_dim=2048,
    cond_dim=512,
    unet_channels=128,
    unet_channel_mult=(1, 1, 2, 3),
    num_diffusion_timesteps=1000,
    beta_schedule='linear',
    spaced_timesteps='ddim100',
    ema_decay=0.9999,
    lambda_diff=1.0,
    use_zero_module=True,
    auto_ema=True,
    num_res_blocks=1,
    # ---- v2 参数化（mask 主线用 x0 单次 loss）----
    diffae_pred_x0=True,
    # ---- mask v3 ----
    diffae_mask_mode='hot',             # 蒙 seg 高敏感块（推有用信息的故事）
    diffae_mask_patch=16,
    diffae_mask_temp=1.0,
    diffae_mask_soft=True,              # 软蒙：加噪前混合到数据集均值色
    diffae_mask_alpha=1.0,              # 混合强度 1.0 = 完全均值色
    diffae_mask_dynamic=True,           # 每图每步 r ~ U[0, ratio_max]
    diffae_mask_ratio_max=0.3,
    diffae_mask_ratio=0.25,             # dynamic 时仅作占位
    # ---- 其余全关 ----
    pcgrad_enabled=False,
    t_weight_enabled=False,
    # ---- 可视化 ----
    vis_interval=12000,
    vis_max_samples=4,
    vis_dir='./work_dirs/ablation_experiment/diffae_mask_v3_36k/vis_diffae'
)

optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={
        'diffae_unet': dict(lr_mult=0.01),
        'diffae_enc_compress': dict(lr_mult=0.01),
    }))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', save_optimizer=False,
        max_keep_ckpts=1, interval=12000, by_epoch=False))

custom_hooks = [
    # dict(type='CheckpointHook', interval=12000, save_optimizer=True,
    #      by_epoch=False, out_dir='./resume_ckpts', max_keep_ckpts=1),
    # dict(type='DTMonitorHook', num_images=16),
    dict(type='MaskMonitorHook', num_images=4),
]

# val_evaluator = [
#     dict(type='IoUMetric', iou_metrics=['mIoU']),
#     dict(type='BoundaryIoU', num_classes=150),
# ]
# test_evaluator = [
#     dict(type='IoUMetric', iou_metrics=['mIoU']),
#     dict(type='BoundaryIoU', num_classes=150),
# ]

work_dir = './work_dirs/ablation_experiment/diffae_mask_v3_36k'