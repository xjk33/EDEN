_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

# ===== 训练规模：4h 版 =====
train_cfg = dict(max_iters=36000, type='IterBasedTrainLoop', val_interval=6000)
param_scheduler = [
    dict(type='PolyLR', eta_min=0.0001, power=0.9, begin=0, end=36000, by_epoch=False),
]

# 最终组合实验：t_weight 动态加权 + 梯度热力图互补蒙块
# 与 diffae_only_36k / mask_grad_36k 成三臂对照；若 mask 已救活通路，
# 本 run 的 D 表有望第一次出现结构，t_weight 从"均匀空转"转入真实追踪。
model = dict(
    type='FCNWithDiffAEPlusPCGrad',
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150),
    # ---- DiffAE 配置（与 baseline 一致）----
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
    # ---- PCGrad（关闭）----
    pcgrad_enabled=False,
    pcgrad_lambda_d=1.0,
    pcgrad_encoder_names=('backbone',),
    pcgrad_mode='global',
    # ---- t 加权（动态）----
    t_weight_enabled=True,
    t_weight_static_window=None,
    t_weight_warmup_iters=5000,
    t_weight_eta=2.0,
    t_weight_eps=0.1,
    t_weight_bins=16,
    t_weight_interval=500,
    t_weight_ema=0.95,
    # ---- 梯度热力图互补蒙块 ----
    diffae_mask_ratio=0.25,
    diffae_mask_patch=16,
    diffae_mask_aux_weight=0.1,      # 必须是 0.1：1.0 已实测把 trunk 拽到 NaN
    diffae_mask_temp=1.0,
    diffae_mask_aux_gate=0.3,
    # ---- 可视化配置 ----
    vis_interval=4000,
    vis_max_samples=4,
    vis_dir='./work_dirs/ablation_experiment/diffae_ade20k_tweight_dynamic_mask_36k/vis_diffae'
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_optimizer=False,
        max_keep_ckpts=3,
        interval=12000,
        by_epoch=False
    )
)

custom_hooks = [
    dict(
        type='CheckpointHook',
        interval=12000,
        save_optimizer=True,
        by_epoch=False,
        out_dir='./resume_ckpts',
        max_keep_ckpts=1,
    )
]

# bIoU：boundary_iou_metric.py 注册后解开
# val_evaluator = [
#     dict(type='IoUMetric', iou_metrics=['mIoU']),
#     dict(type='BoundaryIoU', num_classes=150),
# ]
# test_evaluator = [
#     dict(type='IoUMetric', iou_metrics=['mIoU']),
#     dict(type='BoundaryIoU', num_classes=150),
# ]

work_dir = './work_dirs/ablation_experiment/diffae_ade20k_tweight_dynamic_mask_36k'