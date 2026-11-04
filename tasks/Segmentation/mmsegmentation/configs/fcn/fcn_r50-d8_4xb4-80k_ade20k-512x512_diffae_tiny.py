randomness = dict(seed=0, deterministic=False)

_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

# Hook 注册（D(t) 监控）
custom_imports = dict(
    imports=['mydiffae.dt_monitor_hook'],
    allow_failed_imports=False)

# ===== 训练规模：4h 版 =====
train_cfg = dict(max_iters=36000, type='IterBasedTrainLoop', val_interval=12000)
# PolyLR 逐参数组作用：fcn 组 0.01 -> 1e-4 正常衰减；
# diffae 组初始 lr = 0.01 x 0.01 = 1e-4 == eta_min -> 恒定 1e-4
param_scheduler = [
    dict(type='PolyLR', eta_min=0.0001, power=0.9, begin=0, end=36000, by_epoch=False),
]

# 单独 diffae + UNet/压缩层恒定 1e-4 + 每次验证记录 D(t) 曲线
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
    # ---- PCGrad / t 加权 / mask（全部关闭，纯 diffae）----
    pcgrad_enabled=False,
    t_weight_enabled=False,
    diffae_pred_x0=False,
    diffae_mask_ratio=0.0,
    # ---- 可视化配置 ----
    vis_interval=12000,
    vis_max_samples=4,
    vis_dir='./work_dirs/ablation_experiment/diffae_36k_re/vis_diffae'
)

# 标准 paramwise 分组（mmseg 官方写法）：unet/压缩层 lr 乘 0.01 -> 恒定 1e-4
# 前提：mixin 已去掉 ModuleDict 重复注册（diffae_mixin_nodupl.py）
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={
        'diffae_unet': dict(lr_mult=0.01),
        'diffae_enc_compress': dict(lr_mult=0.01),
    }))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_optimizer=False,
        max_keep_ckpts=1,
        interval=12000,
        by_epoch=False
    )
)

custom_hooks = [
    # dict(
    #     type='CheckpointHook',
    #     interval=12000,
    #     save_optimizer=True,
    #     by_epoch=False,
    #     out_dir='./resume_ckpts',
    #     max_keep_ckpts=1,
    # ),
    # dict(type='DTMonitorHook', num_images=16),   # 每次验证后存 D(t) 图 + npz
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

work_dir = './work_dirs/ablation_experiment/diffae_36k_re'