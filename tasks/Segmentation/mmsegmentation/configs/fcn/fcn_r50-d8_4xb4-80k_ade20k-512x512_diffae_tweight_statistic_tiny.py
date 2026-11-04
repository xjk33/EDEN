_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

# ===== 训练规模：4h 版（与其他 36k 实验成对）=====
train_cfg = dict(max_iters=36000, type='IterBasedTrainLoop', val_interval=6000)
param_scheduler = [
    dict(type='PolyLR', eta_min=0.0001, power=0.9, begin=0, end=36000, by_epoch=False),
]

# +diffae 纯净版：新代码接口，t_weight 关、mask 关。
# 是 mask 实验臂和 t_weight 动态臂的共同对照，三者唯一差异就是下面两个开关。
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
    # ---- t 加权（statistic）----
    t_weight_enabled=True,
    t_weight_static_window=[100, 500],
    # ---- 输入互补蒙块（关闭）----
    diffae_mask_ratio=0.0,
    # ---- 可视化配置 ----
        vis_interval=4000,
        vis_max_samples=4,
        vis_dir='./work_dirs/ablation_experiment/diffae_ade20k_tweight_statistic_36k_1e-4/vis_diffae'
)

optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={
        'diffae_unet': dict(lr_mult=0.01),
        'diffae_enc_compress': dict(lr_mult=0.01),
    }))

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

# bIoU 全臂标配：检验"细长结构类下跌"是否伤在边界上
# val_evaluator = [
#     dict(type='IoUMetric', iou_metrics=['mIoU']),
#     dict(type='BoundaryIoU', num_classes=150),
# ]
# test_evaluator = [
#     dict(type='IoUMetric', iou_metrics=['mIoU']),
#     dict(type='BoundaryIoU', num_classes=150),
# ]

work_dir = './work_dirs/ablation_experiment/diffae_ade20k_tweight_statistic_36k_1e-4'