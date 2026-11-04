_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

# 替换为 DiffAE 集成模型
model = dict(
    type='FCNWithDiffAE',
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150),
    # ---- DiffAE 配置 ----
    image_size=128,
    enc_output_dim=2048,        #ResNet-50 最后一层通道数
    cond_dim=512, 
    unet_channels=128,
    unet_channel_mult=(1, 2, 4),
    num_diffusion_timesteps=1000,
    beta_schedule='linear',
    spaced_timesteps='ddim100',
    ema_decay=0.9999,
    lambda_diff=1.0,         # 扩散 loss 权重，可调
    use_zero_module=True,
    auto_ema=True,           # 自动 EMA，不需要额外 Hook
    num_res_blocks=1,
)

# 可选：给 loss_diffae 单独配学习率（通过 paramwise_cfg）
# 如果不需要，注释掉下面这段
# optim_wrapper = dict(
#     paramwise_cfg=dict(
#         custom_keys={
#             'diffae_unet': dict(lr_mult=1.0),
#             'diffae_cond_proj': dict(lr_mult=1.0),
#         }
#     )
# )

val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    dict(type='BoundaryIoU', num_classes=150),
]
test_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    dict(type='BoundaryIoU', num_classes=150),
]

work_dir = './work_dirs/ablation_experiment/baseline_diffae_ade20k'