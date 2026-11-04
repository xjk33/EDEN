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
    imports=['mydiffae.dt_monitor_hook', 'mydiffae.tweight_monitor_hook'],
    allow_failed_imports=False)

# ===== t_weight 采样应用 + 正值竞争 + domain（终版形态）=====
# 与 tsample 对照轮的唯一差别：计分从全局带符号换成正值竞争 + domain=[-2.5, 2.2]
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
    # ---- ε 训练、mask 关 ----
    diffae_pred_x0=False,
    diffae_mask_ratio=0.0,
    # ---- t 加权：D_eps 正值竞争 + domain，采样应用 ----
    pcgrad_enabled=False,
    t_weight_enabled=True,
    t_weight_joint=False,
    t_weight_apply='sample',           # t_weight 作为 t 的采样分布
    t_weight_pos_only=True,            # 正值竞争（负贡献/域外吃地板）
    # t_weight_domain=[-2.5, 2.2],       # logSNR 定义域 ≈ 已验证的 [100,500] 窗口
    t_weight_z_cap=1.0,
    t_weight_warmup_iters=5000,
    t_weight_eta=2.0,
    t_weight_eps=0.1,
    t_weight_bins=16,
    t_weight_interval=500,
    t_weight_ema=0.95,
    # ---- 可视化 ----
    vis_interval=12000,
    vis_max_samples=4,
    vis_dir='./work_dirs/ablation_experiment/diffae_tsampler_pos_36k/vis_diffae'
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
    dict(type='TWeightMonitorHook'),   # 权重柱 + D 表 + 实测采样频率（绿线）
]

work_dir = './work_dirs/ablation_experiment/diffae_tsampler_pos_36k'