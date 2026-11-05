# EDEN 消融臂 1/4：纯 DiffAE（t 自适应关、结构化噪声关）
# 用法：python tools/train.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512.py']

# 训练期零监控：分析全部离线（visual/probe_*.ipynb）。
randomness = dict(seed=0, deterministic=False)

custom_imports = dict(
    imports=['mmseg.models.segmentors.eden_adapter_fcn'],
    allow_failed_imports=False)

model = dict(
    type='FCNWithDiffAE',
    # ---- DiffAE 配置 ----
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
    # ---- ε 预测、蒙块关、t 自适应关 ----
    diffae_pred_x0=False,
    diffae_mask_ratio=0.0,
    diffae_mask_dynamic=False,
    t_weight_enabled=False,
)

# =====================================================================
# 负收益排查 v2：SODA 式恒定比例 lr + 提高训练量
#
# [改动 1] lr 比例（仿 SODA：encoder 有效 lr = lr_scale × decoder lr，全程恒定）
#   旧配置问题：FCN(encoder) 0.01 poly→1e-4，UNet(decoder) 因 base lr 恰等于
#   PolyLR 的 eta_min=1e-4 而恒定 1e-4，encoder/decoder 比例从 100:1 缩到 1:1。
#   现改为：FCN 全体恒定 unet_lr × lr_scale = 2e-4，扩散分支 lr_mult=1/lr_scale
#   回到恒定 1e-4，比例锁死 2:1（SODA ImageNet 消融最优值）。
#   注意：同调度 FCN 基线对照见 fcn_r50-d8_bs64-320k_ade20k-512x512_baseline.py，
#   旧 80k/0.01-poly 基线在新 lr 体制下不可比。
#
# [改动 2] 训练量：iters 80k→320k（优化步数对齐 DiffAE ffhq64 的 560k 量级）。
#   bs 保持单卡 16：实测 bs16≈30G 已近 A100 上限，且 bs 增大不省时（吞吐速率不变），
#   训练量差距本质是优化步数（原 80k vs DiffAE 560k / SODA 156k）。
#   320k × bs16 = 5.1M 样本（~253 epochs），4× 于原实验。
# =====================================================================
unet_lr = 1e-4   # 扩散分支（decoder）恒定 lr
lr_scale = 2.0   # encoder/decoder lr 比例超参（SODA 消融最优 2）

train_dataloader = dict(batch_size=16, num_workers=8)

max_iters = 320000
val_interval = 16000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)

optimizer = dict(type='SGD', lr=unet_lr * lr_scale, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None,
    paramwise_cfg=dict(custom_keys={
        'diffae_unet': dict(lr_mult=1.0 / lr_scale),
        'diffae_enc_compress': dict(lr_mult=1.0 / lr_scale),
    }))

# 恒定 lr：encoder 2e-4 / decoder 1e-4，比例全程 = lr_scale
param_scheduler = [
    dict(type='ConstantLR', factor=1.0, begin=0, end=max_iters, by_epoch=False)
]

# 320k 下每 40k 存一次（共 8 个 checkpoint 全部保留）
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=40000))

work_dir = './work_dirs/eden/fcn_diffae_s2'
