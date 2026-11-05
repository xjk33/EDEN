# EDEN 消融臂 1/4：纯 DiffAE（t 自适应关、结构化噪声关）
# 【Run A】80k 联合训练、UNet 随机初始化——预训练对照实验（Run B）的对照组，
# 与已跑出的负收益实验完全同配方，结果可直接复用，无需重跑。
# 用法：python tools/train.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512.py']

# 训练期零监控：分析全部离线（visual/probe_*.ipynb）。
# base 调度为 80k（val/ckpt interval=8000，共 10 个 checkpoint 全部保留）。
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

# 扩散分支 lr × 0.01（恒定 1e-4）
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={
        'diffae_unet': dict(lr_mult=0.01),
        'diffae_enc_compress': dict(lr_mult=0.01),
    }))

work_dir = './work_dirs/eden/fcn_diffae'
