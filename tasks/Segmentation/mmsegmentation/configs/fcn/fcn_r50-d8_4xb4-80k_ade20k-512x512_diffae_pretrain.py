# DiffAE 分支预训练（Run B 的第一阶段）：无任务 loss，encoder 只接收纯去噪梯度，
# 与 DiffAE/SODA 的 encoder 监督结构同构。产物供 fcn_..._diffae_pretrain_ft.py 加载。
# 用法：python tools/train.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae_pretrain.py
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512.py']

randomness = dict(seed=0, deterministic=False)

custom_imports = dict(
    imports=['mmseg.models.segmentors.eden_adapter_fcn'],
    allow_failed_imports=False)

model = dict(
    type='FCNWithDiffAE',
    diffae_pretrain=True,       # 预训练模式：跳过 seg 分支，只训扩散 loss
    # ---- DiffAE 配置（与 Run A/B 联合训练完全一致，保证权重可无缝加载）----
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

# SODA 式恒定比例 lr：backbone(encoder) = lr_scale × 扩散分支(decoder)，全程恒定。
# 与联合训练 Run A/B 的 lr 体制不同属正常——预训练→微调本来就允许换调度，
# 且 SODA 的教训（encoder 要比 decoder 学得快）在纯去噪阶段同样成立。
unet_lr = 1e-4     # 扩散分支（decoder）恒定 lr
lr_scale = 2.0     # encoder/decoder 比例超参（SODA 消融最优 2）

optimizer = dict(type='SGD', lr=unet_lr * lr_scale, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None,
    paramwise_cfg=dict(custom_keys={
        'diffae_unet': dict(lr_mult=1.0 / lr_scale),
        'diffae_enc_compress': dict(lr_mult=1.0 / lr_scale),
    }))
param_scheduler = [
    dict(type='ConstantLR', factor=1.0, begin=0, end=40000, by_epoch=False)
]

# 40k iters × bs16 = 640k 样本（~32 epochs）。目的不是收敛，而是让 UNet 反传给
# z_sem 的梯度在联合训练第 0 步就是有效信号。无 seg 头，单 iter 快于联合训练，
# 单卡约 5-6h；赶时间可砍到 20k（~3h），同步修改 ft 配置的 load_from 文件名。
max_iters = 40000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters)

# 预训练不做验证（seg 头未训练，mIoU 无意义）
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None

# 只留最后一个 checkpoint（iter_40000.pth，ft 的 load_from 指向它）
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000,
                    max_keep_ckpts=1))

work_dir = './work_dirs/eden/fcn_diffae_pretrain'
