# EDEN 消融臂：Mask R-CNN R50-FPN 1x + DiffAE 去噪约束（t 自适应关、结构化掩码关）
# 任务侧（数据/增广/优化器/调度/epoch）全部继承 baseline 1x 配置，不做任何修改。
# 扩散侧与分割臂同一套配置：z_sem = FPN 最高层（P5，256ch）的全局池化，
# DiT-Ti 弱仪器 + x0 预测（I-DAE 形式），扩散分支独立 AdamW(1e-4) + 余弦。
# 用法：python tools/train.py configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_diffae.py
_base_ = ['./mask-rcnn_r50_fpn_1x_coco.py']

# 训练期零监控：分析全部离线。tw_*/zsem_std 等标量挂在 losses 里随日志记录。
randomness = dict(seed=0, deterministic=False)

custom_imports = dict(
    imports=['mmdet.models.detectors.eden_adapter_maskrcnn'],
    allow_failed_imports=False)

model = dict(
    type='MaskRCNNWithDiffAE',
    # ---- DiffAE 配置 ----
    image_size=128,
    # z_sem = FPN 最高层（P5，256ch）的全局平均池化——RPN 与 RoI 头共享的
    # 颈部特征（与 cls/seg 同一原则：取任务头真正使用的特征）；
    # enc_output_dim == cond_dim，diffae_enc_compress 退化为 Identity
    enc_output_dim=256,
    cond_dim=256,
    # ---- DiT 去噪器（弱仪器；属性名仍为 diffae_unet）----
    # 'ti' = ViT-Ti/16 ≈ 6.7M（默认，比原 UNet 38.2M 弱约 5.7×）
    denoiser_type='dit',
    dit_size='ti',
    dit_patch=16,
    dit_mlp_ratio=4.0,
    adaln_hidden_mult=0.25,

    num_diffusion_timesteps=1000,
    beta_schedule='linear',
    spaced_timesteps='ddim100',
    ema_decay=0.9999,
    lambda_diff=1.0,
    auto_ema=True,
    # ---- x0 预测、掩码关、t 自适应关 ----
    diffae_pred_x0=True,
    diffae_mask_enabled=False,
    t_weight_enabled=False,
)

# 扩散分支独立优化器 + 独立调度（SplitOptimWrapperConstructor 按参数名前缀切分）：
#   diffae 分支：AdamW(lr=1e-4, betas=(0.9,0.999), wd=0)，LinearLR warmup 2000 步
#     + 余弦退火到 0（与 cls/seg 臂同一形状）。
#   task 分支：用 {{_base_.}} 引用继承 baseline 的 SGD 与 1x 调度——
#     baseline 怎么改，这里自动跟随。
optim_wrapper = dict(
    _delete_=True,
    constructor='SplitOptimWrapperConstructor',
    task=dict(
        type='OptimWrapper',
        optimizer={{_base_.optim_wrapper.optimizer}},
        clip_grad=None),
    diffae=dict(
        type='OptimWrapper',
        optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0),
        clip_grad=None),
)

# 1x = 12 epochs，bs=2/卡。diffae 侧调度按【4 卡】计：118287/(2×4)≈14786 it/ep，
# 共 177432 it；卡数/batch 变化需同步改这里（与 seg 臂 80k 同一约定）。
param_scheduler = dict(
    _delete_=True,   # base 是 list、这里是 dict，合并前需显式丢弃 base 值
    task={{_base_.param_scheduler}},
    diffae=[
        dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=2000),
        dict(type='CosineAnnealingLR', T_max=175432, eta_min=0.0,
             begin=2000, end=177432, by_epoch=False),
    ],
)

work_dir = './work_dirs/eden/maskrcnn_diffae_vit'
