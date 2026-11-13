# EDEN 消融臂 1/4：纯 DiffAE（t 自适应关、结构化噪声关）
# 80k 联合训练、去噪器随机初始化。
# v2 变更（i-DAE 路线）：
#   1) 去噪器由 UNet（~38M，过强、易绕过条件）换为 ViT-Ti 级条件 DiT（~6.7M，
#      patch 16 吃像素 + adaLN-Zero 双条件注入 + 条件 MLP 隐层 1/4）——
#      表征已解耦到任务 encoder，去噪器只是仪器，故不按 i-DAE 的 ViT-L 规模
#      对齐，只做小，把约束压力转移到 z_sem；
#   2) 预测目标由 ε 换为 x0（predict clean data，与结构化掩码的单次 loss 全包兼容）；
#   3) 扩散分支改为独立 AdamW 优化器（lr=1e-4, wd=0）+ 独立调度
#      （LinearLR warmup + 半周期余弦），与 task 分支的 SGD + PolyLR 完全分离；
#      不做 i-DAE 的 batch/256 折算——该规则只对大 batch 有校准，bs16 下外推
#      得到的 lr 会让仪器长期不合格、污染 D_t 信号。
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
    # z_sem = decode head 分类器输入特征（_forward_feature 输出，channels=512）
    # 的全局池化；enc_output_dim == cond_dim，diffae_enc_compress 退化为 Identity
    enc_output_dim=512,
    cond_dim=512,
    # ---- DiT 去噪器（弱仪器；属性名仍为 diffae_unet）----
    # 表征已解耦到任务 encoder（z_sem），去噪器只是施加约束的仪器，只做小：
    #   'ti' = ViT-Ti/16 ≈ 6.7M（默认，比原 UNet 38.2M 弱约 5.7×）
    #   's'  = ViT-S/16  ≈ 25.8M（对照档）    'b' = ViT-B/16 ≈ 100M（对照上限）
    # 换尺寸后对照 D_t 曲线：整体抬升 = 压力成功转移到 z_sem；塌平/噪声化 = 仪器太弱。
    
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
#     + 余弦退火到 0（沿用 i-DAE 的调度形状与 wd=0）。
#   task 分支：不复述、直接用 {{_base_.optimizer}} / {{_base_.param_scheduler}}
#     引用继承 baseline 的 SGD + PolyLR——baseline 怎么改，这里自动跟随。
# 不做 i-DAE 的 batch/256 线性缩放：该规则只对大 batch（512~16384）有校准，
#   bs16 外推出的 6e-6 会让 zero-init 仪器长期不合格，污染 z_sem 梯度与 D_t
#   信号。峰值固定 1e-4、跨任务共享以保证消融可比；要调就扫 lr ∈ [3e-5, 3e-4]。
# 注意：{{_base_.}} 引用必须裸写、不加引号（mmengine 在文本层替换时自带
#   引号；写成 '{{_base_.x}}' 会让占位符带上字面引号、回填静默失败）。
#   diffae 侧调度的 end=80000 仍与 schedule_80k 绑定，若换训练总长需同步改这里。
optim_wrapper = dict(
    # _delete_：整体替换 base 的 OptimWrapper，防止其 type/optimizer 键
    # 合并进来破坏构造器的 2-key 断言
    _delete_=True,
    constructor='SplitOptimWrapperConstructor',
    task=dict(
        type='OptimWrapper',
        optimizer={{_base_.optimizer}},
        clip_grad=None),
    diffae=dict(
        type='OptimWrapper',
        optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0),
        clip_grad=None),
)

param_scheduler = dict(
    _delete_=True,   # base 是 list、这里是 dict，合并前需显式丢弃 base 值
    task={{_base_.param_scheduler}},
    diffae=[
        dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=2000),
        dict(type='CosineAnnealingLR', T_max=78000, eta_min=0.0,
             begin=2000, end=80000, by_epoch=False),
    ],
)

work_dir = './work_dirs/eden/fcn_diffae_vit'
