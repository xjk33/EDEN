# EDEN 消融臂：DiffAE + 任务引导结构化掩码（定稿方案，论文 3.3）
# 与 fcn_diffae 的唯一差异：掩码配置
#（DiT 去噪器、x0 预测、i-DAE 式 lr 控制均从 fcn_diffae 继承；
#  x0 参数化使可见区去噪 + 蒙区靠 z_sem 重构合并为单次 loss）。
# 机制（实现见 eden/masking.py）：
#   - G = |∂L_task/∂x₀|，划块取均值、逐图 min-max 归一化得 h_i；
#   - p_i = σ((h_i − μ_h)/s_h)，各块独立 Bernoulli；
#   - 替换型退化：x̃₀ = m⊙c + (1−m)⊙x₀（c = 数据集均值色），loss 目标不变；
#   - 预热期内退化为均匀随机采样（warmup_ratio，不启用 G 引导）。
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

model = dict(
    # ---- 结构化掩码 ----
    diffae_mask_enabled=True,
    diffae_mask_patch=16,
    diffae_mask_mu_h=0.5,
    diffae_mask_s_h=0.1,
    diffae_mask_warmup_ratio=0.1,
)

work_dir = './work_dirs/eden/fcn_diffae_mask'
