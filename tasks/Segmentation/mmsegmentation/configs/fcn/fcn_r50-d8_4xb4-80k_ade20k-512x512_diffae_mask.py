# EDEN 消融臂：DiffAE + 任务引导结构化掩码（论文 3.3）
# 与 fcn_diffae 的唯一差异：掩码配置
#（DiT 去噪器、x0 预测、i-DAE 式 lr 控制均从 fcn_diffae 继承；
#  x0 参数化使可见区去噪 + 蒙区靠 z_sem 重构合并为单次 loss）。
# 机制（实现见 eden/masking.py）：
#   - G = |∂L_task/∂x₀|，划块取均值、逐图 softmax 得逐块概率 p；
#   - 每图按 p 无放回采固定 round(ratio·n) 个块（Gumbel-top-k）；
#   - 软替换：蒙区向均值色按 alpha 混合，loss 目标仍是完整 x₀；
#   - 预热期内退化为均匀随机采样（同样块数、位置随机，不启用 G 引导）。
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

model = dict(
    # ---- 结构化掩码 ----
    diffae_mask_enabled=True,
    diffae_mask_patch=16,
    diffae_mask_ratio=0.25,     # 掩码量：每图蒙 25% 的块（64 块 × 0.25 = 16 块）
    diffae_mask_alpha=1.0,      # 软替换程度：1 = 完全替换为均值色
)

work_dir = './work_dirs/eden/fcn_diffae_mask'
