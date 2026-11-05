# EDEN 消融臂 3/4：DiffAE + 结构化噪声（seg 梯度热力选址 + 软蒙 + 动态幅度）
# 与 fcn_diffae 的唯一差异：x0 参数化 + 蒙块配置。
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

model = dict(
    # ---- x0 参数化（蒙块主线用 x0：可见区去噪 + 蒙区靠 z_sem 重构，单次 loss）----
    diffae_pred_x0=True,
    # ---- 结构化噪声 ----
    diffae_mask_mode='hot',         # 蒙 seg 高敏感块（梯度热力选址）
    diffae_mask_patch=16,
    diffae_mask_temp=1.0,
    diffae_mask_soft=True,          # 软蒙：加噪前向数据集均值色混合
    diffae_mask_alpha=1.0,          # 混合强度 1.0 = 完全均值色
    diffae_mask_dynamic=True,       # 每图每步 r ~ U[0, ratio_max]
    diffae_mask_ratio_max=0.3,
    diffae_mask_ratio=0.25,         # dynamic 时仅作占位
)

work_dir = './work_dirs/eden/fcn_diffae_mask_s2'
