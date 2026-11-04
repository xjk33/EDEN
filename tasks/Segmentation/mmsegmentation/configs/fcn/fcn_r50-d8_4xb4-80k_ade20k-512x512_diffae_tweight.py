# EDEN 消融臂 2/4：DiffAE + t 自适应（Dt 探针 -> Hedge/EXP3 -> t 采样分布）
# 与 fcn_diffae 的唯一差异：t_weight_enabled 及配套超参。
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

model = dict(
    # ---- t 自适应：D_eps 正值竞争 + 采样应用 ----
    t_weight_enabled=True,
    t_weight_apply='sample',        # t_weight 作为 t 的采样分布（每步满强度）
    t_weight_pos_only=True,         # 正值竞争（负贡献/域外吃地板）
    t_weight_joint=False,           # 计分用 d_table（D_eps）；True = cos 计分
    t_weight_z_cap=1.0,
    t_weight_warmup_iters=5000,
    t_weight_eta=2.0,
    t_weight_eps=0.1,
    t_weight_bins=16,
    t_weight_interval=500,
    t_weight_ema=0.95,
    # t_weight_domain=[-2.5, 2.2],  # 可选：logSNR 定义域，域外恒压地板
    # t_weight_static_window=None,  # 可选：如 [100,500] 则禁用动态更新
)

work_dir = './work_dirs/eden/fcn_diffae_tweight'
