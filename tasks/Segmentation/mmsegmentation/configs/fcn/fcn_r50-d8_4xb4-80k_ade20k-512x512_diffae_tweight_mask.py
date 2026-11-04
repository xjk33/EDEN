# EDEN 消融臂 4/4：DiffAE + t 自适应 + 结构化噪声（终版组合）
# = fcn_diffae_tweight 与 fcn_diffae_mask 的并集。
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

model = dict(
    # ---- t 自适应 ----
    t_weight_enabled=True,
    t_weight_apply='sample',
    t_weight_pos_only=True,
    t_weight_joint=False,
    t_weight_z_cap=1.0,
    t_weight_warmup_iters=5000,
    t_weight_eta=2.0,
    t_weight_eps=0.1,
    t_weight_bins=16,
    t_weight_interval=500,
    t_weight_ema=0.95,
    # ---- x0 参数化 + 结构化噪声 ----
    diffae_pred_x0=True,
    diffae_mask_mode='hot',
    diffae_mask_patch=16,
    diffae_mask_temp=1.0,
    diffae_mask_soft=True,
    diffae_mask_alpha=1.0,
    diffae_mask_dynamic=True,
    diffae_mask_ratio_max=0.3,
    diffae_mask_ratio=0.25,
)

work_dir = './work_dirs/eden/fcn_diffae_tweight_mask'
