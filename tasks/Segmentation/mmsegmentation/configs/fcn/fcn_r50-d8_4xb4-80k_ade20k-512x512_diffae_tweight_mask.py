# EDEN 消融臂：DiffAE + t 自适应 + 结构化掩码（终版组合，论文 3.2 + 3.3）
# = fcn_diffae_tweight 与 fcn_diffae_mask 的并集；
# 预热 W 步内 t 均匀采样 + 掩码均匀随机，预热结束两者同时启用（方案 3.4）。
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

model = dict(
    # ---- t 自适应 ----
    t_weight_enabled=True,
    t_weight_warmup_iters=5000,
    t_weight_bins=16,
    t_weight_interval=500,
    t_weight_ema=0.95,
    t_weight_probe_batch=16,
    t_weight_k=1.0,
    t_weight_delta=1e-3,
    # ---- x0 参数化 + 结构化掩码 ----
    diffae_pred_x0=True,
    diffae_mask_enabled=True,
    diffae_mask_patch=16,
    diffae_mask_mu_h=0.5,
    diffae_mask_s_h=0.1,
    diffae_mask_warmup_ratio=0.1,
)

work_dir = './work_dirs/eden/fcn_diffae_tweight_mask'
