# EDEN 消融臂：DiffAE + 贡献度驱动的 t 自适应采样（定稿方案，论文 3.2）
# 机制（实现见 eden/t_adaptive.py）：
#   - UtProbe：每 500 步对 16 个 logSNR 等频桶的代表 t 做配对测量（同图同噪声，
#     保留/去除 z_sem 各前向一次），由批统计合成 μ̂ = (D̂ − k·σ̂_D)/(L̂ + δ)，
#     再对 μ̂ 做 EMA(0.95)；
#   - TWeightScheduler：p_j = (1−1/K)·max(μ̄_j,0)/Σmax(μ̄,0) + 1/K²（严格归一，
#     均匀保底总占 1/K），作为 t 的采样分布；warmup 5000 步内保持均匀
#     （与基线臂前 5000 步严格一致）。
# 产出：t_weight / mu_table 随 ckpt 保存，训完直接读"学到的最优 t 分布"。
# 用法：python tools/train.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae_tweight.py
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

model = dict(
    t_weight_enabled=True,
    # 以下均为 adapter 默认值，显式写出便于调参（改这里即可，不用动代码）
    t_weight_warmup_iters=5000,
    t_weight_bins=16,
    t_weight_interval=500,
    t_weight_ema=0.95,
    t_weight_probe_batch=16,
    t_weight_k=0.5,            # k：不确定性惩罚系数（论文 k 消融最优值）
    t_weight_delta=1e-3,
)

work_dir = './work_dirs/eden/fcn_diffae_tweight'
