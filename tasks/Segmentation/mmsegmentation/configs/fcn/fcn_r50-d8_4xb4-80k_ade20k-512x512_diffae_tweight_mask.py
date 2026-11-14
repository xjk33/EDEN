# EDEN 终版组合：DiffAE + t 自适应 + 结构化掩码（论文 3.2 + 3.3，ADE20K FCN-R50 80k）
# = fcn_diffae_tweight 与 fcn_diffae_mask 的并集；
# 预热 W 步内 t 均匀采样 + 掩码均匀随机（同样块数、位置随机），
# 预热结束两者同时启用引导（方案 3.4）。
#
# t 自适应（实现见 eden/t_adaptive.py）：
#   - UtProbe：每 I=500 步取当前 batch 前 B_p=16 张图，对 K=16 个 logSNR 等频
#     区间的代表 t 做配对测量（同图同噪声，保留/去除 z_sem 各前向一次），
#     由批统计合成 μ̂ = (D̂ − k·σ̂_D)/(L̂ + δ)，EMA(ρ=0.95) 进 μ̄；
#   - TWeightScheduler：p_j = (1−1/K)·max(μ̄_j,0)/Σmax(μ̄,0) + 1/K²
#     （正部归一化 + 均匀保底防坍缩，严格归一）；
#   - k=0.5：不确定性惩罚系数，论文 k 消融最优值。
# 结构化掩码（实现见 eden/masking.py，两个控制量：数量限制 + 掩码强度）：
#   - 选址：G = |∂L_task/∂x₀| 划块取均值，逐图 softmax 得逐块概率，
#     每图按概率无放回定量采 round(ratio·n) 块（Gumbel-top-k）——数量限制；
#   - 施加：蒙区向该图均值色按 alpha 混合（软替换）——掩码强度，
#     alpha=1 即 x̃₀ = m⊙c + (1−m)⊙x₀ 的完全替换；
#   - loss 保持 I-DAE 的 x̂₀ 预测形式（diffae_pred_x0=True，继承自 base），
#     蒙区内容的唯一信息源是 z_sem；大面积无掩码区域维持标准扩散约束托底。
# 用法：python tools/train.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae_tweight_mask.py
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

model = dict(
    # ---- t 自适应（3.2）----
    t_weight_enabled=True,
    t_weight_warmup_iters=5000,   # W：预热步数（与掩码预热共用，3.4）
    t_weight_bins=16,             # K：logSNR 等频区间数
    t_weight_interval=500,        # I：探针间隔（步）
    t_weight_ema=0.95,            # ρ：贡献度 EMA 系数
    t_weight_probe_batch=16,      # B_p：每轮探针样本数
    t_weight_k=0.5,               # k：不确定性惩罚系数（论文 k 消融最优）
    t_weight_delta=1e-3,          # δ：量纲归一保护常数
    # ---- 结构化掩码（3.3）----
    diffae_pred_x0=True,          # x̂₀ 预测（I-DAE 形式；掩码 loss 单次全包）
    diffae_mask_enabled=True,
    diffae_mask_patch=16,         # 蒙块边长（128 图上 8×8 = 64 块）
    diffae_mask_ratio=0.25,       # 数量限制：每图蒙 25% 的块（64 × 0.25 = 16 块）
    diffae_mask_alpha=1.0,        # 掩码强度：1 = 完全替换为均值色
)

work_dir = './work_dirs/eden/fcn_diffae_tweight_mask'
