# 对照臂：启用 t 自适应（Dt 探针 + EXP3/Hedge 权重，i-DAE 路线）
# 与 fcn_diffae 的唯一差异：t_weight_enabled=True。
# 机制（实现见 eden/t_adaptive.py，全部为教科书算法）：
#   - DtProbe：每 500 步对 16 个 logSNR 桶的代表 t 做配对测量
#     （同图同噪声，cond vs zero-cond 的 D_x0 中位数），收益 EMA(0.95)；
#   - TWeightScheduler：Hedge 指数加权把收益表转成 t 采样分布，
#     eps=0.1 均匀探索保底，z_cap=1 防单桶劫持，pos_only 只奖励正贡献桶；
#   - warmup 5000 步内权重保持均匀（与基线臂前 5000 步严格一致）。
# 产出：t_weight / d_table 随 ckpt 保存，训完直接读"现有方法学到的最优 t 分布"。
# 用法：python tools/train.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae_tweight.py
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

model = dict(
    t_weight_enabled=True,
    # 以下均为 adapter 默认值，显式写出便于调参（改这里即可，不用动代码）
    t_weight_apply='sample',        # 权重作为 t 的采样分布（而非 loss 加权）
    t_weight_pos_only=True,         # 只对 D>0 的桶竞争，负贡献桶压到探索地板
    t_weight_joint=False,           # False = D_x0 计分；True = cos 计分（增益不变）
    t_weight_warmup_iters=5000,
    t_weight_eta=2.0,
    t_weight_eps=0.1,
    t_weight_bins=16,
    t_weight_interval=500,
    t_weight_ema=0.95,
    t_weight_z_cap=1.0,
)

work_dir = './work_dirs/eden/fcn_diffae_tweight'
