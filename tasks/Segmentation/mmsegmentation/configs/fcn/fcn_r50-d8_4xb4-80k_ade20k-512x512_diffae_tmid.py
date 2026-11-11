# 对照臂：固定中 t 窗口采样（低/中/高三段对照的中段）
# 与 fcn_diffae 的唯一差异：t 只在 [334, 666] 内均匀采样（spaced 网格上约 33 点）。
# 窗口与 logSNR 的对应（linear schedule, T=1000）：
#   t ∈ [334, 666]  <=>  logSNR ∈ [-0.8, -4.5]（中噪区：结构/布局）
# 用法：python tools/train.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae_tmid.py
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

model = dict(
    custom_timestep_sampler=[334, 666],
)

work_dir = './work_dirs/eden/fcn_diffae_tmid'
