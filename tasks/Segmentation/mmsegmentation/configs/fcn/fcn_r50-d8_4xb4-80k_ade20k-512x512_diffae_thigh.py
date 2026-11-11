# 对照臂：固定高 t 窗口采样（直接检验"对高 t 适配"假说）
# 与 fcn_diffae 的唯一差异：t 只在 [667, 990] 内均匀采样（spaced 网格上约 33 点）。
# 窗口与 logSNR 的对应（linear schedule, T=1000）：
#   t ∈ [667, 990]  <=>  logSNR ∈ [-4.5, -9.9]（重噪区：全局语义；
#   你探针图里 meta-grad 翻转就发生在 logSNR < -5 一带，即 t > 750，正落在本窗内）
# 若假说成立，本臂应最接近（甚至优于）全 t 均匀基线。
# 用法：python tools/train.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae_thigh.py
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

model = dict(
    custom_timestep_sampler=[667, 990],
)

work_dir = './work_dirs/eden/fcn_diffae_thigh'
