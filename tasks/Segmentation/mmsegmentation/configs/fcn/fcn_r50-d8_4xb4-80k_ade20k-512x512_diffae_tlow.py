# 对照臂：固定低 t 窗口采样（纯均匀采样的分桶对照，检验"高 t 适配"假说）
# 与 fcn_diffae 的唯一差异：t 只在 [0, 333] 内均匀采样（spaced 网格 ddim100 上筛出 34 点）。
# 窗口与 logSNR 的对应（linear schedule, T=1000）：
#   t ∈ [0, 333]  <=>  logSNR ∈ [+9.2, -0.8]（轻噪区：局部纹理/颜色去噪）
# 若"收益来自高 t 语义学习"成立，本臂应显著弱于全 t 均匀基线。
# 用法：python tools/train.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae_tlow.py
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

model = dict(
    custom_timestep_sampler=[0, 333],   # 原始 t 坐标；接口已内建，无需改代码
)

work_dir = './work_dirs/eden/fcn_diffae_tlow'
