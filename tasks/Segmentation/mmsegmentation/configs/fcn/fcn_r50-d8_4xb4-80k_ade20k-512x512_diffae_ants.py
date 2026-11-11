# 对照臂：ANTS 复现（Kim et al., CVPR 2025, arXiv:2411.09998）
# "现有方法学出来的最优 t"参照系：可学习 Beta(a,b) 采样器 + REINFORCE，
# reward = 去噪器更新前后扩散 loss 在代表时间步 [200,500,800] 上的下降量。
# 与 fcn_diffae 的唯一差异：ants_enabled=True（t 采样来源由均匀换成学习分布）。
#
# 与原文的偏差（实现见 eden/beta_sampler.py 头注释，均有意为之）：
#   1. 代表时间步集 S 固定 3 点（原文用队列特征选择）；
#   2. 训练目标保持 x0-MSE（原文用 c_t 加权 L_VLB）——保证"采样分布"是唯一变量；
#   3. φ 优化器状态 / reward 队列不进 checkpoint（resume 后重新热启动）。
# 注意：原文 reward 优化的是生成目标（扩散 loss 下降），不是下游指标——
# 论文中线性 schedule 下学到的分布偏向低 t。若本臂弱于高 t 窗口臂，
# 即"生成最优 t ≠ 表征最优 t"的直接证据。
# 监控：ants_a / ants_b / ants_reward / ants_ent 随训练日志记录，
# 训完用 (a,b) 轨迹即可还原学到的 Beta 采样分布。
# 用法：python tools/train.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae_ants.py
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

model = dict(
    ants_enabled=True,
    ants_lr=1e-3,                       # 原文采样器 lr
    ants_entropy_coef=1e-2,             # 原文熵正则
    ants_interval=40,                   # 原文 f_s=40
    ants_queue=20,                      # 原文 |Q|=20
    ants_rep_timesteps=[200, 500, 800], # 简化：固定代表点（均在 ddim100 网格上）
)

work_dir = './work_dirs/eden/fcn_diffae_ants'
