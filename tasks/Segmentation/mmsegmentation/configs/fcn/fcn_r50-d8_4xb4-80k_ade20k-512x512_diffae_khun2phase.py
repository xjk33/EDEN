# 对照臂：Khungurn 两阶段复现（arXiv:2504.21368, CVPRW 2025）
# 文献中少有的"训练侧调度 t、且明确服务于表征"的方法：
#   阶段一（前 20k iters，总长 1/4，原文未调优）恒 t=990（ddim100 网格最大点，
#   近似原文 t=1 纯噪声）：去噪器从输入拿不到信息，被迫把结构信息压进 z_sem
#   （≈ vanilla AE 状态）；原文验证此机制下 z 的线性探针精度几乎不掉
#   （72.27% vs 72.71%）。
#   阶段二（后 60k iters）恢复随机采样，密度偏低噪区 t = 990·(1-√u)，u~U(0,1)，
#   对应原文 Sc-4 schedule"更多时间花在低噪区"的效果（补细节）。
#
# 与原文的偏差（有意为之，保证与其他臂单变量可比）：
#   1. 保持 EDEN 的线性-β schedule + x0 预测，不引入原文的 Sc-4 schedule
#      与 v-prediction（原文 Table 1 显示 x-pred 在低噪 schedule 下同样可行）；
#   2. 用 Beta(1,2) 型偏低噪密度代替 Sc-4，只复现"两阶段课程"核心机制。
# 与 fcn_diffae 的唯一差异：khun_enabled=True（t 采样由均匀换成两阶段课程）。
# 监控：khun_phase（1/2）与 ts_mean 随训练日志记录，ts_mean 在 20k 处应从
# ~990 掉到 ~330（E[1-√u]=1/3），可用于确认阶段切换生效。
# 用法：python tools/train.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae_khun2phase.py
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

model = dict(
    khun_enabled=True,
    khun_phase1_iters=20000,    # 80k 总长的 1/4（原文比例）
    khun_phase1_t=990,          # ddim100 网格最大点 ≈ 原文 t=1（纯噪声）
)

work_dir = './work_dirs/eden/fcn_diffae_khun2phase'
