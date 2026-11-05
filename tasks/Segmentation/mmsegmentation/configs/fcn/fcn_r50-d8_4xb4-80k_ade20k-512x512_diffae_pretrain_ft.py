# 【Run B】预训练 DiffAE 初始化 + 与 Run A 完全相同的 80k 联合训练。
# 与 Run A（fcn_..._diffae.py）的唯一差异：load_from 预训练权重。
# 判定逻辑：B 翻正/显著优于 A → 去噪信号为正、负收益源于 UNet 冷启动噪声；
#           B 仍为负 → 去噪辅助在本任务上的价值存疑（同样是强结论）。
# 前置：先跑 fcn_..._diffae_pretrain.py（40k iters）。
# 用法：python tools/train.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae_pretrain_ft.py
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

# 预训练 checkpoint（含 backbone/enc_compress/UNet 及 EMA shadow；seg 头是
# 预训练时的随机初始化，等价于正常新建，无影响）。预训练 iters 改动时同步改文件名。
load_from = 'work_dirs/eden/fcn_diffae_pretrain/iter_40000.pth'

work_dir = './work_dirs/eden/fcn_diffae_pretrain_ft'
