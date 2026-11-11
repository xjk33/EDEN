# 对照臂：DiT 学习率常数化（检验 Dt/meta-grad 谱的突变是否由余弦退火引起）
# 与 fcn_diffae 的唯一差异：diffae 侧调度 = LinearLR warmup 2000 -> 恒定 1e-4。
# warmup 段保留且与基线完全一致 -> 两臂在 0~2000 步严格同步，
# 2000 步之后唯一的分叉就是 lr 曲线；若 meta-grad 翻转的时标跟着 lr 走，
# 本臂应观察到翻转消失或时标漂移。
# 用法：python tools/train.py configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae_constlr.py
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512_diffae.py']

param_scheduler = dict(
    # task 键不写 -> 自动继承基线的 SGD + PolyLR
    diffae=[
        dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=2000),
        dict(type='ConstantLR', factor=1.0, by_epoch=False, begin=2000, end=80000),
    ],
)

work_dir = './work_dirs/eden/fcn_diffae_constlr'
