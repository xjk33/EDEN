# 【备用/搁置】320k 长训方案的 FCN 基线对照：与 fcn_r50-d8_4xb4-320k_ade20k-512x512_diffae.py
# 完全同调度（恒定 lr 2e-4、单卡 bs16、320k iters），不带 DiffAE 分支。
# 没有它，该 lr 体制下无法判定 DiffAE 的正/负收益。
# 用法：python tools/train.py configs/fcn/fcn_r50-d8_4xb4-320k_ade20k-512x512_baseline.py
_base_ = ['./fcn_r50-d8_4xb4-80k_ade20k-512x512.py']

randomness = dict(seed=0, deterministic=False)

train_dataloader = dict(batch_size=16, num_workers=8)

max_iters = 320000
val_interval = 16000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)

# 与 diffae_s2 臂的 FCN(encoder) 完全一致：恒定 2e-4（= unet_lr × lr_scale）
optimizer = dict(type='SGD', lr=2e-4, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
param_scheduler = [
    dict(type='ConstantLR', factor=1.0, begin=0, end=max_iters, by_epoch=False)
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=40000))

work_dir = './work_dirs/eden/fcn_baseline_s2'
