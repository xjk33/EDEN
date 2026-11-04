randomness = dict(seed=0, deterministic=False)

_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

# ===== 训练规模：4h 版（0.396s/iter x 36000 = 4.0h，≈7 epoch）=====
train_cfg = dict(max_iters=36000, type='IterBasedTrainLoop', val_interval=12000)
# PolyLR 终点必须和 max_iters 同步，否则 LR 没衰减完训练就结束了
param_scheduler = [
    dict(type='PolyLR', eta_min=0.0001, power=0.9, begin=0, end=36000, by_epoch=False),
]

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_optimizer=False,
        max_keep_ckpts=1,
        interval=12000,                 # 4h 版：存 12k/24k/36k
        by_epoch=False
    )
)

custom_hooks = [
    # dict(
    #     type='CheckpointHook',
    #     interval=12000,
    #     save_optimizer=True,
    #     by_epoch=False,
    #     out_dir='./resume_ckpts',
    #     max_keep_ckpts=1,
    # )
]

val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    dict(type='BoundaryIoU', num_classes=150),
]
test_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    dict(type='BoundaryIoU', num_classes=150),
]

work_dir = './work_dirs/ablation_experiment/baseline_nothing_36k'