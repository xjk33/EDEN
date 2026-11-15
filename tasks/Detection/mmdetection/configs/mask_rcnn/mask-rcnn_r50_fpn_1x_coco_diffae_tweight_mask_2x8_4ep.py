# 快速验证版：Mask R-CNN R50-FPN + DiffAE + t 自适应 + 结构化掩码
# 目标规模：2 卡 × bs=8（全局 16），4 epochs，约 7-10 h 跑完。
# 与全量版（..._tweight_mask.py，1x=12ep）的差异仅在训练规模：
#   1. batch_size 2 -> 8/卡（显存 9G -> 约 28-32G，40G 卡安全）；
#   2. max_epochs 12 -> 4，task 侧 MultiStepLR milestones 按比例 [8,11]->[2,3]
#      （衰减后末段 lr=2e-4，与原 1x 末段一致）；
#   3. 全局 bs=16 == mmdet base_batch_size，task lr=0.02 无需缩放；
#   4. diffae 侧余弦按新总步数重算：118287/16 ≈ 7393 it/ep × 4 = 29572 it，
#      warmup 2000 步不变，cosine T_max=27572, end=29572；
#   5. t_weight 预热步数按总步数等比缩：5000 × 29572/177432 ≈ 1000。
# 用法：bash tools/dist_train.sh \
#   configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_diffae_tweight_mask_2x8_4ep.py 2
_base_ = ['./mask-rcnn_r50_fpn_1x_coco_diffae_tweight_mask.py']

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=4, val_interval=1)

train_dataloader = dict(batch_size=8, num_workers=4)

model = dict(
    t_weight_warmup_iters=1000,   # 等比缩短预热（原 5000 / 177432 it）
)

param_scheduler = dict(
    _delete_=True,
    task=[
        dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
        dict(type='MultiStepLR', begin=0, end=4, by_epoch=True,
             milestones=[2, 3], gamma=0.1),
    ],
    diffae=[
        dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=2000),
        dict(type='CosineAnnealingLR', T_max=27572, eta_min=0.0,
             begin=2000, end=29572, by_epoch=False),
    ],
)

work_dir = './work_dirs/eden/maskrcnn_diffae_tweight_mask_2x8_4ep'
