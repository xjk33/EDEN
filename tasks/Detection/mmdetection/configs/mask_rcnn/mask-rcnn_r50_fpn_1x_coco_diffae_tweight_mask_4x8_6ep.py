# 4 卡重跑版：Mask R-CNN R50-FPN + DiffAE + t 自适应 + 结构化掩码
# 目标：4 卡 × bs=8（全局 32），6 epochs ≈ 22182 it，约 6h 内跑完
# （与 2x8_4ep 同 per-GPU 负载，吞吐约 2x；若实测 ETA 超 6h 按末尾注释降档）。
# 与 2x8_4ep 的差异：
#   1. 卡数 2 -> 4，per-GPU bs=8 不变（显存已验证 ~28-32G），全局 16 -> 32；
#   2. task SGD lr 线性缩放 0.02 -> 0.04（全局 bs 16->32 的标准做法）；
#      diffae AdamW 保持 1e-4 不动——扩散分支是辅助仪器，稳定优先，不随 bs 缩放；
#   3. max_epochs 4 -> 6：it/ep = 118287/32 ≈ 3697，共 22182 it；
#      MultiStepLR milestones [2,3] -> [4,5]（按 1x [8,11]/12 等比）；
#   4. diffae 余弦按新总步数重算：warmup 2000 不变，T_max=20182, end=22182；
#   5. t_weight 预热等比：5000 × 22182/177432 ≈ 625 -> 取 750
#      （与 4ep 版同占比 ~3.4%）。
# 用法：bash tools/dist_train.sh \
#   configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_diffae_tweight_mask_4x8_6ep.py 4
#
# 【降档规则】开跑 ~10 min 后看日志里的 ETA（mmdet 会打印）：
#   若 ETA <= 6h：不动；
#   若 ETA > 6h：换 5ep 档（杀掉改参重启，前几百步可弃）——
#     max_epochs=5, milestones=[3,4], T_max=16485, end=18485, t_warmup=625
#   （5ep 共 18485 it ≈ 6h × (18485/22182) ≈ 5.0h @ 实测速度）
_base_ = ['./mask-rcnn_r50_fpn_1x_coco_diffae_tweight_mask_2x8_4ep.py']

# 多卡必需：EDENDDP（继承自 2x8_4ep，显式保留防误删）
model_wrapper_cfg = dict(type='EDENDistributedDataParallel')

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=1)

train_dataloader = dict(batch_size=8, num_workers=4)

model = dict(
    t_weight_warmup_iters=750,   # 等比预热（占比与 4ep 版一致 ~3.4%）
    diffae_mask_tau=1.0,         # 掩码采样温度（尺度不变化版：softmax((h/h̄)/τ)）
)

optim_wrapper = dict(
    task=dict(optimizer=dict(lr=0.04)),   # 线性缩放：全局 bs 16 -> 32
)

param_scheduler = dict(
    task=[
        dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
        dict(type='MultiStepLR', begin=0, end=6, by_epoch=True,
             milestones=[4, 5], gamma=0.1),
    ],
    diffae=[
        dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=2000),
        dict(type='CosineAnnealingLR', T_max=20182, eta_min=0.0,
             begin=2000, end=22182, by_epoch=False),
    ],
)

work_dir = './work_dirs/eden/maskrcnn_diffae_tweight_mask_4x8_6ep'
