# EDEN 终版组合：Mask R-CNN R50-FPN 1x + DiffAE + t 自适应 + 结构化掩码
#（论文 3.2 + 3.3；与分割臂 fcn_r50-d8_..._diffae_tweight_mask.py 同一套超参）
# 预热 W 步内 t 均匀采样 + 掩码均匀随机（同样块数、位置随机），
# 预热结束两者同时启用引导（方案 3.4）。
# 用法：python tools/train.py configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_diffae_tweight_mask.py
_base_ = ['./mask-rcnn_r50_fpn_1x_coco_diffae.py']

model = dict(
    # ---- t 自适应（3.2）----
    t_weight_enabled=True,
    t_weight_warmup_iters=5000,   # W：预热步数（与掩码预热共用，3.4）
    t_weight_bins=16,             # K：logSNR 等频区间数
    t_weight_interval=500,        # I：探针间隔（步）
    t_weight_ema=0.95,            # ρ：贡献度 EMA 系数
    t_weight_probe_batch=16,      # B_p：每轮探针样本数
    t_weight_k=0.5,               # k：不确定性惩罚系数（论文 k 消融最优）
    t_weight_delta=1e-3,          # δ：量纲归一保护常数
    # ---- 结构化掩码（3.3）----
    diffae_pred_x0=True,          # x̂₀ 预测（I-DAE 形式；掩码 loss 单次全包）
    diffae_mask_enabled=True,
    diffae_mask_patch=16,         # 蒙块边长（128 图上 8×8 = 64 块）
    diffae_mask_ratio=0.25,       # 数量限制：每图蒙 25% 的块（64 × 0.25 = 16 块）
    diffae_mask_alpha=1.0,        # 掩码强度：1 = 完全替换为均值色
)

work_dir = './work_dirs/eden/maskrcnn_diffae_tweight_mask'
