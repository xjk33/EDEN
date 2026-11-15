"""
检测臂（MaskRCNN + EDEN, COCO）探针接入层
==========================================

与 cls_probe.py 同一角色：probe_core.py 的测量/绘图是任务无关的，只有三个
任务接口需要适配——建模、取数、task loss。本模块提供检测版，返回的 batch
dict 与 seg 版同构（外加 'img01'，掩码可视化展示用）：

    xis   list of [1,3,H,W]  归一化输入（带梯度测量用；同一 val batch，
                             已 padding 同尺寸，可 cat 成批）
    dsis  list of DetDataSample（含 gt_instances；task loss 用）
    img01 list of [1,3,H,W]  [0,1] 图像（展示用）

用法（须在 mmdetection 环境下、以 mmdetection 仓库根为工作目录运行）：

    import probe_core as pc
    import det_probe as dp

    model, cfg = dp.build_det_model(CFG, CKPT)
    batch = dp.build_det_val_batch(model, cfg, num_images=8)
    grad_fn = dp.make_det_grad_fn(model)
    res = pc.measure_mask_viz(model, batch, grad_fn, batch_grad=True)
    pc.plot_mask_viz(res, ...)
"""
import copy
import os

import torch
import torch.nn.functional as F

IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [58.395, 57.12, 57.375]

# 旧 run dump config 里的退役机制键（与 probe_core.LEGACY_MODEL_KEYS 同步，
# 本文件自成体系、不依赖 probe_core import 顺序）
LEGACY_MODEL_KEYS = (
    't_weight_apply', 't_weight_pos_only', 't_weight_joint', 't_weight_p_min',
    't_weight_p_max', 't_weight_eta', 't_weight_eps', 't_weight_z_cap',
    't_weight_static_window',
    'diffae_mask_mode', 'diffae_mask_temp', 'diffae_mask_soft',
    'diffae_mask_dynamic', 'diffae_mask_ratio_max',
    'diffae_mask_mu_h', 'diffae_mask_s_h', 'diffae_mask_warmup_ratio',
)


def _register_adapter():
    """注册 MaskRCNNWithDiffAE（config 的 custom_imports 只在 Runner 里执行，
    探针不走 Runner，手动 import；force=True 重复 import 无副作用）。"""
    import mmdet.models.detectors.eden_adapter_maskrcnn  # noqa: F401


def build_det_model(cfg_path, ckpt=None, device=None, cfg_overrides=None):
    """从 config 建 MaskRCNNWithDiffAE（可选加载 checkpoint）。
    strict=False 兼容旧 ckpt 键名；退役机制键自动剔除（同 probe_core.build_model）；
    diffae_ema_shadow 先灌进 EMAManager 再加载（同 seg 路径的理由）。"""
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS

    _register_adapter()
    cfg = Config.fromfile(cfg_path)
    dropped = [k for k in LEGACY_MODEL_KEYS if k in cfg.model]
    for k in dropped:
        cfg.model.pop(k)
    if dropped:
        print(f'[build_det_model] 旧 config 兼容：剔除 {dropped}')
    if cfg_overrides:
        cfg.merge_from_dict({'model': cfg_overrides})
    init_default_scope('mmdet')
    model = MODELS.build(cfg.model)
    if ckpt is not None:
        from mmengine.runner.checkpoint import load_state_dict
        state = torch.load(ckpt, map_location='cpu')
        sd = state.get('state_dict', state)
        if hasattr(model, 'ema') and 'diffae_ema_shadow' in sd:
            model.ema.load_state_dict(sd.pop('diffae_ema_shadow'))
        load_state_dict(model, sd, strict=False)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model.to(device).eval(), cfg


def build_det_val_batch(model, cfg, num_images=8):
    """取 val 前 num_images 张（跨 ckpt 固定同批图，保证严格可比）。

    mmdet val_dataloader 默认 batch_size=1：逐 batch 过 data_preprocessor
    累积（各自 padding，尺寸可能不同——因此 xis 保持 list，
    measure_mask_viz 用 batch_grad=False 逐图反传）。"""
    from mmengine.runner import Runner

    device = next(model.parameters()).device
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    val_dataloader = Runner.build_dataloader(cfg.val_dataloader)
    xis, dsis, img01 = [], [], []
    for raw in val_dataloader:
        data = model.data_preprocessor(raw, training=False)
        inputs = data['inputs'].to(device)
        for i in range(inputs.shape[0]):
            xi = inputs[i:i + 1]
            xis.append(xi)
            dsis.append(data['data_samples'][i])
            # DetDataPreprocessor 已做 bgr->rgb + 归一化，反归一化即 [0,1] RGB
            img01.append(((xi * std + mean) / 255.0).clamp(0, 1))
        if len(xis) >= num_images:
            break
    return {'xis': xis[:num_images], 'dsis': dsis[:num_images],
            'img01': img01[:num_images]}


def make_det_grad_fn(model):
    """检测 task loss callable（复刻 MaskRCNNWithDiffAE.loss 的任务侧：
    backbone+neck 一次前向 -> RPN loss + RoI loss 求和；不含 DiffAE 分支）。

    返回 grad_fn(X, dsis)：X [N,3,H,W]（requires_grad）、dsis = DetDataSample
    list（传单个 DetDataSample 会自动包成 list，兼容逐图调用），返回
    scalar（对 X 可微）。整批一次反传即得逐图 G（loss 为逐图项求和，
    ∂L/∂x_i 天然逐图）。
    """

    def grad_fn(X, dsis):
        if not isinstance(dsis, (list, tuple)):
            dsis = [dsis]
        x = model.extract_feat(X)
        losses = {}
        # RPN（与 TwoStageDetector.loss 逐行一致：labels 置 0 的深拷贝样本）
        if model.with_rpn:
            proposal_cfg = model.train_cfg.get('rpn_proposal',
                                               model.test_cfg.rpn)
            rpn_dsis = copy.deepcopy(dsis)
            for ds in rpn_dsis:
                ds.gt_instances.labels = \
                    torch.zeros_like(ds.gt_instances.labels)
            rpn_losses, rpn_results = model.rpn_head.loss_and_predict(
                x, rpn_dsis, proposal_cfg=proposal_cfg)
            for k in list(rpn_losses.keys()):
                if 'loss' in k and 'rpn' not in k:
                    rpn_losses[f'rpn_{k}'] = rpn_losses.pop(k)
            losses.update(rpn_losses)
        else:
            rpn_results = [ds.proposals for ds in dsis]
        roi_losses = model.roi_head.loss(x, rpn_results, dsis)
        losses.update(roi_losses)
        return sum(v for k, v in losses.items()
                   if isinstance(v, torch.Tensor) and 'loss' in k)

    return grad_fn
