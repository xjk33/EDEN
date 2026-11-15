"""
ReID 臂（TransReID + EDEN, MSMT17）探针接入层
==============================================

与 cls_probe.py 同一角色：为 visual/ 下的探针提供 ReID 版的【配置 / 取数 /
建模 / task loss】四个任务接口，测量/绘图复用 probe_core。

任务侧特点（与 seg/cls/det 不同的地方都收在本文件里）：
  - 输入 256×128（H×W），PIXEL_MEAN/STD = 0.5（像素已映射到 [-1,1]，
    img01 = x*0.5+0.5）；
  - task loss = ID(CE) + triplet（make_loss 原样构造，与训练逐位一致）——
    triplet 是批内配对挖掘，**必须整批反传**（batch_grad=True，单图无配对）；
  - 裸模型 training 分支才返回 (score, global_feat)；eval 模式下手动复刻该
    分支（eval 下 dropout 恒等、BN 用 running stats，不污染状态）。

用法（须在 transreid 环境下运行；内核工作目录任意，本模块自行处理路径）：

    import probe_core as pc
    import reid_probe as rp

    cfg = rp.build_reid_cfg(YML)
    val_loader, num_classes, cam_num, view_num = rp.build_reid_data(cfg)
    batch = rp.build_reid_val_batch(val_loader, num_images=8)
    model = rp.build_reid_model(cfg, num_classes, cam_num, view_num, CKPT)
    grad_fn = rp.make_reid_grad_fn(model, cfg, num_classes)
    res = pc.measure_mask_viz(model, batch, grad_fn, batch_grad=True)
    pc.plot_mask_viz(res, ...)
"""
import ast
import os
import sys

import torch

_REID_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'tasks', 'Person-ReID', 'TransReID-SSL',
    'transreid_pytorch'))
REID_DIR = _REID_DIR          # notebook 拼 yml / ckpt 路径用


def _ensure_reid_importable():
    """transreid_pytorch 下 config / datasets / model / loss 等模块名与常见包
    重名，须保证其优先解析（与 cls_probe 对 Swin 目录的处理同一理由）。"""
    if _REID_DIR not in sys.path:
        sys.path.insert(0, _REID_DIR)


def _preregister_new_keys(cfg, yml_path):
    """旧版 yacs 兼容（与 train_eden._preregister_new_keys 同实现）：
    合并前把 yml 中 cfg 尚未定义的键（EDEN 节点）预登记；字符串做
    literal_eval 解码，保证类型一致不触发 yacs 类型检查。"""
    import yaml
    from yacs.config import CfgNode as CN

    with open(yml_path, encoding='utf-8') as f:
        yml = yaml.safe_load(f)

    def _decode(v):
        if not isinstance(v, str):
            return v
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v

    def _ensure(node, d):
        for k, v in (d or {}).items():
            if isinstance(v, dict):
                if k not in node:
                    node[k] = CN()
                _ensure(node[k], v)
            elif k not in node:
                node[k] = _decode(v)

    _ensure(cfg, yml)


def build_reid_cfg(yml_path, load_pretrain=False):
    """yml -> 冻结的 yacs cfg（含 EDEN 节点）。
    load_pretrain=False 时把 PRETRAIN_CHOICE 置为 'none'——探针有完整 ckpt，
    不依赖预训练文件（make_model 只认 'imagenet'/'self' 两个分支，其余跳过）。
    DATASETS.ROOT_DIR 相对路径解析到 transreid_pytorch 下（yml 写的是 'data'）。"""
    _ensure_reid_importable()
    from config import cfg

    if hasattr(cfg, 'set_new_allowed'):
        cfg.set_new_allowed(True)
    _preregister_new_keys(cfg, yml_path)
    cfg.merge_from_file(yml_path)
    if not load_pretrain:
        cfg.MODEL.PRETRAIN_CHOICE = 'none'
    root = cfg.DATASETS.ROOT_DIR
    if not os.path.isabs(root):
        cfg.DATASETS.ROOT_DIR = os.path.normpath(os.path.join(_REID_DIR, root))
    cfg.freeze()
    return cfg


def build_reid_data(cfg):
    """make_dataloader -> (val_loader, num_classes, camera_num, view_num)。
    num_classes 是建模型与 make_loss 的必需输入，必须先取数再建模。"""
    _ensure_reid_importable()
    from datasets import make_dataloader

    (_, _, val_loader, num_query, num_classes,
     camera_num, view_num) = make_dataloader(cfg)
    return val_loader, num_classes, camera_num, view_num


def build_reid_val_batch(val_loader, num_images=8, device=None):
    """取 val 第一个 batch 的前 num_images 张（跨 ckpt 固定同批图，严格可比）。

    归一化在 dataset transform 里已完成（mean/std=0.5），无需模型参与。
    Returns: dict ——
        xis   list of [1,3,256,128]  归一化输入（梯度测量用，同尺寸可 cat）
        dsis  list of (label, camid, viewid) 三元组（task loss 用）
        img01 list of [1,3,256,128]  [0,1] 图像（展示用）
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for (img, vid, camid, camids, target_view, _) in val_loader:
        break
    n = min(num_images, img.shape[0])
    X = img[:n].to(device)
    dsis = [(int(vid[i]), int(camids[i]), int(target_view[i]))
            for i in range(n)]
    img01 = [(X[i:i + 1] * 0.5 + 0.5).clamp(0, 1) for i in range(n)]
    return {'xis': [X[i:i + 1] for i in range(n)], 'dsis': dsis,
            'img01': img01}


def build_reid_model(cfg, num_classes, camera_num, view_num, ckpt=None,
                     device=None):
    """裸 TransReID + TransReIDWithDiffAE 封装（可选加载 ckpt）。

    ckpt 是 do_train_eden 保存的裸 state_dict（transformer_eden_*.pth）；
    DiffAEMixin.load_state_dict 兼容摊平/嵌套两种 EMA shadow 存储。
    """
    _ensure_reid_importable()
    from model import make_model
    from eden_adapter_reid import (TransReIDWithDiffAE,
                                   eden_kwargs_from_config)

    base = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                      view_num=view_num)
    model = TransReIDWithDiffAE(base, **eden_kwargs_from_config(cfg))
    if ckpt is not None:
        state = torch.load(ckpt, map_location='cpu')
        sd = state if isinstance(state, dict) and any(
            isinstance(v, torch.Tensor) for v in state.values()) \
            else state.get('model', state)
        # 注意：DiffAEMixin.load_state_dict 无返回值（pop EMA 键后调用
        # super() 但丢弃 IncompatibleKeys），这里手动比对键集做报告。
        model_keys = set(model.state_dict().keys())
        model.load_state_dict(sd, strict=False)   # sd 原地被 pop 掉 EMA 键
        fed = set(sd.keys())                      # 实际喂给模型的键
        missing = sorted(model_keys - fed)
        unexpected = sorted(fed - model_keys)
        print(f'[build_reid_model] loaded {os.path.basename(ckpt)}: '
              f'missing={len(missing)} unexpected={len(unexpected)}')
        for k in missing + unexpected:
            print('   ', k)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model.to(device).eval()


def make_reid_grad_fn(model, cfg, num_classes):
    """ReID task loss callable：make_loss 原样构造（ID CE + triplet，与训练
    逐位一致）；前向在 eval 下复刻裸模型的 training 分支（dropout 在 eval
    恒等，BN 不更新 running stats）。

    返回 grad_fn(X, dsis)：X [N,3,256,128]（requires_grad），dsis =
    (label, camid, viewid) list，返回 scalar（对 X 可微）。triplet 是批内
    配对挖掘，必须整批调用（measure_mask_viz(batch_grad=True)）。
    """
    _ensure_reid_importable()
    from loss import make_loss

    loss_func, _ = make_loss(cfg, num_classes=num_classes)
    core = model.model          # 裸 TransReID（build_transformer）

    def grad_fn(X, dsis):
        device = X.device
        labels = torch.as_tensor([d[0] for d in dsis], device=device)
        cams = torch.as_tensor([d[1] for d in dsis], device=device)
        views = torch.as_tensor([d[2] for d in dsis], device=device)
        # 复刻裸模型 forward 的 training 分支（eval 下 dropout 恒等）
        global_feat = core.base(X, cam_label=cams, view_label=views)
        if core.reduce_feat_dim:
            global_feat = core.fcneck(global_feat)
        feat = core.bottleneck(global_feat)
        score = core.classifier(core.dropout(feat))
        return loss_func(score, global_feat, labels, cams)

    return grad_fn
