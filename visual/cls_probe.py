"""
分类臂（SwinV2 + EDEN, ImageNet）探针接入层
============================================

probe_core.py 的测量/绘图函数是任务无关的，只有三个任务接口需要适配：
建模、取数、task loss。task loss 已在 probe_core.make_task_loss_fn 内按
wrapper 类型自动分发（_is_cls_wrapper）；本模块提供分类版的【建模 + 取数】，
返回的 batch dict 与 seg 版 build_val_batch 同构：

    cond     [n, cond_dim]      压缩后语义条件（无梯度，供 Dt 谱/重建）
    x_diffae [n, 3, S, S]       [-1,1] 扩散输入
    xis      list of [1,3,H,W]  归一化原图（带梯度测量用）
    dsis     list of 0-dim long tensor  类别标签（task loss 用，替代 seg 的
                                data_samples）

notebook 用法（其余探针同理，measure_* / plot_* 原样调用）：

    import probe_core as pc
    import cls_probe as cp

    model, config = cp.build_cls_model(CFG, CKPT, data_path=IMAGENET_ROOT)
    batch = cp.build_cls_val_batch(model, config, num_images=16)
    res = pc.measure_ut_spectrum(model, batch, n_t=32, ...)
    pc.plot_ut_spectrum(res, ...)

注意：build_cls_model 会把 Swin-Transformer 目录插到 sys.path 最前——该目录
下 config / models / data 等模块名与常见包重名，须保证其优先解析；与 seg 探针
同 kernel 混用时，mmengine Config 不受影响（各自函数内延迟 import）。
"""
import os
import sys
import types

import torch

_SWIN_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'tasks', 'Classification', 'imagenet',
    'Swin-Transformer'))


def _ensure_swin_importable():
    if _SWIN_DIR not in sys.path:
        sys.path.insert(0, _SWIN_DIR)


def build_cls_model(cfg_path, ckpt=None, device=None, data_path=None,
                    batch_size=16):
    """从 yaml 建 SwinV2WithDiffAE（可选加载训练 ckpt，含 EMA shadow——
    DiffAEMixin.load_state_dict 兼容摊平/嵌套两种存储）。

    cfg_path : 训练用 yaml（含 _BASE_ 链，如 ..._eden_tweight_mask_mini.yaml）
    ckpt     : eden_save_checkpoint 存的 ckpt_epoch_*.pth（键 'model'）
    返回 (model, config)——config 供 build_cls_val_batch 复用。
    """
    _ensure_swin_importable()
    from config import get_config
    from models import build_model as build_swin
    from eden_adapter_swinv2 import (SwinV2WithDiffAE,
                                     eden_kwargs_from_config)

    # 与 main_eden.parse_option 同字段；只保留探针需要的覆盖
    args = types.SimpleNamespace(
        cfg=cfg_path, opts=None, batch_size=batch_size,
        data_path=data_path, zip=False, cache_mode='no',
        pretrained=None, resume=None, accumulation_steps=None,
        use_checkpoint=False, disable_amp=True,      # 探针全程 fp32
        amp_opt_level=None, output='output', tag=None, eval=False,
        throughput=False, local_rank=0, fused_window_process=False,
        fused_layernorm=False, optim=None)
    config = get_config(args)

    base = build_swin(config)
    model = SwinV2WithDiffAE(base, **eden_kwargs_from_config(config))
    if ckpt is not None:
        state = torch.load(ckpt, map_location='cpu')
        sd = state.get('model', state)
        msg = model.load_state_dict(sd, strict=False)
        print(f'[build_cls_model] loaded {os.path.basename(ckpt)} '
              f'(epoch {state.get("epoch", "?")}): '
              f'missing={len(msg.missing_keys)} '
              f'unexpected={len(msg.unexpected_keys)}')
        for k in list(msg.missing_keys) + list(msg.unexpected_keys):
            print('   ', k)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model.to(device).eval(), config


def build_cls_val_batch(model, config, num_images=16, batch_size=16,
                        num_workers=4):
    """取 val 前 num_images 张（顺序采样，全部测量共用，保证跨 ckpt 可比）。
    返回与 seg 版 build_val_batch 同构的 dict（dsis = 逐图 label tensor）。
    """
    _ensure_swin_importable()
    from data.build import build_dataset   # data/__init__ 只导出 build_loader

    device = next(model.parameters()).device
    dataset_val, _ = build_dataset(is_train=False, config=config)
    loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)

    conds, xds, xis, dsis = [], [], [], []
    got = 0
    for samples, targets in loader:
        xi = samples.to(device, non_blocking=True)
        with torch.no_grad():
            # 与训练同分布：走模型自己的 encode_diffae（forward_features）
            conds.append(model.diffae_enc_compress(model.encode_diffae(xi)))
            xds.append(model._to_diffae_range(xi))
        xis.extend([xi[i:i + 1] for i in range(xi.shape[0])])
        dsis.extend([targets[i].to(device) for i in range(targets.shape[0])])
        got += xi.shape[0]
        if got >= num_images:
            break
    n = min(num_images, got)
    return {'cond': torch.cat(conds, 0)[:n],
            'x_diffae': torch.cat(xds, 0)[:n],
            'xis': xis[:n], 'dsis': dsis[:n]}
