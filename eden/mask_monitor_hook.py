"""
MaskMonitorHook：每次验证后可视化蒙块——seg 热力区域 + 最终分块蒙版叠加。

每行一个样本，三列：[原图 | seg 输入梯度热力图 | 蒙版叠加（蒙区染红）]。
蒙块用模型的 _heatmap_to_mask 现采（含动态幅度），与训练时同一套选址逻辑。
仅当 mask 开启（ratio>0 或 dynamic）时工作。
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class MaskMonitorHook(Hook):
    def __init__(self, num_images: int = 4, priority: str = 'LOW'):
        self.num_images = num_images
        self._priority = priority

    def after_val_epoch(self, runner, metrics=None) -> None:
        model = runner.model
        if (getattr(model, 'diffae_mask_ratio', 0.0) <= 0
                and not getattr(model, 'diffae_mask_dynamic', False)):
            return
        device = next(model.parameters()).device

        # 1) 逐张处理（val 图尺寸不一，各自算完热力/缩完图再收集，形状统一）
        mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
        heats, imgs128 = [], []
        was_training = model.training
        model.eval()
        for batch in runner.val_dataloader:
            data = model.data_preprocessor(batch, training=False)
            xi = data['inputs'].to(device)                            # [1,3,H,W]，H 不一
            dsi = data['data_samples']

            # seg 输入梯度热力图（与训练时 mask 选址同源）
            xg = xi.detach().requires_grad_(True)
            with torch.enable_grad():
                feats = model.backbone(xg)
                losses = model._decode_head_forward_train(feats, dsi)
                task = sum(v for k, v in losses.items()
                           if isinstance(v, torch.Tensor) and "loss" in k)
                gx = torch.autograd.grad(task, xg)[0]
            h = gx.abs().mean(dim=1, keepdim=True)
            h = F.adaptive_avg_pool2d(h, model.image_size)            # [1,1,128,128]
            heats.append(h.detach())

            with torch.no_grad():
                img01 = ((xi.detach() * std + mean) / 255.0).clamp(0, 1)
                im = F.interpolate(img01, size=(model.image_size, model.image_size),
                                   mode='bilinear', align_corners=False)
                imgs128.append(im)
            if len(heats) >= self.num_images:
                break

        heat = torch.cat(heats, 0)                                    # [n,1,128,128]
        n = heat.shape[0]
        img128 = torch.cat(imgs128, 0)
        heat_n = heat / heat.flatten(1).max(dim=1).values.view(-1, 1, 1).clamp_min(1e-8)

        # 3) 用模型自己的选址逻辑现采蒙块（含动态幅度），并转成"有效保留率"
        with torch.no_grad():
            mask = model._heatmap_to_mask(heat)                        # [n,1,128,128] 0/1
        if getattr(model, 'diffae_mask_soft', False):
            w = 1.0 - model.diffae_mask_alpha * (1.0 - mask)           # 软蒙：蒙块处 1-α
        else:
            w = mask

        # 4) 叠加图：蒙区染红
        red = torch.tensor([1.0, 0.1, 0.1], device=device).view(1, 3, 1, 1)
        overlay = img128 * w + (red.expand_as(img128) * 0.55 + img128 * 0.45) * (1.0 - w)

        # 5) 存图
        out_dir = os.path.join(runner.work_dir, 'mask_vis')
        os.makedirs(out_dir, exist_ok=True)
        it = runner.iter

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(n, 3, figsize=(10, 3.4 * n))
        if n == 1:
            axes = axes[None, :]
        for i in range(n):
            axes[i, 0].imshow(img128[i].permute(1, 2, 0).cpu().numpy())
            axes[i, 0].set_title('image')
            im = axes[i, 1].imshow(heat_n[i, 0].cpu().numpy(), cmap='turbo', vmin=0, vmax=1)
            axes[i, 1].set_title('seg heat')
            fig.colorbar(im, ax=axes[i, 1], fraction=0.046)
            axes[i, 2].imshow(overlay[i].permute(1, 2, 0).cpu().numpy())
            axes[i, 2].set_title(f'mask (soft={model.diffae_mask_soft}, '
                                 f'dyn={model.diffae_mask_dynamic}, '
                                 f'kept={w[i].mean().item():.2f})')
            for j in range(3):
                axes[i, j].axis('off')
        fig.suptitle(f'mask selection @ iter {it}')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'iter_{it:06d}_mask.png'), dpi=150)
        plt.close(fig)

        if was_training:
            model.train()
        runner.logger.info(f'[MaskMonitor] mask vis saved: {out_dir}/iter_{it:06d}_mask.png')