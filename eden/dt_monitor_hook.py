"""
DTMonitorHook：每次验证结束后，用当前模型做全 t 扫描，存五个指标 × 两个横轴。

指标（全部逐样本计算、batch 内取中位数）：
  deps   : D_eps   = mean((p_z-ε)²) - mean((p_c-ε)²)          loss 差，ε 空间
  dx0    : D_x0    = mean((x̂0_z - x̂0_c)²)，x̂0 由固定公式从输出换算
  prod   : D_eps × D_x0（逐样本乘积再取中位数）
  cos    : 1 - cos(p_c, p_z)                                   方向比值，增益不变
  energy : ‖p_z - p_c‖² / ‖p_c‖²                              能量比值，增益不变

每种指标单独一张图（左面板横轴 t，右面板横轴 logSNR），共 5 张 PNG；
所有数组存进同一个 npz。
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class DTMonitorHook(Hook):
    def __init__(self, num_images: int = 16, priority: str = 'LOW'):
        self.num_images = num_images
        self._priority = priority

    @torch.no_grad()
    def after_val_epoch(self, runner, metrics=None) -> None:
        model = runner.model
        if not hasattr(model, 'diffae_unet') or not hasattr(model, 'sampler'):
            return
        device = next(model.parameters()).device

        # 1) 逐张收集（val batch_size=1 且图片尺寸不一，各自算好再拼接）
        mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
        conds, xds = [], []
        cnt = 0
        was_training = model.training
        model.eval()
        for batch in runner.val_dataloader:
            data = model.data_preprocessor(batch, training=False)
            xi = data['inputs'].to(device)
            with torch.no_grad():
                feats = model.backbone(xi)
                z = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)
                conds.append(model.diffae_enc_compress(z))
            xd = (xi * std + mean) / 255.0 * 2.0 - 1.0
            xd = F.interpolate(xd, size=(model.image_size, model.image_size),
                               mode='bilinear', align_corners=False)
            xds.append(xd)
            cnt += xi.shape[0]
            if cnt >= self.num_images:
                break
        if cnt == 0:
            if was_training:
                model.train()
            return
        cond = torch.cat(conds, 0)[:self.num_images]
        x_diffae = torch.cat(xds, 0)[:self.num_images]
        B = x_diffae.shape[0]

        # 2) 全 t 扫描：五种指标逐样本计算
        unet = model.diffae_unet
        L = {k: [] for k in ('deps', 'dx0', 'prod', 'cos', 'energy')}
        for t_val in model.sampler.use_timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            noise = torch.randn_like(x_diffae)
            x_t = model.sampler.q_sample(x_diffae, t, noise)
            p_c = unet(x_t, t, cond)
            p_z = unet(x_t, t, torch.zeros_like(cond))

            ac_b = float(model.sampler.alphas_cumprod[t_val])
            sq_a, sq_1a = ac_b ** 0.5, (1.0 - ac_b) ** 0.5

            if getattr(model.sampler, 'predict_x0', False):
                # x0 模式：输出即 x̂0；D_eps 由换算到 ε 空间得
                s_x0 = (p_z - p_c).pow(2).flatten(1).mean(1)
                eps_c = (x_t - sq_a * p_c) / sq_1a
                eps_z = (x_t - sq_a * p_z) / sq_1a
                s_eps = (eps_z - noise).pow(2).flatten(1).mean(1) - (eps_c - noise).pow(2).flatten(1).mean(1)
            else:
                # ε 模式：输出即 ε̂；D_x0 由换算到 x̂0 空间得
                s_eps = (p_z - noise).pow(2).flatten(1).mean(1) - (p_c - noise).pow(2).flatten(1).mean(1)
                x0_c = (x_t - sq_1a * p_c) / sq_a
                x0_z = (x_t - sq_1a * p_z) / sq_a
                s_x0 = (x0_z - x0_c).pow(2).flatten(1).mean(1)

            # 参数化无关的两个比值
            a = p_c.flatten(1)
            b = p_z.flatten(1)
            s_cos = 1.0 - (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)
            d = b - a
            s_energy = d.pow(2).sum(1) / a.pow(2).sum(1).clamp_min(1e-12)

            L['deps'].append(s_eps.median().item())
            L['dx0'].append(s_x0.median().item())
            L['prod'].append((s_eps * s_x0).median().item())
            L['cos'].append(s_cos.median().item())
            L['energy'].append(s_energy.median().item())
        if was_training:
            model.train()

        # 3) 存数据 + 每种指标单独出图（各 2 面板：t / logSNR）
        ts = np.array(model.sampler.use_timesteps)
        ab = model.sampler.alphas_cumprod.detach().cpu().numpy()
        logsnr = np.log(ab / (1 - ab))[ts]

        out_dir = os.path.join(runner.work_dir, 'dt_curves')
        os.makedirs(out_dir, exist_ok=True)
        it = runner.iter

        np.savez(os.path.join(out_dir, f'iter_{it:06d}.npz'),
                 t=ts, logsnr=logsnr, **{k: np.array(v) for k, v in L.items()})

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for k, v in L.items():
            arr = np.array(v)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(ts, arr, lw=2)
            axes[0].set_xlabel('t')
            axes[1].plot(logsnr, arr, lw=2)
            axes[1].set_xlabel('log SNR')
            for ax in axes:
                ax.grid(alpha=0.3)
                ax.set_ylabel(k)
            fig.suptitle(f'{k} @ iter {it} | mean|.| = {np.abs(arr).mean():.2e}')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'iter_{it:06d}_{k}.png'), dpi=150)
            plt.close(fig)
        runner.logger.info(f'[DTMonitor] 5 metric curves saved to {out_dir} (iter {it})')