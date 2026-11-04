# probe_diffae.py
import torch
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmseg.registry import MODELS

CFG  = '你的配置文件.py'
CKPT = 'work_dirs/ablation_experiment/diffae_36k/iter_36000.pth'

cfg = Config.fromfile(CFG)
cfg.model.backbone.init_cfg = None      # 跳过 ImageNet 预训练下载
cfg.val_dataloader.batch_size = 8
init_default_scope('mmseg')

import mmseg.models.segmentors.fcn_diffae              # ← 注册 FCNWithDiffAEPlusPCGrad 的模块，按你的实际路径改
model = MODELS.build(cfg.model)
load_checkpoint(model, CKPT, map_location='cpu', strict=False)
model.cuda().eval()

data   = next(iter(Runner.build_dataloader(cfg.val_dataloader)))
inputs = data['inputs'].cuda()          # [B,3,H,W]，已归一化（和训练时 loss() 拿到的一样）

@torch.no_grad()
def run_probes(model, inputs):
    model.ema.apply_shadow()            # 采样相关一律用 EMA 权重（和 vis 一致）
    try:
        # ---- 1) zsem_std（在 val 图像上）----
        cond = model.diffae_enc_compress(model.encode_diffae(inputs))
        print(f'zsem_std = {cond.std(dim=0).mean().item():.4f}')

        # ---- 2) D(t) = L(cond=0) − L(z_sem)，逐 t 配对（决定性判据）----
        x128 = model._to_diffae_range(inputs)      # [-1,1] @128，和训练输入同分布
        zero = torch.zeros_like(cond)
        grid, D = model.sampler.use_timesteps, []
        for t in grid:                              # ddim100 的 101 个点
            tt    = torch.full((len(x128),), t, device=x128.device, dtype=torch.long)
            noise = torch.randn_like(x128)
            x_t   = model.sampler.q_sample(x128, tt, noise)
            l_c   = (model.diffae_unet(x_t, tt, cond) - noise).pow(2).mean().item()
            l_z   = (model.diffae_unet(x_t, tt, zero) - noise).pow(2).mean().item()
            D.append(l_z - l_c)
            print(f'  t={t:4d}  D={D[-1]:+.4f}')
        plt.plot(grid, D, marker='o'); plt.xlabel('t'); plt.ylabel('D(t) = L0 - Lz'); plt.grid(); plt.show()
    finally:
        model.ema.restore()

run_probes(model, inputs)