"""
离线探针核心：所有分析 ipynb 共用的测量与绘图函数。

设计约定：
  - 训练代码（eden/）不含任何监控逻辑；这里全部离线工作：加载 config + checkpoint，
    取 val 前若干张图，测完出图。
  - 每个 ipynb 只做三件事：build_model / build_val_batch -> 调本文件的 measure_* -> 调 plot_*。
  - 在 mmsegmentation 环境下运行（需要 mmseg/mmengine 建模型与 dataloader）。

包含四组能力：
  1. Dt 谱 + meta-gradient 金标准（Dt 价值判据：两者相关性越高 Dt 越可信）
  2. t_weight / d_table 快照与跨 ckpt 演化（t 自适应学到了什么）
  3. 重建可视化（z_sem 复活判据）
  4. 蒙块选址可视化（结构化噪声蒙的位置与剂量）

注意：旧 checkpoint 的 t 自适应 buffer 键名（顶层 t_weight/d_table/...）在重构后
变为 t_adaptive.probe.*/t_adaptive.scheduler.*，strict=False 加载即可——
Dt 谱、meta-grad、重建不受影响（只依赖网络权重）；旧 ckpt 的 t_weight 快照
会显示为初始均匀值。
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [58.395, 57.12, 57.375]


# =====================================================================
# 建模与取数
# =====================================================================
def build_model(cfg_path, ckpt=None, device=None, cfg_overrides=None):
    """从 config 建模型（可选加载 checkpoint）。strict=False 兼容旧 ckpt 键名。

    cfg_overrides: dict，建模前 merge 进 cfg.model——用于让旧 ckpt 的仪器结构
    与当前代码默认对齐（如 UNet 时代的 run 需 dict(denoiser_type='unet')，
    否则旧 config 里没这个键、会按新默认建出 DiT，键全对不上）。
    """
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmseg.registry import MODELS

    cfg = Config.fromfile(cfg_path)
    if cfg_overrides:
        cfg.merge_from_dict({'model': cfg_overrides})
    init_default_scope('mmseg')
    model = MODELS.build(cfg.model)
    if ckpt is not None:
        from mmengine.runner import load_checkpoint
        load_checkpoint(model, ckpt, map_location='cpu', strict=False)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model.to(device).eval(), cfg


def build_val_batch(model, cfg, num_images=16):
    """
    取 val 前 num_images 张，一次取数、全部测量共用（保证跨 ckpt 可比）。
    Returns:
        dict: cond   —— [n, cond_dim] 压缩后语义条件（无梯度，供 Dt 谱/重建）
              x_diffae —— [n,3,S,S] [-1,1] 扩散输入
              xis    —— list of [1,3,H,W] 归一化原图（带梯度测量用）
              dsis   —— list of data_samples（task loss 用）
    """
    from mmengine.runner import Runner

    device = next(model.parameters()).device
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    val_dataloader = Runner.build_dataloader(cfg.val_dataloader)

    conds, xds, xis, dsis = [], [], [], []
    for batch in val_dataloader:
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
        xis.append(xi)
        dsis.extend(data['data_samples'])
        if len(xds) >= num_images:
            break
    n = min(num_images, len(xds))
    return {'cond': torch.cat(conds, 0)[:n], 'x_diffae': torch.cat(xds, 0)[:n],
            'xis': xis[:n], 'dsis': dsis[:n]}


def make_task_loss_fn(model, xis, dsis):
    """下游任务 loss callable（带梯度）。当前为 seg 实现：decode head 训练 loss。"""
    n = len(xis)

    def task_loss_fn():
        losses = {}
        for i, xi in enumerate(xis):
            feats = model.backbone(xi)
            li = model._decode_head_forward_train(feats, [dsis[i]])
            for k, v in li.items():
                if isinstance(v, torch.Tensor) and 'loss' in k:
                    losses[k] = losses.get(k, 0) + v
        return sum(losses.values()) / n

    return task_loss_fn


def make_cond_fn(model, xis):
    """带梯度的 cond callable：扩散梯度必须能经 z_sem 流回 encoder
    （detach 的 cond 会让 encoder 梯度恒为 0）。"""
    def cond_fn():
        zs = []
        for xi in xis:
            feats = model.backbone(xi)
            zs.append(F.adaptive_avg_pool2d(feats[-1], 1).flatten(1))
        return model.diffae_enc_compress(torch.cat(zs, 0))

    return cond_fn


# =====================================================================
# 统计工具
# =====================================================================
def spearman(x, y):
    """Spearman 秩相关（含并列名次的平均名次处理）。返回 (rho, p)。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return float('nan'), float('nan')

    def rank(a):
        order = a.argsort()
        r = np.empty(n, float)
        r[order] = np.arange(n)
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return sums[inv] / cnt[inv]

    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom < 1e-12:
        return float('nan'), float('nan')
    rho = float((rx * ry).sum() / denom)
    if abs(rho) >= 1.0:
        return rho, 0.0
    from math import erf, sqrt
    t_stat = rho * sqrt((n - 2) / (1 - rho ** 2))
    p = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))   # 正态近似，n>=8 够用
    return rho, p


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return float('nan')
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


# =====================================================================
# 测量 1：Dt 谱
# =====================================================================
@torch.no_grad()
def measure_dt_spectrum(model, batch):
    """
    Dt 谱：对训练网格上每个 t，同图同噪声配对测 cond / zero-cond，
    逐样本算指标取中位数。
    Returns: dict —— t / logsnr / deps / dx0 / prod / cos / energy（[len(grid)] np 数组）
    """
    unet, sampler = model.diffae_unet, model.sampler
    x, cond = batch['x_diffae'], batch['cond']
    device = x.device
    B = x.shape[0]
    L = {k: [] for k in ('deps', 'dx0', 'prod', 'cos', 'energy')}

    for t_val in sampler.use_timesteps:
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_t = sampler.q_sample(x, t, noise)
        p_c = unet(x_t, t, cond)
        p_z = unet(x_t, t, torch.zeros_like(cond))

        ac_b = float(sampler.alphas_cumprod[t_val])
        sq_a, sq_1a = ac_b ** 0.5, (1.0 - ac_b) ** 0.5

        if getattr(sampler, 'predict_x0', False):
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

    ts = np.array(sampler.use_timesteps)
    ab = sampler.alphas_cumprod.detach().cpu().numpy()
    logsnr = np.log(ab / (1.0 - ab))[ts]
    return {'t': ts, 'logsnr': logsnr, **{k: np.array(v) for k, v in L.items()}}


# =====================================================================
# 测量 2：meta-gradient 谱（Dt 价值的金标准）
# =====================================================================
def measure_meta_grad_spectrum(model, batch):
    """
    一阶 meta gradient（foMAML 形式）：把 t 的采样权重 w(t) 视为外层变量，
        ∂L_task(θ - α Σ_t w_t ∇L_diff(t)) / ∂w_t  ∝  -⟨∇L_task, ∇L_diff(t)⟩
    内积越大（越同向）= 在该 t 上去噪训练对下游任务的真实增益越大。
    与 Dt 谱的相关性越高，Dt 作为"下游贡献度代理"的价值越高。

    Returns: dict —— rep_t / rep_logsnr / meta / meta_cos（[num_bins] np 数组）
    """
    sampler, unet = model.sampler, model.diffae_unet
    probe = model.t_adaptive.probe
    x = batch['x_diffae']
    device = x.device
    B = x.shape[0]

    params = [p for n_, p in model.named_parameters()
              if p.requires_grad and ('backbone' in n_ or 'diffae_enc_compress' in n_)]

    # g_task：一次反传
    task_loss = make_task_loss_fn(model, batch['xis'], batch['dsis'])()
    g_task = torch.autograd.grad(task_loss, params, allow_unused=True)
    g_task = [torch.zeros_like(p) if g is None else g.detach()
              for p, g in zip(params, g_task)]
    norm_task = torch.sqrt(sum((g ** 2).sum() for g in g_task)).clamp_min(1e-12)

    # cond 带梯度、只算一次；各桶共享其子图，反传 retain_graph 保住它
    cond = make_cond_fn(model, batch['xis'])()

    meta, meta_cos = [], []
    for bi in range(probe.num_bins):
        t_val = int(probe.bin_rep_t[bi])
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_t = sampler.q_sample(x, t, noise)
        pred = unet(x_t, t, cond)
        target = x if getattr(sampler, 'predict_x0', False) else noise
        l_diff = F.mse_loss(pred, target)
        g_diff = torch.autograd.grad(l_diff, params, allow_unused=True,
                                     retain_graph=True)
        g_diff = [torch.zeros_like(p) if g is None else g.detach()
                  for p, g in zip(params, g_diff)]

        ip = sum((gt * gd).sum() for gt, gd in zip(g_task, g_diff))
        norm_diff = torch.sqrt(sum((g ** 2).sum() for g in g_diff)).clamp_min(1e-12)
        meta.append(ip.item())
        meta_cos.append((ip / (norm_task * norm_diff)).item())

    rep_t = probe.bin_rep_t.detach().cpu().numpy().astype(int)
    ab = sampler.alphas_cumprod.detach().cpu().numpy()
    rep_logsnr = np.log(ab / (1.0 - ab))[rep_t]
    return {'rep_t': rep_t, 'rep_logsnr': rep_logsnr,
            'meta': np.array(meta), 'meta_cos': np.array(meta_cos)}


def dt_meta_correlation(dt, mg):
    """Dt（桶代表 t 处取值）与 meta-grad 的 Spearman / Pearson。"""
    bin_deps = np.array([dt['deps'][int(np.argmin(np.abs(dt['t'] - t)))]
                         for t in mg['rep_t']])
    rho_s, p_s = spearman(bin_deps, mg['meta'])
    rho_p = pearson(bin_deps, mg['meta'])
    return rho_s, p_s, rho_p


# =====================================================================
# 测量 3：t_weight 快照（从 ckpt buffer 读，无需任何前向）
# =====================================================================
def read_tweight_snapshot(model):
    """从当前模型读 t 自适应状态。Returns: dict —— t_weight / d_table / rep_t / logsnr / entropy"""
    w = model.t_weight.detach().float().cpu().numpy()
    d = model.d_table.detach().float().cpu().numpy()
    rep_t = model.bin_rep_t.detach().cpu().numpy().astype(int)
    ab = model.sampler.alphas_cumprod.detach().float().cpu().numpy()
    logsnr = np.log(ab / (1.0 - ab))[rep_t]
    ent = float(-(w * np.log(np.clip(w, 1e-12, None))).sum() / np.log(len(w)))
    return {'t_weight': w, 'd_table': d, 'rep_t': rep_t, 'logsnr': logsnr,
            'entropy': ent}


# =====================================================================
# 测量 4：蒙块选址（用模型自己的 mask_selector 现采，与训练同一套逻辑）
# =====================================================================
def measure_mask_selection(model, batch, num_images=4):
    """
    seg 输入梯度热力图 + mask_selector 现采蒙块。
    Returns: dict —— img [n,3,S,S] (0~1) / heat [n,1,S,S]（归一化）/ keep [n,1,S,S]（有效保留率）
    """
    xis = batch['xis'][:num_images]
    dsis = batch['dsis'][:num_images]
    device = xis[0].device
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    heats, imgs = [], []
    for xi, dsi in zip(xis, dsis):
        xg = xi.detach().requires_grad_(True)
        with torch.enable_grad():
            feats = model.backbone(xg)
            losses = model._decode_head_forward_train(feats, [dsi])
            task = sum(v for k, v in losses.items()
                       if isinstance(v, torch.Tensor) and 'loss' in k)
            gx = torch.autograd.grad(task, xg)[0]
        h = gx.abs().mean(dim=1, keepdim=True)
        h = F.adaptive_avg_pool2d(h, model.image_size)
        heats.append(h.detach())
        with torch.no_grad():
            img01 = ((xi.detach() * std + mean) / 255.0).clamp(0, 1)
            imgs.append(F.interpolate(img01, size=(model.image_size, model.image_size),
                                      mode='bilinear', align_corners=False))

    heat = torch.cat(heats, 0)
    img = torch.cat(imgs, 0)
    heat_n = heat / heat.flatten(1).max(dim=1).values.view(-1, 1, 1, 1).clamp_min(1e-8)
    with torch.no_grad():
        keep = model.mask_selector.select(heat)          # 0/1 keep-mask
    if model.mask_applier.soft:                          # 软蒙：蒙块处保留 1-alpha
        keep = 1.0 - model.mask_applier.alpha * (1.0 - keep)
    return {'img': img, 'heat': heat_n, 'keep': keep}


# =====================================================================
# 绘图
# =====================================================================
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_dt_vs_meta(dt, mg=None, title='', save_path=None):
    """Dt 谱（蓝，全网格）+ meta-grad 谱（红，桶代表 t）双 y 轴合图，标注相关系数。"""
    plt = _plt()
    rho_s = p_s = rho_p = float('nan')
    if mg is not None:
        rho_s, p_s, rho_p = dt_meta_correlation(dt, mg)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, x_dt, x_mg, xlab in [
            (axes[0], dt['t'], mg['rep_t'] if mg else None, 't'),
            (axes[1], dt['logsnr'], mg['rep_logsnr'] if mg else None, 'log SNR')]:
        ax.plot(x_dt, dt['deps'], lw=2, color='tab:blue', label='D_eps (Dt)')
        ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.set_xlabel(xlab)
        ax.set_ylabel('D_eps', color='tab:blue')
        ax.grid(alpha=0.3)
        if mg is not None:
            ax2 = ax.twinx()
            ax2.plot(x_mg, mg['meta'], lw=2, marker='o', ms=4,
                     color='tab:red', label='meta-grad <g_task, g_diff>')
            ax2.set_ylabel('meta gradient', color='tab:red')
            lines = ax.get_lines() + ax2.get_lines()
            ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
        else:
            ax.legend(fontsize=8)
    suptitle = title
    if mg is not None:
        suptitle += f' | Spearman ρ={rho_s:.3f} (p={p_s:.3f}) | Pearson r={rho_p:.3f}'
    fig.suptitle(suptitle)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, (rho_s, p_s, rho_p)


def plot_tweight(snap, title='', ts_freq=None, save_path=None):
    """t_weight 柱状 + D 表折线（+ 可选实际采样频率），双面板：t / logSNR。"""
    plt = _plt()
    w, d, rep_t, logsnr = snap['t_weight'], snap['d_table'], snap['rep_t'], snap['logsnr']
    K = len(w)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    d_n = d / np.abs(d).max() * w.max() if np.abs(d).max() > 0 else d
    for ax, x, xlab in [(axes[0], rep_t, 't (bin rep)'),
                        (axes[1], logsnr, 'log SNR (bin rep)')]:
        ax.bar(x, w, width=(x[1] - x[0]) * 0.7 if K > 1 else 10,
               alpha=0.7, label='t_weight')
        ax.plot(x, d_n, color='tab:red', marker='o', ms=3, label='D table (scaled)')
        if ts_freq is not None:
            ax.plot(x, ts_freq, color='tab:green', marker='s', ms=3, lw=1.8,
                    label='realized t freq')
        ax.axhline(1.0 / K, color='k', ls='--', lw=0.8, label='uniform')
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('prob / weight')
    fig.suptitle(f'{title} | entropy={snap["entropy"]:.3f} | '
                 f'max={w.max():.3f}@bin{int(w.argmax())}')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_tweight_evolution(snaps, labels, save_path=None):
    """跨 ckpt 的 t_weight 演化：热力图（行=ckpt，列=桶）+ 熵曲线。"""
    plt = _plt()
    W = np.stack([s['t_weight'] for s in snaps])
    ents = [s['entropy'] for s in snaps]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4),
                             gridspec_kw={'width_ratios': [2.2, 1]})
    im = axes[0].imshow(W, aspect='auto', cmap='viridis')
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xticks(range(W.shape[1]))
    axes[0].set_xticklabels([str(t) for t in snaps[0]['rep_t']], rotation=45, fontsize=7)
    axes[0].set_xlabel('bin rep t')
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].plot(range(len(labels)), ents, 'o-', color='tab:orange')
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('entropy')
    axes[1].grid(alpha=0.3)
    fig.suptitle('t_weight evolution')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_mask_selection(sel, title='', save_path=None):
    """蒙块选址三联图：原图 | seg 梯度热力 | 蒙版叠加（蒙区染红）。"""
    plt = _plt()
    img, heat, keep = sel['img'], sel['heat'], sel['keep']
    n = img.shape[0]
    red = torch.tensor([1.0, 0.1, 0.1], device=img.device).view(1, 3, 1, 1)
    overlay = img * keep + (red.expand_as(img) * 0.55 + img * 0.45) * (1.0 - keep)

    fig, axes = plt.subplots(n, 3, figsize=(10, 3.4 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        axes[i, 0].imshow(img[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title('image')
        im = axes[i, 1].imshow(heat[i, 0].cpu().numpy(), cmap='turbo', vmin=0, vmax=1)
        axes[i, 1].set_title('seg heat')
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046)
        axes[i, 2].imshow(overlay[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 2].set_title(f'mask (kept={keep[i].mean().item():.2f})')
        for j in range(3):
            axes[i, j].axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def save_recon(model, batch, save_path, max_samples=4):
    """重建四列对比（原图 | 反演 | 重建 | 仅 z_sem 生成），走 mixin 接口。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_diffae_visualization(
        batch['x_diffae'], save_path,
        max_samples=max_samples, cond=batch['cond'])
    return save_path
