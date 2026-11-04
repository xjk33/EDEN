"""
DtDeltaUConsistencyHook：在指定训练进度做 ΔU 探针实验，
验证 Dt 谱与真实外层增益 ΔU 谱的一致性（即 "RL meta-gradient" 的探针估计）。

每个检查点（默认 20% / 50% / 80%）：
  1) Dt 侧：num_bins 个 t 区间，每区间重复采样 samples_per_bin 次，
     记录 D_eps = L(xt,t,0) - L(xt,t,z_sem)（沿用原 DTMonitorHook 口径）。
  2) ΔU 侧：先存模型快照，对每个 bin：从快照出发，仅在该 bin 内采样 t，
     用 (L_task + λ·L_denoise) 微调 probe_iters 步，
     ΔU = L_task_eval(前) - L_task_eval(后)（下游 loss 降幅=增益），随后恢复快照。
  3) 输出：npz（两条谱 + Spearman ρ/p）+ PNG（logSNR 横轴、min-max 归一化
     双曲线 + ρ 标注；右面板为未归一化原值对照）。

注意：_task_loss() 需按你训练脚本里下游 loss 的实际计算方式调整（已标注）。
"""
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def _spearman(x, y):
    """自包含的 Spearman 秩相关（含并列名次的平均名次处理）。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return float('nan'), float('nan')

    def rank(a):
        order = a.argsort()
        r = np.empty(n, float)
        r[order] = np.arange(n)
        # 并列取平均名次
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
    # t 近似 p 值（双侧）
    if abs(rho) >= 1.0:
        return rho, 0.0
    from math import erf, sqrt
    t_stat = rho * sqrt((n - 2) / (1 - rho ** 2))
    p = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))  # 正态近似，n>=8 够用
    return rho, p


@HOOKS.register_module()
class DtDeltaUConsistencyHook(Hook):
    def __init__(self,
                 check_fracs=(0.2, 0.5, 0.8),  # 在训练的哪些进度点测试
                 num_bins=10,                  # t 区间分箱数（Spearman 的 n，8~12 为宜）
                 num_images=16,                # Dt 侧测量用图数
                 samples_per_bin=8,            # Dt 侧每 bin 噪声重采样次数（取均值）
                 probe_iters=20,               # ΔU 侧每 bin 探针微调步数
                 probe_lr=1e-5,                # 探针微调学习率（临时优化器）
                 probe_repeats=1,              # 每 bin 探针重复次数（>1 更稳，更贵）
                 eval_batches=4,               # 下游评估 loss 用固定 batch 数
                 denoise_weight=1.0,           # 探针中 L_denoise 权重（与训练一致）
                 out_name='dt_du_consistency',
                 priority='LOW'):
        self.check_fracs = tuple(check_fracs)
        self.num_bins = num_bins
        self.num_images = num_images
        self.samples_per_bin = samples_per_bin
        self.probe_iters = probe_iters
        self.probe_lr = probe_lr
        self.probe_repeats = probe_repeats
        self.eval_batches = eval_batches
        self.denoise_weight = denoise_weight
        self.out_name = out_name
        self._priority = priority
        self._done = set()

    # ------------------------------------------------------------------ #
    # 触发：before_train_iter，到达 check_fracs 对应 iter 时执行一次
    # ------------------------------------------------------------------ #
    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        if runner.max_iters is None:
            return
        it = runner.iter
        for f in self.check_fracs:
            target = int(f * runner.max_iters)
            if f not in self._done and it >= target:
                self._done.add(f)
                self._run_consistency_test(runner, f)

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def _run_consistency_test(self, runner, frac):
        model = runner.model
        if not hasattr(model, 'diffae_unet') or not hasattr(model, 'sampler'):
            return
        device = next(model.parameters()).device
        it = runner.iter

        # ---- 准备：t 分箱 + Dt 用图 + 固定评估 batch ----
        timesteps = np.array(model.sampler.use_timesteps)
        bins = [b for b in np.array_split(timesteps, self.num_bins) if len(b) > 0]
        cond, x_diffae = self._collect_images(model, runner, device)
        if cond is None:
            return
        eval_cache = self._collect_eval_batches(model, runner)

        # ---- Dt 谱 ----
        dt_mean, dt_std = self._measure_dt(model, cond, x_diffae, bins, device)

        # ---- ΔU 谱（探针） ----
        du_mean, du_std = self._measure_du(model, runner, bins, eval_cache, device)

        # ---- Spearman + 归一化 ----
        rho, p = _spearman(dt_mean, du_mean)
        dt_n, du_n = self._minmax(dt_mean), self._minmax(du_mean)

        # ---- 横轴：bin 中心 t 与 logSNR ----
        ab = model.sampler.alphas_cumprod.detach().cpu().numpy()
        t_center = np.array([b.mean() for b in bins])
        ac_c = np.array([ab[int(round(tc))] for tc in t_center])
        logsnr_c = np.log(ac_c / (1 - ac_c))

        # ---- 存盘 ----
        out_dir = os.path.join(runner.work_dir, self.out_name)
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, f'iter_{it:06d}.npz'),
                 frac=frac, iter=it, t_center=t_center, logsnr=logsnr_c,
                 dt_mean=dt_mean, dt_std=dt_std, du_mean=du_mean, du_std=du_std,
                 rho=rho, p=p)
        self._plot(out_dir, it, frac, t_center, logsnr_c,
                   dt_mean, dt_std, du_mean, du_std, dt_n, du_n, rho, p)
        runner.logger.info(
            f'[DtΔU] frac={frac:.0%} iter={it} | Spearman ρ={rho:.3f} (p={p:.2e}) '
            f'| saved to {out_dir}')

    # ------------------------------------------------------------------ #
    # Dt 侧
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _collect_images(self, model, runner, device):
        mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
        conds, xds, cnt = [], [], 0
        was_training = model.training
        model.eval()
        for batch in runner.val_dataloader:
            data = model.data_preprocessor(batch, training=False)
            xi = data['inputs'].to(device)
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
        if was_training:
            model.train()
        if cnt == 0:
            return None, None
        return (torch.cat(conds, 0)[:self.num_images],
                torch.cat(xds, 0)[:self.num_images])

    @torch.no_grad()
    def _measure_dt(self, model, cond, x_diffae, bins, device):
        B, unet = x_diffae.shape[0], model.diffae_unet
        was_training = model.training
        model.eval()
        means, stds = [], []
        for ts_bin in bins:
            vals = []
            for _ in range(self.samples_per_bin):
                t_val = int(np.random.choice(ts_bin))
                t = torch.full((B,), t_val, device=device, dtype=torch.long)
                noise = torch.randn_like(x_diffae)
                x_t = model.sampler.q_sample(x_diffae, t, noise)
                p_c = unet(x_t, t, cond)
                p_z = unet(x_t, t, torch.zeros_like(cond))
                if getattr(model.sampler, 'predict_x0', False):
                    ac_b = float(model.sampler.alphas_cumprod[t_val])
                    sq_a, sq_1a = ac_b ** 0.5, (1.0 - ac_b) ** 0.5
                    eps_c = (x_t - sq_a * p_c) / sq_1a
                    eps_z = (x_t - sq_a * p_z) / sq_1a
                    s_eps = ((eps_z - noise).pow(2).flatten(1).mean(1)
                             - (eps_c - noise).pow(2).flatten(1).mean(1))
                else:
                    s_eps = ((p_z - noise).pow(2).flatten(1).mean(1)
                             - (p_c - noise).pow(2).flatten(1).mean(1))
                vals.append(s_eps.median().item())
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        if was_training:
            model.train()
        return np.array(means), np.array(stds)

    # ------------------------------------------------------------------ #
    # ΔU 侧（探针微调）
    # ------------------------------------------------------------------ #
    def _collect_eval_batches(self, model, runner):
        """收集固定的下游评估 batch（所有 bin、前后共用同一批，保证可比）。"""
        cache = []
        for i, batch in enumerate(runner.val_dataloader):
            if i >= self.eval_batches:
                break
            cache.append(batch)
        return cache

    def _task_loss(self, model, batch):
        """下游任务 loss。

        !!! 按你训练脚本里下游 loss 的实际计算方式修改这里 !!!
        默认按 mmseg 1.x 约定：data_preprocessor -> model.loss(inputs, data_samples)。
        """
        data = model.data_preprocessor(batch, training=True)
        losses = model.loss(data['inputs'], data['data_samples'])
        return sum(v for v in losses.values() if 'loss' in v)

    def _denoise_loss(self, model, batch, t_val, device):
        """单 bin 内采样 t 的去噪 loss。z_sem 保留梯度（与真实训练一致）。"""
        data = model.data_preprocessor(batch, training=True)
        xi = data['inputs'].to(device)
        mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
        xd = (xi * std + mean) / 255.0 * 2.0 - 1.0
        xd = F.interpolate(xd, size=(model.image_size, model.image_size),
                           mode='bilinear', align_corners=False)

        feats = model.backbone(xi)
        z = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)
        cond = model.diffae_enc_compress(z)          # 带梯度：去噪塑造 encoder

        B = xd.shape[0]
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(xd)
        x_t = model.sampler.q_sample(xd, t, noise)
        pred = model.diffae_unet(x_t, t, cond)
        if getattr(model.sampler, 'predict_x0', False):
            ac_b = float(model.sampler.alphas_cumprod[t_val])
            sq_a, sq_1a = ac_b ** 0.5, (1.0 - ac_b) ** 0.5
            pred = (x_t - sq_a * pred) / sq_1a       # 换算到 ε 空间
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def _eval_task_loss(self, model, eval_cache):
        model.eval()
        vals = []
        for batch in eval_cache:
            vals.append(float(self._task_loss(model, batch).detach().cpu()))
        model.train()
        return float(np.mean(vals))

    def _measure_du(self, model, runner, bins, eval_cache, device):
        snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rng_state = (torch.get_rng_state(),
                     torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     np.random.get_state(), random.getstate())
        try:
            means, stds = [], []
            for ts_bin in bins:
                deltas = []
                for _ in range(self.probe_repeats):
                    model.load_state_dict(
                        {k: v.to(device) for k, v in snap.items()})
                    l0 = self._eval_task_loss(model, eval_cache)

                    opt = torch.optim.AdamW(model.parameters(), lr=self.probe_lr)
                    model.train()
                    train_iter = iter(runner.train_dataloader)
                    for _ in range(self.probe_iters):
                        try:
                            batch = next(train_iter)
                        except StopIteration:
                            train_iter = iter(runner.train_dataloader)
                            batch = next(train_iter)
                        t_val = int(np.random.choice(ts_bin))
                        loss = (self._task_loss(model, batch)
                                + self.denoise_weight
                                * self._denoise_loss(model, batch, t_val, device))
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                    l1 = self._eval_task_loss(model, eval_cache)
                    deltas.append(l0 - l1)           # loss 降幅 = 下游增益
                means.append(float(np.mean(deltas)))
                stds.append(float(np.std(deltas)))
        finally:
            model.load_state_dict({k: v.to(device) for k, v in snap.items()})
            torch.set_rng_state(rng_state[0])
            if rng_state[1] is not None:
                torch.cuda.set_rng_state_all(rng_state[1])
            np.random.set_state(rng_state[2])
            random.setstate(rng_state[3])
            model.train()
        return np.array(means), np.array(stds)

    # ------------------------------------------------------------------ #
    # 画图
    # ------------------------------------------------------------------ #
    @staticmethod
    def _minmax(a):
        r = a.max() - a.min()
        return (a - a.min()) / r if r > 1e-12 else np.zeros_like(a)

    def _plot(self, out_dir, it, frac, t_c, logsnr_c,
              dt_mean, dt_std, du_mean, du_std, dt_n, du_n, rho, p):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        order = np.argsort(-logsnr_c)  # logSNR 从大到小 = 噪声从小到大
        x = logsnr_c[order]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        # 左：归一化双曲线（论文主图形态）
        ax = axes[0]
        ax.plot(x, dt_n[order], 'o-', lw=2, label='Dt spectrum')
        ax.plot(x, du_n[order], 's--', lw=2, label='ΔU probe (outer gain)')
        ax.set_xlabel('log SNR (bin center)')
        ax.set_ylabel('min-max normalized')
        ax.set_title(f'ρ = {rho:.3f} (p = {p:.2e}) @ {frac:.0%} training')
        ax.legend()
        ax.grid(alpha=0.3)

        # 右：原值双 y 轴对照（检查量级与平坦度，防止归一化造假象）
        ax = axes[1]
        ax2 = ax.twinx()
        ax.bar(range(len(x)), dt_mean[order], alpha=0.5, label='Dt (raw)')
        ax2.plot(range(len(x)), du_mean[order], 'rs--', lw=2, label='ΔU (raw)')
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([f'{v:.1f}' for v in x], rotation=45, fontsize=8)
        ax.set_ylabel('Dt')
        ax2.set_ylabel('ΔU')
        ax.set_title('raw values')
        ax.grid(alpha=0.3)

        fig.suptitle(f'Dt–ΔU consistency @ iter {it}')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'iter_{it:06d}_consistency.png'), dpi=150)
        plt.close(fig)