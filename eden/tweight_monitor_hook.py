"""
TWeightMonitorHook（v2）：每次验证后记录 t_weight 与【实际采样分布】。

在 v1（权重柱状图 + D 表叠加）基础上新增：
  - 训练中 mixin 累积每个 logSNR 桶被实际采到的次数（_ts_counts），本 hook 读取、
    归一化后作为折线叠加在权重柱上（'realized t freq'），读取后清零重新累积。
  - 标题给出期望/实测的采样 t 均值；npz 与 CSV 同步新增 ts_freq / ts_mean_* 字段。
  - config 开关：log_tstats=False 可关闭采样统计（退回 v1 行为）。

产出（work_dir/tweight_curves/）：
  iter_XXXXXX_tweight.png / .npz / tweight_history.csv（列: iter, w_0..w_K-1, ts_mean_realized, ts_mean_expected）
"""
import os
import csv
import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class TWeightMonitorHook(Hook):
    def __init__(self, log_tstats: bool = True, priority: str = 'LOW'):
        self.log_tstats = log_tstats
        self._priority = priority

    @torch.no_grad()
    def after_val_epoch(self, runner, metrics=None) -> None:
        model = runner.model
        if not hasattr(model, 't_weight') or not hasattr(model, 'bin_rep_t'):
            return

        w = model.t_weight.detach().float().cpu().numpy()
        d = model.d_table.detach().float().cpu().numpy()
        rep_t = model.bin_rep_t.detach().cpu().numpy().astype(int)
        ab = model.sampler.alphas_cumprod.detach().float().cpu().numpy()
        logsnr = np.log(ab / (1.0 - ab))[rep_t]
        K = len(w)
        it = runner.iter

        # 实际采样分布（自上次验证以来累积的桶计数）
        ts_freq = None
        ts_mean_realized = float('nan')
        if self.log_tstats and hasattr(model, '_ts_counts'):
            cnt = model._ts_counts.detach().float().cpu()
            tot = cnt.sum().item()
            if tot > 0:
                ts_freq = (cnt / tot).numpy()
                ts_mean_realized = float((cnt * torch.from_numpy(rep_t).float()).sum() / tot)
                model._ts_counts.zero_()          # 读取后清零，重新累积
        probs = w / w.sum()
        ts_grid = model.sampler.use_timesteps
        ts_mean_expected = float(np.dot(probs, rep_t))

        out_dir = os.path.join(runner.work_dir, 'tweight_curves')
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, f'iter_{it:06d}_tweight.npz'),
                 w=w, d_table=d, rep_t=rep_t, logsnr=logsnr,
                 ts_freq=ts_freq if ts_freq is not None else np.full(K, np.nan),
                 ts_mean_realized=ts_mean_realized, ts_mean_expected=ts_mean_expected)

        hist_path = os.path.join(out_dir, 'tweight_history.csv')
        with open(hist_path, 'a', newline='') as f:
            csv.writer(f).writerow([it] + [f'{x:.6f}' for x in w]
                                   + [f'{ts_mean_realized:.2f}', f'{ts_mean_expected:.2f}'])

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        d_n = d / np.abs(d).max() * w.max() if np.abs(d).max() > 0 else d
        for ax, x, xlab in [(axes[0], rep_t, 't (bin rep)'),
                            (axes[1], logsnr, 'log SNR (bin rep)')]:
            ax.bar(x, w, width=(x[1] - x[0]) * 0.7 if K > 1 else 10,
                   alpha=0.7, label='t_weight (sampling dist)' )
            ax.plot(x, d_n, color='tab:red', marker='o', ms=3, label='D table (scaled)')
            if ts_freq is not None:
                ax.plot(x, ts_freq, color='tab:green', marker='s', ms=3, lw=1.8,
                        label='realized t freq')
            ax.axhline(1.0 / K, color='k', ls='--', lw=0.8, label='uniform')
            ax.set_xlabel(xlab)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel('prob / weight')
        ent = -(w * np.log(np.clip(w, 1e-12, None))).sum() / np.log(K)
        fig.suptitle(f't_weight @ iter {it} | entropy={ent:.3f} | max={w.max():.3f}@{int(w.argmax())} '
                     f'| ts_mean realized={ts_mean_realized:.0f} expected={ts_mean_expected:.0f}')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'iter_{it:06d}_tweight.png'), dpi=150)
        plt.close(fig)
        runner.logger.info(f'[TWeightMonitor] t_weight saved: {out_dir}/iter_{it:06d}_tweight.png '
                           f'(entropy={ent:.3f}, ts_mean realized={ts_mean_realized:.0f} '
                           f'expected={ts_mean_expected:.0f})')