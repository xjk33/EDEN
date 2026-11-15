"""
t 自适应：贡献度探针 + t 权重调度（定稿方案）

拆分自 diffae_mixin.py，两个组件各司其职：
  UtProbe          —— μt 计算：logSNR 分桶、配对探针测量（同图同噪声同 t）、
                      每轮合成 μ̂ 后做 EMA
  TWeightScheduler —— 从 μ̄ 得到采样分布：取正部归一化 + 1/K 均匀混合防坍缩

TAdaptive 把两者组合为一个 nn.Module（buffer 随 checkpoint 保存），
全部超参数在它的 __init__ 里集中初始化；DiffAEMixin 只持有一个实例，
不再自己管理 t 加权的任何细节。

方法学映射（与论文 3.2 节一一对应）：
  - 区间划分 = 在模型训练网格上按 logSNR 等频划 K 个区间，代表 t 取区间中心最近网格点
  - 每轮探针 = 每 interval 步取当前 batch 前 probe_batch 张图，对每个代表 t_j
    同图同噪声配对前向（保留/去除 z_sem），得逐样本 D_{j,i} 与 L_0
  - 贡献度   = μ̂_j = (D̂_j − k·σ̂_{D,j}) / (L̂_j + δ)
    （D̂/L̂ 为批均值，σ̂_D 为批标准差；δ 固定小常数防比值爆炸）
  - 非平稳   = 只对贡献度做 EMA：μ̄ ← ρ·μ̄ + (1−ρ)·μ̂
  - 采样分布 = p_j = (1−ε)·max(μ̄_j,0)/Σ_l max(μ̄_l,0) + ε/K，ε = 1/K
    （即 (1−1/K)·w_j + 1/K²：引导占 1−1/K，均匀保底总占 1/K，严格归一；
    正部归一化排剂量；均匀保底防坍缩，任何区间"复活"可观测）
  - 预热     = 前 warmup_iters 步只累计 μ̄，采样保持均匀
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional

from .sampler import get_named_beta_schedule


class UtProbe(nn.Module):
    """
    μt 计算：把训练网格按 logSNR 等频分桶，每 interval 步对每个桶的代表 t
    测一轮逐样本配对 D = L(zero) − L(z_sem) 与 L_0 = L(zero)（同图、同噪声、同 t），
    由本轮批均值/批标准差合成 μ̂ = (D̂ − k·σ̂_D)/(L̂ + δ)，EMA 进 mu_table。
    只负责"测"，不负责"用"。
    """

    def __init__(
        self,
        num_timesteps: int,
        beta_schedule: str,
        use_timesteps: List[int],     # 模型实际训练的 t 网格（spaced 后）
        num_bins: int = 16,
        interval: int = 500,          # 每多少训练步测一轮
        ema: float = 0.95,            # 贡献度 EMA 系数 ρ
        probe_batch: int = 16,        # 每轮探针用多少样本（B_p）
        k: float = 0.5,               # 不确定性惩罚系数（μ 分子 D − k·σ_D）
        delta: float = 1e-3,          # 量纲归一保护常数（μ 分母 E[L_0] + δ）
    ):
        super().__init__()
        self.num_bins = num_bins
        self.interval = interval
        self.ema = ema
        self.probe_batch = probe_batch
        self.k = k
        self.delta = delta
        self._probe_step = 0          # 已见的训练步数（预热计时用）

        # ---- logSNR 分桶：在【训练网格】上按 logSNR 等频划分 ----
        betas = get_named_beta_schedule(beta_schedule, num_timesteps)
        ac = np.cumprod(1.0 - betas)
        logsnr = np.log(ac / (1.0 - ac)).astype(np.float32)          # [T]
        grid = np.array(use_timesteps, dtype=np.int64)
        grid_ls = logsnr[grid]

        edges = np.quantile(grid_ls, np.linspace(0, 1, num_bins + 1)).astype(np.float32)
        edges[0] -= 1.0
        edges[-1] += 1.0
        centers = 0.5 * (edges[:-1] + edges[1:])
        rep_t = grid[np.array([np.argmin(np.abs(grid_ls - c)) for c in centers])]

        self.register_buffer('logsnr_per_t', torch.from_numpy(logsnr))
        self.register_buffer('bin_edges', torch.from_numpy(edges))
        self.register_buffer('bin_rep_t', torch.from_numpy(rep_t.astype(np.int64)))
        self.register_buffer('mu_table', torch.zeros(num_bins))      # 贡献度 μ̄ 的 EMA
        # 每个网格点的桶号（把桶权重映射回网格采样概率用）
        self.register_buffer('grid_bin', self.t_to_bin(torch.from_numpy(grid)).long())

    def t_to_bin(self, t: torch.Tensor) -> torch.Tensor:
        """t [B] long -> 桶 id [B]"""
        ls = self.logsnr_per_t.to(t.device)[t]
        return torch.bucketize(ls, self.bin_edges.to(t.device)[1:-1].contiguous())

    @torch.no_grad()
    def measure(self, unet: nn.Module, sampler, x_start: torch.Tensor,
                cond: torch.Tensor) -> bool:
        """
        轮换探针：到点（每 interval 步）就对每个桶的代表 t 测一轮逐样本配对
        D 与 L_0（同图、同噪声、同 t），由批统计合成 μ̂ 并 EMA 进 mu_table。
        返回本轮是否实际测量。
        """
        self._probe_step += 1
        if self._probe_step % self.interval != 0:
            return False

        B = min(self.probe_batch, x_start.shape[0])
        x, c = x_start[:B], cond[:B]
        dev = x.device
        K = self.num_bins

        # 去噪器切 eval 关掉 dropout，探针测量更稳（GroupNorm 不受 train/eval 影响）
        was_training = unet.training
        unet.eval()

        mu_new = torch.zeros(K, device=dev)
        for bi in range(K):
            t = torch.full((B,), int(self.bin_rep_t[bi]), device=dev, dtype=torch.long)
            noise = torch.randn_like(x)
            x_t = sampler.q_sample(x, t, noise)
            p_c = unet(x_t, t, c)
            p_z = unet(x_t, t, torch.zeros_like(c))
            # 逐样本配对损失差：D_i = L_0,i − L_z,i（loss 目标随参数化而定）
            target = x if sampler.predict_x0 else noise
            l_c = (p_c - target).pow(2).flatten(1).mean(1)
            l_z = (p_z - target).pow(2).flatten(1).mean(1)
            d_i = l_z - l_c
            # μ̂ = (D̂ − k·σ̂_D) / (L̂ + δ)：批均值、批标准差（无偏，与离线口径一致）
            d_hat = d_i.mean()
            sig_hat = d_i.std() if B > 1 else torch.zeros((), device=dev)
            l0_hat = l_z.mean()
            mu_new[bi] = (d_hat - self.k * sig_hat) / (l0_hat + self.delta)

        if was_training:
            unet.train()

        # 只对贡献度做 EMA：μ̄ ← ρ·μ̄ + (1−ρ)·μ̂（跟踪非平稳贡献度、抑制探针震荡）
        m = self.ema
        self.mu_table.mul_(m).add_(mu_new, alpha=1.0 - m)
        return True


class TWeightScheduler(nn.Module):
    """
    从贡献度表（mu_table）得到 t 采样分布，只做一步：
        p_j = (1−ε)·max(μ̄_j,0) / Σ_l max(μ̄_l,0) + ε/K，ε = 1/K（默认）
    即 (1−1/K)·w_j + 1/K²：引导部分占 1−1/K，均匀保底总占 1/K，严格归一。
    取正部：贡献非正的区间不参与分配（μ̄<=0 = 该粒度去噪约束与特征脱耦甚至有害）；
    均匀保底防坍缩：任何区间概率不为 0，训练漂移使其"复活"时可被重新观测。
    只负责"用"，不负责"测"。
    """

    def __init__(
        self,
        num_bins: int = 16,
        warmup_iters: int = 5000,      # 预热期：只累计贡献度，采样保持均匀
        eps: Optional[float] = None,   # 均匀保底总份额；None = 1/K（即 1/K² 形式）
    ):
        super().__init__()
        self.num_bins = num_bins
        self.warmup_iters = warmup_iters
        self.eps = eps
        self.register_buffer('t_weight', torch.full((num_bins,), 1.0 / num_bins))

    @torch.no_grad()
    def update(self, probe: UtProbe) -> None:
        """预热结束后刷新采样分布：取正部 -> 归一化 -> (1−ε)·w + ε/K 混合。"""
        if probe._probe_step < self.warmup_iters:
            return  # 预热期只累计贡献度

        K = self.num_bins
        eps = self.eps if self.eps is not None else 1.0 / K
        w = torch.where(torch.isfinite(probe.mu_table), probe.mu_table,
                        torch.zeros_like(probe.mu_table)).clamp_min(0.0)
        if w.sum() <= 0:
            # 全部区间贡献非正：退回均匀分布
            p = torch.full_like(w, 1.0 / K)
        else:
            p = w / w.sum() * (1.0 - eps) + eps / K   # 总和严格为 1
        self.t_weight.copy_(p)

    def sample_probs(self, probe: UtProbe) -> torch.Tensor:
        """把桶权重映射为训练网格上的采样分布。"""
        probs = self.t_weight[probe.grid_bin]
        return probs / probs.sum().clamp_min(1e-12)

    def entropy(self) -> torch.Tensor:
        """归一化熵（监控用）。"""
        w = self.t_weight
        return -(w * torch.log(w.clamp_min(1e-12))).sum() / np.log(self.num_bins)


class TAdaptive(nn.Module):
    """
    t 自适应总控：UtProbe（测 μt） + TWeightScheduler（μ̄ -> t 采样分布）。
    全部超参数在此集中初始化；buffer 随 checkpoint 保存/恢复。
    """

    def __init__(
        self,
        enabled: bool = False,
        # ---- 分桶所需的网格信息（由外层 sampler 传入）----
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        use_timesteps: Optional[List[int]] = None,
        # ---- UtProbe 超参数 ----
        bins: int = 16,
        interval: int = 500,
        ema: float = 0.95,
        probe_batch: int = 16,
        k: float = 0.5,              # μ 的不确定性惩罚系数（D − k·σ_D）
        delta: float = 1e-3,         # μ 的量纲归一保护常数（E[L_0] + δ）
        # ---- TWeightScheduler 超参数 ----
        warmup_iters: int = 5000,
        eps: Optional[float] = None,   # 均匀保底总份额；None = 1/K（1/K² 形式）
    ):
        super().__init__()
        self.enabled = enabled
        if use_timesteps is None:
            use_timesteps = list(range(num_timesteps))

        self.probe = UtProbe(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            use_timesteps=use_timesteps,
            num_bins=bins,
            interval=interval,
            ema=ema,
            probe_batch=probe_batch,
            k=k,
            delta=delta,
        )
        self.scheduler = TWeightScheduler(
            num_bins=bins,
            warmup_iters=warmup_iters,
            eps=eps,
        )
        # 实际采样桶计数（监控用；普通属性即可，不进 checkpoint，恒在 CPU）
        self._ts_counts = torch.zeros(bins)

    def on_train_step(self, unet: nn.Module, sampler, x_start: torch.Tensor,
                      cond: torch.Tensor, training: bool) -> Optional[torch.Tensor]:
        """
        每次扩散前向前调用：探针测 μt -> 调度器更新采样分布；
        返回本步 t 的采样分布（启用时），否则 None（均匀采样）。
        """
        if self.enabled and training:
            if self.probe.measure(unet, sampler, x_start, cond):
                self.scheduler.update(self.probe)
        if self.enabled:
            return self.scheduler.sample_probs(self.probe)
        return None

    @torch.no_grad()
    def record_samples(self, t: torch.Tensor) -> None:
        """记录实际采样到的桶频次（监控用，与是否启用无关）。"""
        self._ts_counts += torch.bincount(
            self.probe.t_to_bin(t).cpu(),
            minlength=self.probe.num_bins,
        ).to(self._ts_counts.device)
