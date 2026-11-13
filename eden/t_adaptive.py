"""
t 自适应：Dt 探针 + t 权重调度（discounted-EXP3 / Hedge）

拆分自 diffae_mixin.py，两个组件各司其职：
  DtProbe          —— Dt 计算：logSNR 分桶、配对探针测量（同图同噪声同 t）、收益 EMA
  TWeightScheduler —— 从 Dt 得到 wt：Hedge/EXP3 权重更新，
                      以采样分布（apply='sample'）或 loss 加权（apply='loss'）控制 t

TAdaptive 把两者组合为一个 nn.Module（buffer 随 checkpoint 保存），
全部超参数在它的 __init__ 里集中初始化；DiffAEMixin 只持有一个实例，
不再自己管理 t 加权的任何细节。

方法学映射：
  - 臂(arm)  = logSNR 区间（按模型训练网格等频划分）
  - 每臂收益 = 贡献度 μ = (E[D] − k·σ_D) / (E[L_0] + δ)，D = L(zero) − L(z_sem)
    （配对同图同噪声；期望/方差由在线 EMA 矩累积估计，见 DtProbe）
  - 非平稳   = 收益 EMA（动量平滑在探针侧完成，权重侧不再叠加滤波）
  - 权重更新 = 取正部 + 归一化 + [p_min, p_max] 防坍缩截断（简化控制器）
  - 探索项   = p_min 探索地板（任何区间概率不为 0，"复活"可观测）
  - 预热     = 前 warmup_iters 步只累计收益，权重保持均匀
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional

from .sampler import get_named_beta_schedule


class DtProbe(nn.Module):
    """
    Dt / μt 计算：把训练网格按 logSNR 等频分桶，轮换探针测量每桶的逐样本配对
    D = L(cond=0) - L(z_sem) 与 L_0 = L(cond=0)（同图、同噪声、同 t），
    在线 EMA 累积 D 的一阶/二阶矩与 L_0 均值，合成贡献度
        μ = (E[D] − k·σ_D) / (E[L_0] + δ)   （σ_D = sqrt(max(E[D²]−E[D]², 0))）
    均值/方差由在线矩累积估计，无需单次大批量；离群值的影响由 k·σ_D 项显式承担
    （因此逐轮聚合用均值——E[D] 的无偏估计——而非旧版的中位数）。
    只负责"测"，不负责"用"。
    """

    def __init__(
        self,
        num_timesteps: int,
        beta_schedule: str,
        use_timesteps: List[int],     # 模型实际训练的 t 网格（spaced 后）
        num_bins: int = 16,
        interval: int = 500,          # 每多少训练步测一轮
        ema: float = 0.95,            # 收益 EMA 系数（非平稳折扣近似）
        probe_batch: int = 16,        # 每轮探针最多用多少样本
        k: float = 1.0,               # 不确定性惩罚系数（μ 分子 D − k·σ_D）
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
        self.register_buffer('d_table', torch.zeros(num_bins))       # E[D] 的 EMA（一阶矩）
        self.register_buffer('d_sq_table', torch.zeros(num_bins))    # E[D²] 的 EMA（二阶矩，算 σ_D）
        self.register_buffer('l0_table', torch.zeros(num_bins))      # E[L_0] 的 EMA（量纲归一分母）
        self.register_buffer('mu_table', torch.zeros(num_bins))      # 贡献度 μ（Hedge 的收益表）
        self.register_buffer('d_eps_table', torch.zeros(num_bins))   # 诊断：epsilon 空间 D
        self.register_buffer('d_cos_table', torch.zeros(num_bins))   # joint：cos 计分（增益不变）
        self.register_buffer('d_count', torch.zeros(num_bins))
        # 每个网格点的桶号（apply='sample' 时把桶权重归一到网格概率）
        self.register_buffer('grid_bin', self.t_to_bin(torch.from_numpy(grid)).long())

    def t_to_bin(self, t: torch.Tensor) -> torch.Tensor:
        """t [B] long -> 桶 id [B]"""
        ls = self.logsnr_per_t.to(t.device)[t]
        return torch.bucketize(ls, self.bin_edges.to(t.device)[1:-1].contiguous())

    @torch.no_grad()
    def measure(self, unet: nn.Module, sampler, x_start: torch.Tensor,
                cond: torch.Tensor, joint: bool = False) -> bool:
        """
        轮换探针：到点（每 interval 步）就对每个桶的代表 t 测一轮逐样本配对
        D 与 L_0（同图、同噪声、同 t），把 batch 均值的 D / D² / L_0 EMA 进
        在线矩表，并刷新贡献度 μ。返回本轮是否实际测量。
        """
        self._probe_step += 1
        if self._probe_step % self.interval != 0:
            return False

        B = min(self.probe_batch, x_start.shape[0])
        x, c = x_start[:B], cond[:B]
        dev = x.device
        K = self.num_bins

        # UNet 切 eval 关掉 dropout，探针测量更稳（GroupNorm 不受 train/eval 影响）
        was_training = unet.training
        unet.eval()

        d_mean_new = torch.zeros(K, device=dev)
        d_sq_new = torch.zeros(K, device=dev)
        l0_new = torch.zeros(K, device=dev)
        d_eps_new = torch.zeros(K, device=dev)
        d_cos_new = torch.zeros(K, device=dev)   # cos 计分（joint）
        for bi in range(K):
            t = torch.full((B,), int(self.bin_rep_t[bi]), device=dev, dtype=torch.long)
            noise = torch.randn_like(x)
            x_t = sampler.q_sample(x, t, noise)
            p_c = unet(x_t, t, c)
            p_z = unet(x_t, t, torch.zeros_like(c))
            # 逐样本配对损失差：D_i = L_0,i − L_z,i（loss 目标随参数化而定）
            if sampler.predict_x0:
                l_c = (p_c - x).pow(2).flatten(1).mean(1)
                l_z = (p_z - x).pow(2).flatten(1).mean(1)
            else:
                l_c = (p_c - noise).pow(2).flatten(1).mean(1)
                l_z = (p_z - noise).pow(2).flatten(1).mean(1)
            d_i = l_z - l_c
            d_mean_new[bi] = d_i.mean()
            d_sq_new[bi] = d_i.pow(2).mean()
            l0_new[bi] = l_z.mean()
            # 诊断：换算到 epsilon 空间的 D（含 1/√(1-ᾱ) 增益 -> 天然偏置低 t）
            if sampler.predict_x0:
                ac_b = float(sampler.alphas_cumprod[int(self.bin_rep_t[bi])])
                sq_a, sq_1a = ac_b ** 0.5, (1.0 - ac_b) ** 0.5
                eps_c = (x_t - sq_a * p_c) / sq_1a
                eps_z = (x_t - sq_a * p_z) / sq_1a
                d_eps = (eps_z - noise).pow(2).flatten(1).mean(1) - (eps_c - noise).pow(2).flatten(1).mean(1)
                d_eps_new[bi] = d_eps.median()
            else:
                d_eps_new[bi] = d_i.median()   # eps 模式下主量与诊断同源，保留中位数抗离群
            if joint:
                # 1 - cos(ε̂_cond, ε̂_zero)：仿射不变 => 与参数化(ε/x0)无关；
                # 尺度不变 => 免疫两种增益爆炸；语义 = z_sem 对预测方向的相对影响
                a = p_c.flatten(1)
                b = p_z.flatten(1)
                cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)
                d_cos_new[bi] = (1.0 - cos).median()

        if was_training:
            unet.train()

        # 在线矩 EMA（非平稳折扣近似）：期望/方差跨轮累积，无需单次大批量
        m = self.ema
        self.d_table.mul_(m).add_(d_mean_new, alpha=1.0 - m)
        self.d_sq_table.mul_(m).add_(d_sq_new, alpha=1.0 - m)
        self.l0_table.mul_(m).add_(l0_new, alpha=1.0 - m)
        self.d_eps_table.mul_(m).add_(d_eps_new, alpha=1.0 - m)
        self.d_cos_table.mul_(m).add_(d_cos_new, alpha=1.0 - m)
        self.d_count.add_(1.0)

        # 贡献度 μ = (E[D] − k·σ_D) / (E[L_0] + δ)，σ_D 由在线二阶矩给出
        var = (self.d_sq_table - self.d_table.pow(2)).clamp_min(0.0)
        self.mu_table.copy_(
            (self.d_table - self.k * var.sqrt()) / (self.l0_table + self.delta))
        return True


class TWeightScheduler(nn.Module):
    """
    从贡献度收益表（mu_table）得到 t 权重 wt：动量平滑已在探针侧完成
    （在线矩 EMA），权重侧只做三步——取正部、归一化、防坍缩截断。
    提供两种应用方式：
      apply='sample' —— 区间权重映射回训练网格，作为 t 的采样分布（每步满强度）
      apply='loss'   —— 按区间加权扩散 loss（期望等价，有限 batch 下强度不均）
    只负责"用"，不负责"测"。
    """

    def __init__(
        self,
        num_bins: int = 16,
        warmup_iters: int = 5000,      # 预热期：只累计收益，权重保持均匀
        p_min: float = 0.005,          # 探索地板（每区间概率下界，防永久封杀）
        p_max: float = 0.3,            # 防坍缩截断（每区间概率上界，防单区间劫持）
        joint: bool = False,           # True = 用 cos 计分（增益不变、天然中段峰）
        apply: str = 'loss',           # 'loss' | 'sample'
        static_window: Optional[List[int]] = None,  # 如 [100,500]：权重固定为窗内均匀，禁用动态更新
        probe_rep_t: Optional[torch.Tensor] = None,  # 静态窗口初始化需要各区间代表 t
    ):
        super().__init__()
        assert apply in ('loss', 'sample'), f"unknown apply mode: {apply}"
        assert 0.0 <= p_min < 1.0 / num_bins < p_max <= 1.0, \
            f'需满足 0 <= p_min < 1/K < p_max <= 1，得到 p_min={p_min}, p_max={p_max}, K={num_bins}'
        self.num_bins = num_bins
        self.warmup_iters = warmup_iters
        self.p_min = p_min
        self.p_max = p_max
        self.joint = joint
        # 注意：不能命名为 self.apply——会覆盖 nn.Module.apply()，
        # 导致 model.to(device) / .apply(fn) 递归时 TypeError: 'str' object is not callable
        self.apply_mode = apply
        self.static_window = static_window

        self.register_buffer('t_weight', torch.full((num_bins,), 1.0 / num_bins))

        # 静态窗口模式：窗口内均分剩余概率、窗口外探索地板
        if static_window is not None:
            lo, hi = static_window
            in_win = (probe_rep_t >= lo) & (probe_rep_t <= hi)
            n_in = int(in_win.sum())
            if n_in == 0:
                raise ValueError(
                    f"t_weight_static_window [{lo}, {hi}] 内没有任何区间的代表 t "
                    f"(rep_t 范围 [{probe_rep_t.min()}, {probe_rep_t.max()}])，请检查窗口设置。"
                )
            w = torch.full((num_bins,), p_min)
            w[in_win] += (1.0 - p_min * num_bins) / n_in
            self.t_weight.copy_(w)

    @torch.no_grad()
    def update(self, probe: DtProbe) -> None:
        """预热结束且非静态窗口时刷新权重：取正部 -> 归一化 -> 截断 -> 再归一化。"""
        if self.static_window is not None:
            return  # 静态窗口：权重在 __init__ 已固定，探针只记录
        if probe._probe_step < self.warmup_iters:
            return  # 预热期只累计收益

        score = probe.d_cos_table if self.joint else probe.mu_table
        K = self.num_bins

        # 取正部：贡献非正的区间不参与分配（μ<=0 = 该粒度去噪约束与特征脱耦甚至有害）。
        # 它们仍由 p_min 的探索地板偶尔采到——训练漂移使其"复活"时可被重新观测。
        w = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
        w = w.clamp_min(0.0)
        if w.sum() <= 0:
            # 全部区间贡献非正：退回均匀分布
            p = torch.full_like(w, 1.0 / K)
        else:
            p = w / w.sum()
        # 防坍缩截断 + 再归一化（截断会破坏归一性，迭代两轮近似收敛）
        for _ in range(2):
            p = p.clamp(self.p_min, self.p_max)
            p = p / p.sum()
        self.t_weight.copy_(p)

    def sample_probs(self, probe: DtProbe) -> Optional[torch.Tensor]:
        """apply='sample'：把桶权重映射为训练网格上的采样分布；否则返回 None。"""
        if self.apply_mode != 'sample':
            return None
        probs = self.t_weight[probe.grid_bin]
        return probs / probs.sum().clamp_min(1e-12)

    def weight_loss(self, loss_per_sample: torch.Tensor, t: torch.Tensor,
                    probe: DtProbe) -> torch.Tensor:
        """apply='loss'：按桶加权 loss；否则均匀聚合。"""
        if self.apply_mode != 'loss':
            return loss_per_sample.mean()
        w = self.t_weight.to(loss_per_sample.device)[probe.t_to_bin(t)]
        return (loss_per_sample * w).sum() / w.sum().clamp_min(1e-12)

    def entropy(self) -> torch.Tensor:
        """归一化熵（监控用）。"""
        w = self.t_weight
        return -(w * torch.log(w.clamp_min(1e-12))).sum() / np.log(self.num_bins)


class TAdaptive(nn.Module):
    """
    t 自适应总控：DtProbe（测 Dt/μt） + TWeightScheduler（μt -> wt -> 控制采样）。
    全部超参数在此集中初始化；buffer 随 checkpoint 保存/恢复。
    """

    def __init__(
        self,
        enabled: bool = False,
        # ---- 分桶所需的网格信息（由外层 sampler 传入）----
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        use_timesteps: Optional[List[int]] = None,
        # ---- DtProbe 超参数 ----
        bins: int = 16,
        interval: int = 500,
        ema: float = 0.95,
        probe_batch: int = 16,
        k: float = 1.0,              # μ 的不确定性惩罚系数（D − k·σ_D）
        delta: float = 1e-3,         # μ 的量纲归一保护常数（E[L_0] + δ）
        # ---- TWeightScheduler 超参数 ----
        warmup_iters: int = 5000,
        p_min: float = 0.005,
        p_max: float = 0.3,
        joint: bool = False,
        apply: str = 'loss',
        static_window: Optional[List[int]] = None,
    ):
        super().__init__()
        self.enabled = enabled
        if use_timesteps is None:
            use_timesteps = list(range(num_timesteps))

        self.probe = DtProbe(
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
            p_min=p_min,
            p_max=p_max,
            joint=joint,
            apply=apply,
            static_window=static_window,
            probe_rep_t=self.probe.bin_rep_t,
        )
        # 实际采样桶计数（监控用；普通属性即可，不进 checkpoint，恒在 CPU）
        self._ts_counts = torch.zeros(bins)

    def on_train_step(self, unet: nn.Module, sampler, x_start: torch.Tensor,
                      cond: torch.Tensor, training: bool) -> Optional[torch.Tensor]:
        """
        每次扩散前向前调用：探针测量 Dt -> 调度器更新 wt；
        返回本步 t 的采样分布（apply='sample' 时），否则 None（均匀采样）。
        """
        if self.enabled and training:
            if self.probe.measure(unet, sampler, x_start, cond,
                                  joint=self.scheduler.joint):
                self.scheduler.update(self.probe)
        if self.enabled:
            return self.scheduler.sample_probs(self.probe)
        return None

    def weight_loss(self, loss_per_sample: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """扩散 loss 聚合：启用时按调度器规则加权，未启用时均匀。"""
        if not self.enabled:
            return loss_per_sample.mean()
        return self.scheduler.weight_loss(loss_per_sample, t, self.probe)

    @torch.no_grad()
    def record_samples(self, t: torch.Tensor) -> None:
        """记录实际采样到的桶频次（监控用，与是否启用无关）。"""
        self._ts_counts += torch.bincount(
            self.probe.t_to_bin(t).cpu(),
            minlength=self.probe.num_bins,
        ).to(self._ts_counts.device)
