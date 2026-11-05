"""
t 自适应：Dt 探针 + t 权重调度（discounted-EXP3 / Hedge）

拆分自 diffae_mixin.py，两个组件各司其职：
  DtProbe          —— Dt 计算：logSNR 分桶、配对探针测量（同图同噪声同 t）、收益 EMA
  TWeightScheduler —— 从 Dt 得到 wt：Hedge/EXP3 权重更新，
                      以采样分布（apply='sample'）或 loss 加权（apply='loss'）控制 t

TAdaptive 把两者组合为一个 nn.Module（buffer 随 checkpoint 保存），
全部超参数在它的 __init__ 里集中初始化；DiffAEMixin 只持有一个实例，
不再自己管理 t 加权的任何细节。

方法学映射（全部为教科书算法，无自造成分）：
  - 臂(arm)  = logSNR 桶（按模型训练网格等频划分）
  - 每臂收益 = 探针测得的 per-桶 D = median(L(zero) - L(z_sem))，配对同图同噪声
  - 权重更新 = 指数加权 / Hedge（Freund & Schapire 1997），对抗版本即 EXP3（Auer et al. 2002）
  - 探索项   = eps-均匀混合（EXP3 标准做法，保证任何桶权重不为 0）
  - 非平稳   = 收益 EMA（折扣 EXP3 的工程近似，Garivier & Moulines 2008 一族）
  - 预热     = 前 warmup_iters 步只累计收益，权重保持均匀
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional

from .sampler import get_named_beta_schedule


class DtProbe(nn.Module):
    """
    Dt 计算：把训练网格按 logSNR 等频分桶，轮换探针测量每桶的
    D = median( L(cond=0) - L(z_sem) )（同图、同噪声、同 t，配对），
    结果 EMA 进收益表。只负责"测"，不负责"用"。
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
    ):
        super().__init__()
        self.num_bins = num_bins
        self.interval = interval
        self.ema = ema
        self.probe_batch = probe_batch
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
        self.register_buffer('d_table', torch.zeros(num_bins))       # 主收益表（x0/eps 视参数化而定）
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
        轮换探针：到点（每 interval 步）就对每个桶的代表 t 测一轮配对 D，
        EMA 进收益表。返回本轮是否实际测量。
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

        d_new = torch.zeros(K, device=dev)
        d_eps_new = torch.zeros(K, device=dev)
        d_cos_new = torch.zeros(K, device=dev)   # cos 计分（joint）
        for bi in range(K):
            t = torch.full((B,), int(self.bin_rep_t[bi]), device=dev, dtype=torch.long)
            noise = torch.randn_like(x)
            x_t = sampler.q_sample(x, t, noise)
            p_c = unet(x_t, t, c)
            p_z = unet(x_t, t, torch.zeros_like(c))
            if sampler.predict_x0:
                # x0 模式：pred 即 x̂0。同一次前向读出两种 D：
                #   d_x0 = cond 对重建的影响（直接、有界）
                #   d_eps = 换算到 epsilon 空间（固定公式，含 1/√(1-ᾱ) 增益 -> 天然偏置低 t）
                d_x0 = (p_z - p_c).pow(2).flatten(1).mean(1)
                d_new[bi] = d_x0.median()
                ac_b = float(sampler.alphas_cumprod[int(self.bin_rep_t[bi])])
                sq_a, sq_1a = ac_b ** 0.5, (1.0 - ac_b) ** 0.5
                eps_c = (x_t - sq_a * p_c) / sq_1a
                eps_z = (x_t - sq_a * p_z) / sq_1a
                d_eps = (eps_z - noise).pow(2).flatten(1).mean(1) - (eps_c - noise).pow(2).flatten(1).mean(1)
                d_eps_new[bi] = d_eps.median()
            else:
                l_c = (p_c - noise).pow(2).flatten(1).mean(1)
                l_z = (p_z - noise).pow(2).flatten(1).mean(1)
                d_med = (l_z - l_c).median()   # 中位数抗离群
                d_eps_new[bi] = d_med
                d_new[bi] = d_med              # 非 joint 时 d_table 就是 deps（原行为）
                if joint:
                    # joint 时 d_table 改用换算到 x̂0 空间的读数（cos 计分的伴生诊断）
                    ac_b = float(sampler.alphas_cumprod[int(self.bin_rep_t[bi])])
                    sq_a, sq_1a = ac_b ** 0.5, (1.0 - ac_b) ** 0.5
                    x0_c = (x_t - sq_1a * p_c) / sq_a
                    x0_z = (x_t - sq_1a * p_z) / sq_a
                    d_new[bi] = (x0_z - x0_c).pow(2).flatten(1).mean(1).median()
            if joint:
                # 1 - cos(ε̂_cond, ε̂_zero)：仿射不变 => 与参数化(ε/x0)无关；
                # 尺度不变 => 免疫两种增益爆炸；语义 = z_sem 对预测方向的相对影响
                a = p_c.flatten(1)
                b = p_z.flatten(1)
                cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)
                d_cos_new[bi] = (1.0 - cos).median()

        if was_training:
            unet.train()

        # 收益 EMA（非平稳折扣近似）
        m = self.ema
        self.d_table.mul_(m).add_(d_new, alpha=1.0 - m)
        self.d_eps_table.mul_(m).add_(d_eps_new, alpha=1.0 - m)
        self.d_cos_table.mul_(m).add_(d_cos_new, alpha=1.0 - m)
        self.d_count.add_(1.0)
        return True


class TWeightScheduler(nn.Module):
    """
    从 Dt 收益表得到 t 权重 wt（Hedge/EXP3），并提供两种应用方式：
      apply='sample' —— 桶权重映射回训练网格，作为 t 的采样分布（每步满强度）
      apply='loss'   —— 按桶加权扩散 loss（期望等价，有限 batch 下强度不均）
    只负责"用"，不负责"测"。
    """

    def __init__(
        self,
        num_bins: int = 16,
        warmup_iters: int = 5000,      # 预热期：只累计收益，权重保持均匀
        eta: float = 2.0,              # Hedge 学习率（z-score 放大倍数）
        eps: float = 0.1,              # 均匀探索混合比例
        z_cap: float = 1.0,            # 标准化分数截断（防单桶劫持：cap=1 -> 单桶上限约 1/3）
        domain: Optional[List[float]] = None,     # logSNR 定义域 [lo, hi]；域外桶恒压地板
        pos_only: bool = False,        # True = 正值竞争（D<=0/域外一律地板）；False = 全局带符号 softmax
        joint: bool = False,           # True = 用 cos 计分（增益不变、天然中段峰）
        apply: str = 'loss',           # 'loss' | 'sample'
        static_window: Optional[List[int]] = None,  # 如 [100,500]：权重固定为窗内均匀，禁用动态更新
        probe_rep_t: Optional[torch.Tensor] = None,  # 静态窗口初始化需要各桶代表 t
    ):
        super().__init__()
        assert apply in ('loss', 'sample'), f"unknown apply mode: {apply}"
        self.num_bins = num_bins
        self.warmup_iters = warmup_iters
        self.eta = eta
        self.eps = eps
        self.cap = z_cap
        self.domain = domain
        self.pos_only = pos_only
        self.joint = joint
        # 注意：不能命名为 self.apply——会覆盖 nn.Module.apply()，
        # 导致 model.to(device) / .apply(fn) 递归时 TypeError: 'str' object is not callable
        self.apply_mode = apply
        self.static_window = static_window

        self.register_buffer('t_weight', torch.full((num_bins,), 1.0 / num_bins))

        # 静态窗口模式：权重固定为窗口内桶均分、窗口外地板（与 eps 均匀混合）
        if static_window is not None:
            lo, hi = static_window
            in_win = (probe_rep_t >= lo) & (probe_rep_t <= hi)
            n_in = int(in_win.sum())
            if n_in == 0:
                raise ValueError(
                    f"t_weight_static_window [{lo}, {hi}] 内没有任何桶的代表 t "
                    f"(rep_t 范围 [{probe_rep_t.min()}, {probe_rep_t.max()}])，请检查窗口设置。"
                )
            w = torch.full((num_bins,), eps / num_bins)
            w[in_win] = (1.0 - eps) / n_in + eps / num_bins
            self.t_weight.copy_(w)

    @torch.no_grad()
    def update(self, probe: DtProbe) -> None:
        """预热结束且非静态窗口时，按 Hedge/EXP3 从收益表刷新权重。"""
        if self.static_window is not None:
            return  # 静态窗口：权重在 __init__ 已固定，探针只记录
        if probe._probe_step < self.warmup_iters:
            return  # 预热期只累计收益

        score = probe.d_cos_table if self.joint else probe.d_table
        K = self.num_bins
        cap = self.cap

        # 参与竞争的桶 = 定义域内 且 score > 0（正值 = z_sem 对该 t 的去噪有正贡献）。
        # 其余（负贡献、零贡献、域外）一律给 -cap：真抑制（拿探索地板），不靠符号运气。
        ls_bin = probe.logsnr_per_t.to(score.device)[probe.bin_rep_t]   # 每桶代表 t 的 logSNR
        inside = torch.ones(K, dtype=torch.bool, device=score.device)
        if self.domain is not None:
            lo, hi = self.domain
            inside = (ls_bin >= lo) & (ls_bin <= hi)
        if self.pos_only:
            # 正值竞争 + 域外地板（升级路径；防低 t 劫持、防负值语义混杂）
            # 有限性过滤：inf/NaN 桶不许参赛（inf>0 会 True，必须显式挡）
            pos = inside & (score > 0) & torch.isfinite(score)
            n_pos = int(pos.sum())
            if n_pos == 0:
                if not getattr(self, '_warned_flat', False):
                    self._warned_flat = True
                    print('[t_weight] WARNING: no positive bin (all D<=0 or outside domain), '
                          'weights stay at floor.')
                z = torch.zeros(K, device=score.device)
            elif n_pos == 1:
                # 单桶退化的保护：无偏 std 在 n=1 时是 NaN；独苗直接给满分，不做除法
                z = torch.where(pos, torch.full_like(score, cap),
                                torch.full_like(score, -cap))
            else:
                mu = score[pos].mean()
                sd = score[pos].std(correction=0).clamp_min(1e-12)   # 有偏 std，n>=2 恒有限
                z = torch.full_like(score, -cap)
                z = torch.where(pos, self.eta * (score - mu) / sd, z)
        else:
            # 全局带符号 softmax + z_cap（对照实验做法）
            std = score.std().clamp_min(1e-12)
            z = self.eta * (score - score.mean()) / std
        z = z.clamp(-cap, cap)
        w = torch.exp(z - z.max())
        w = w / w.sum()
        self.t_weight.mul_(0.0).add_((1.0 - self.eps) * w + self.eps / K)

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
    t 自适应总控：DtProbe（测 Dt） + TWeightScheduler（Dt -> wt -> 控制采样）。
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
        # ---- TWeightScheduler 超参数 ----
        warmup_iters: int = 5000,
        eta: float = 2.0,
        eps: float = 0.1,
        z_cap: float = 1.0,
        domain: Optional[List[float]] = None,
        pos_only: bool = False,
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
        )
        self.scheduler = TWeightScheduler(
            num_bins=bins,
            warmup_iters=warmup_iters,
            eta=eta,
            eps=eps,
            z_cap=z_cap,
            domain=domain,
            pos_only=pos_only,
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
