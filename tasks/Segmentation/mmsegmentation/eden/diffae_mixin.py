"""
DiffAE Mixin + EMA Manager + 动态 t 加权（discounted-EXP3 / Hedge）

设计原则：
  - 只提供扩散预训练的"能力"和"工具函数"。
  - 不假设任何下游框架的 forward / train_step 签名。

v2：x0 预测 + 空间蒙块（单次 loss 全包，无辅助 loss）
  - diffae_pred_x0=True 时 UNet 直接预测 x0：可见区=去噪，蒙区=内容重构（唯一信息源 z_sem）
  - 蒙块模式：'random' 均匀 | 'hot' 蒙 seg 高敏感区（高赌注） | 'cold' 蒙低敏感区（低赌注，
    z_sem 持全局信息补全）。hot/cold 用 seg 输入梯度热力图选址（每步 autograd.grad，不写参数梯度）
  - 先加噪后蒙 x_t（蒙区输入为 0，内容与噪声同时归零）；loss 全程在图像空间，任何 t 有界
t 加权的方法映射（全部为教科书算法，无自造成分）：
  - 臂(arm)        = logSNR 桶（按模型训练网格等频划分）
  - 每臂收益       = 探针测得的 per-桶 D = median(L(zero) - L(z_sem))，配对同图同噪声
  - 权重更新       = 指数加权 / Hedge（Freund & Schapire 1997），对抗版本即 EXP3（Auer et al. 2002）
  - 探索项         = eps-均匀混合（EXP3 标准做法，保证任何桶权重不为 0）
  - 非平稳处理     = 收益 EMA（折扣 EXP3 的工程近似，Garivier & Moulines 2008 一族）
  - 预热           = 前 t_weight_warmup_iters 步只累计收益，权重保持均匀
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List

from .sampler import DDIMSampler, space_timesteps, get_named_beta_schedule
from .unet import ConditionalUNet


class EMAManager:
    """指数移动平均管理器。可管理单个模块或模块列表，按参数对象去重，
    不注册为 nn.Module buffer（也不通过 ModuleDict 二次注册——
    二次注册会让同一参数在模型里有两个名字，导致 paramwise 分组报重复）。"""

    def __init__(self, model, decay: float = 0.9999):
        self.decay = decay
        self.models = model if isinstance(model, (list, tuple)) else [model]
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._tracked = []          # [(name, param)]，构建时按 id 去重

        seen = set()
        for m in self.models:
            for name, param in m.named_parameters():
                if param.requires_grad and id(param) not in seen:
                    seen.add(id(param))
                    self._tracked.append((name, param))
                    self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def step(self):
        for name, param in self._tracked:
            if self.shadow[name].device != param.device:
                self.shadow[name] = self.shadow[name].to(param.device)
            self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self):
        self.backup.clear()
        for name, param in self._tracked:
            self.backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self._tracked:
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.shadow = state_dict


class DiffAEMixin(nn.Module):
    """
    DiffAE 功能 Mixin。

    用法：
        class MyFCN(DiffAEMixin, EncoderDecoder):
            def encode_diffae(self, x):
                feat = self.backbone(x)[-1]
                return F.adaptive_avg_pool2d(feat, 1).flatten(1)

            def loss(self, inputs, data_samples):
                losses = super().loss(inputs, data_samples)
                x_diffae = self.normalize_to_diffae(inputs)
                cond = self.diffae_enc_compress(self.encode_diffae(inputs))
                diff_out = self.forward_diffae(x_diffae, cond)
                losses['loss_diffae'] = diff_out['loss_diffae']
                return losses
    """

    def __init__(
        self,
        *args,
        image_size: int = 128,
        in_channels: int = 3,
        cond_dim: int = 512,
        enc_output_dim: int = 2048,
        unet_channels: int = 128,
        unet_channel_mult: Tuple[int, ...] = (1, 1, 2, 3, 4),
        num_diffusion_timesteps: int = 1000,
        beta_schedule: str = "linear",
        spaced_timesteps: Optional[str] = None,
        ema_decay: float = 0.9999,
        lambda_diff: float = 1.0,
        use_zero_module: bool = True,
        auto_ema: bool = True,
        num_res_blocks: int = 2,
        custom_timestep_sampler: Optional[List[int]] = None,
        # ---- 动态 t 加权（discounted-EXP3 / Hedge）----
        t_weight_enabled: bool = False,
        t_weight_warmup_iters: int = 5000,
        t_weight_eta: float = 2.0,
        t_weight_eps: float = 0.1,
        t_weight_bins: int = 16,
        t_weight_interval: int = 500,
        t_weight_ema: float = 0.95,
        t_weight_probe_batch: int = 16,
        t_weight_static_window: Optional[List[int]] = None,  # 如 [100,500]，设置后禁用动态更新
        t_weight_joint: bool = False,  # True = cos 计分：1-cos(ε̂_cond, ε̂_zero)，增益不变、天然中段峰
        t_weight_z_cap: float = 1.0,   # 标准化分数截断（防单桶权重劫持：cap=1 -> 单桶上限约 1/3）
        t_weight_domain: Optional[List[float]] = None,  # logSNR 定义域 [lo, hi]；域外桶恒压地板
        t_weight_apply: str = 'loss',     # 'loss' = 加权 loss；'sample' = 作为 t 的采样分布
        t_weight_pos_only: bool = False,  # False = 之前的做法（全局带符号 softmax）；True = 正值竞争+domain
        # ---- v2：x0 预测 + 空间蒙块（单次 loss 全包，无辅助 loss）----
        diffae_pred_x0: bool = False,         # True = 网络直接预测 x0（可见区去噪 + 蒙区靠 z_sem 重构）
        diffae_mask_ratio: float = 0.0,       # 蒙块比例 0~1，0 = 关闭
        diffae_mask_patch: int = 16,          # 蒙块边长（像素）
        diffae_mask_mode: str = "random",     # 'random' | 'hot' | 'cold'（后两者需 seg 热力图）
        diffae_mask_temp: float = 1.0,        # 热力采样温度：<1 更平，>1 更尖
        # ---- v3：软蒙 + 动态幅度 ----
        diffae_mask_soft: bool = False,       # True = 蒙区向数据集均值色混合（加噪前），False = 加噪后硬置零
        diffae_mask_alpha: float = 1.0,       # 软蒙混合强度：0=不蒙, 1=完全均值色
        diffae_mask_dynamic: bool = False,    # True = 每张图每步采样比率 r~U[0, ratio_max]；False = 固定 mask_ratio
        diffae_mask_ratio_max: float = 0.3,   # 动态幅度的上限（文献剂量最优区上沿）
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lambda_diff = lambda_diff
        self.cond_dim = cond_dim
        self.auto_ema = auto_ema

        # ---- Encoder 输出压缩：原始维度 -> cond_dim ----
        self.enc_output_dim = enc_output_dim if enc_output_dim is not None else cond_dim
        if self.enc_output_dim != cond_dim:
            self.diffae_enc_compress = nn.Sequential(
                nn.Linear(self.enc_output_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
        else:
            self.diffae_enc_compress = nn.Identity()

        # ---- UNet Decoder (Denoising Network) ----
        self.diffae_unet = ConditionalUNet(
            in_channels=in_channels,
            model_channels=unet_channels,
            out_channels=in_channels,
            channel_mult=unet_channel_mult,
            cond_dim=cond_dim,
            use_zero_module=use_zero_module,
            image_size=image_size,
            num_res_blocks=num_res_blocks,
        )

        # ---- Sampler ----
        use_timesteps = None
        if spaced_timesteps is not None:
            use_timesteps = space_timesteps(num_diffusion_timesteps, spaced_timesteps)

        self.sampler = DDIMSampler(
            num_timesteps=num_diffusion_timesteps,
            beta_schedule=beta_schedule,
            use_timesteps=use_timesteps,
            clip_denoised=True,
            pred_xstart_detach=True,
            custom_timestep_sampler=custom_timestep_sampler,
            predict_x0=diffae_pred_x0,
        )

        # ---- EMA（直接收模块列表，不用 ModuleDict 二次注册） ----
        self.ema = EMAManager([self.diffae_unet, self.diffae_enc_compress], decay=ema_decay)
        self._ema_enabled = True

        # ==================================================================
        # 动态 t 加权状态（buffer：自动随 checkpoint 保存/恢复）
        # ==================================================================
        self.t_weight_enabled = t_weight_enabled
        self.t_weight_warmup_iters = t_weight_warmup_iters
        self.t_weight_eta = t_weight_eta
        self.t_weight_eps = t_weight_eps
        self.t_weight_bins = t_weight_bins
        self.t_weight_interval = t_weight_interval
        self.t_weight_ema = t_weight_ema
        self.t_weight_probe_batch = t_weight_probe_batch
        self.t_weight_static_window = t_weight_static_window
        self.t_weight_joint = t_weight_joint
        self.t_weight_z_cap = t_weight_z_cap
        self.t_weight_domain = t_weight_domain
        self.t_weight_apply = t_weight_apply
        self.t_weight_pos_only = t_weight_pos_only
        self._probe_step = 0
        self._ts_counts = torch.zeros(t_weight_bins)

        # v2 配置
        self.diffae_pred_x0 = diffae_pred_x0
        self.diffae_mask_ratio = diffae_mask_ratio
        self.diffae_mask_patch = diffae_mask_patch
        self.diffae_mask_mode = diffae_mask_mode
        self.diffae_mask_temp = diffae_mask_temp
        self.diffae_mask_soft = diffae_mask_soft
        self.diffae_mask_alpha = diffae_mask_alpha
        self.diffae_mask_dynamic = diffae_mask_dynamic
        self.diffae_mask_ratio_max = diffae_mask_ratio_max

        # 软蒙的目标色：ImageNet 均值在 [-1,1] 空间的取值（恰好等于 padding 的颜色）
        _mu = [(123.675 / 255.0) * 2 - 1, (116.28 / 255.0) * 2 - 1, (103.53 / 255.0) * 2 - 1]
        self.register_buffer('mask_mean', torch.tensor(_mu).view(1, 3, 1, 1))

        # logSNR 分桶：在【训练网格】上按 logSNR 等频划分
        betas_np = get_named_beta_schedule(beta_schedule, num_diffusion_timesteps)
        ac = np.cumprod(1.0 - betas_np)
        logsnr = np.log(ac / (1.0 - ac)).astype(np.float32)          # [T]
        grid = np.array(self.sampler.use_timesteps, dtype=np.int64)  # 模型实际训练过的 t
        grid_ls = logsnr[grid]

        edges = np.quantile(grid_ls, np.linspace(0, 1, t_weight_bins + 1)).astype(np.float32)
        edges[0] -= 1.0
        edges[-1] += 1.0
        centers = 0.5 * (edges[:-1] + edges[1:])
        rep_t = grid[np.array([np.argmin(np.abs(grid_ls - c)) for c in centers])]

        self.register_buffer('logsnr_per_t', torch.from_numpy(logsnr))
        self.register_buffer('bin_edges', torch.from_numpy(edges))
        self.register_buffer('bin_rep_t', torch.from_numpy(rep_t.astype(np.int64)))
        self.register_buffer('d_table', torch.zeros(t_weight_bins))
        self.register_buffer('d_eps_table', torch.zeros(t_weight_bins))  # 诊断：epsilon 空间 D
        self.register_buffer('d_cos_table', torch.zeros(t_weight_bins))  # joint：cos 计分（增益不变）
        self.register_buffer('d_count', torch.zeros(t_weight_bins))
        self.register_buffer('t_weight', torch.full((t_weight_bins,), 1.0 / t_weight_bins))

        # 每个网格点的桶号（t_weight_apply='sample' 时把桶权重归一到网格概率）
        self.register_buffer('grid_bin',
                             self._t_to_bin(torch.from_numpy(grid)).long())

        # 静态窗口模式：权重固定为窗口内桶 1、窗口外 0（与 eps 均匀混合）
        if t_weight_static_window is not None:
            lo, hi = t_weight_static_window
            in_win = (rep_t >= lo) & (rep_t <= hi)
            w = torch.full((t_weight_bins,), t_weight_eps / t_weight_bins)
            w[in_win] = (1.0 - t_weight_eps) / in_win.sum() + t_weight_eps / t_weight_bins
            self.t_weight.copy_(w)

    # ------------------------------------------------------------------
    # 子类必须实现的接口
    # ------------------------------------------------------------------
    def encode_diffae(self, x: torch.Tensor) -> torch.Tensor:
        """语义编码器：将输入图像映射为语义向量 [B, enc_output_dim]。"""
        raise NotImplementedError("子类必须重写 encode_diffae(self, x) 方法。")

    # ------------------------------------------------------------------
    # 梯度热力图 -> 互补蒙块
    # ------------------------------------------------------------------
    def _heatmap_to_mask(self, heatmap: torch.Tensor) -> torch.Tensor:
        """选址蒙块：返回 0/1 keep-mask [B,1,H,W]（1=保留，0=被选中蒙住）。
        - 动态幅度 diffae_mask_dynamic=True：每张图每步独立采 r~U[0, ratio_max]，块数 k_i=round(r*n)；
          False：固定蒙 diffae_mask_ratio。
        - 每张图独立无放回采样（random 模式按均匀随机分数；hot/cold 按热度/Gumbel 分数），
          每行按自己的 k_i 取前 k_i 名（argsort + rank 技巧，支持每行不同 k）。
        软/硬的区别在于 forward_diffae 里的应用方式，本函数只负责选址。
        """
        B = heatmap.shape[0]
        p = self.diffae_mask_patch
        H, W = heatmap.shape[-2:]
        g_h, g_w = max(1, H // p), max(1, W // p)
        n = g_h * g_w

        if self.diffae_mask_dynamic:
            r = torch.rand(B, device=heatmap.device) * self.diffae_mask_ratio_max   # U[0, rmax]
            k = (r * n).round().long().clamp(min=1)                                  # 每图块数
        else:
            k = torch.full((B,), max(1, int(round(self.diffae_mask_ratio * n))),
                           device=heatmap.device, dtype=torch.long)

        if self.diffae_mask_mode == 'random':
            score = torch.rand(B, n, device=heatmap.device)
        else:
            patch = F.adaptive_avg_pool2d(heatmap, (g_h, g_w)).reshape(B, n).clamp_min(0)
            if self.diffae_mask_temp != 1.0:
                patch = patch ** self.diffae_mask_temp
            if self.diffae_mask_mode == 'cold':
                patch = patch.amax(dim=1, keepdim=True) - patch
            u = torch.rand_like(patch).clamp_min(1e-12)
            score = torch.log(patch.clamp_min(1e-12)) + torch.log(-torch.log(u))     # Gumbel

        order = score.argsort(dim=1, descending=True)            # [B, n]
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(n, device=heatmap.device).unsqueeze(0).expand(B, n))
        sel = rank < k.view(B, 1)                                # [B, n] 每图前 k_i 个被选中

        keep = (~sel).float().view(B, 1, g_h, g_w)
        mask = keep.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
        return mask[:, :, :H, :W]

    # ------------------------------------------------------------------
    # t 加权工具
    # ------------------------------------------------------------------
    def _t_to_bin(self, t: torch.Tensor) -> torch.Tensor:
        """t [B] long -> 桶 id [B]"""
        ls = self.logsnr_per_t.to(t.device)[t]
        return torch.bucketize(ls, self.bin_edges.to(t.device)[1:-1].contiguous())

    @torch.no_grad()
    def _tweight_probe(self, x_start: torch.Tensor, cond: torch.Tensor) -> None:
        """
        轮换探针：每个 logSNR 桶取代表 t，在当前训练 batch 上测
        D = median( L(zero) - L(z_sem) )（同图、同噪声、同 t，配对）。
        结果 EMA 进 d_table；预热结束后按 Hedge/EXP3 更新 t_weight。
        """
        if (not self.t_weight_enabled) or (not self.training):
            return
        if self.t_weight_static_window is not None:
            return  # 静态窗口模式：探针只记录不更新
        self._probe_step += 1
        if self._probe_step % self.t_weight_interval != 0:
            return

        B = min(self.t_weight_probe_batch, x_start.shape[0])
        x, c = x_start[:B], cond[:B]
        dev = x.device
        K = self.t_weight_bins

        # UNet 切 eval 关掉 dropout，探针测量更稳（GroupNorm 不受 train/eval 影响）
        was_training = self.diffae_unet.training
        self.diffae_unet.eval()

        d_new = torch.zeros(K, device=dev)
        d_eps_new = torch.zeros(K, device=dev)
        d_cos_new = torch.zeros(K, device=dev)   # cos 计分（joint）
        for bi in range(K):
            t = torch.full((B,), int(self.bin_rep_t[bi]), device=dev, dtype=torch.long)
            noise = torch.randn_like(x)
            x_t = self.sampler.q_sample(x, t, noise)
            p_c = self.diffae_unet(x_t, t, c)
            p_z = self.diffae_unet(x_t, t, torch.zeros_like(c))
            if self.sampler.predict_x0:
                # x0 模式：pred 即 x̂0。同一次前向读出两种 D：
                #   d_x0 = cond 对重建的影响（直接、有界）
                #   d_eps = 换算到 epsilon 空间（固定公式，含 1/√(1-ᾱ) 增益 -> 天然偏置低 t）
                d_x0 = (p_z - p_c).pow(2).flatten(1).mean(1)
                d_new[bi] = d_x0.median()
                ac_b = float(self.sampler.alphas_cumprod[int(self.bin_rep_t[bi])])
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
                d_new[bi] = d_med              # 【修复】非 joint 时 d_table 就是 deps（原行为）
                if self.t_weight_joint:
                    # joint 时 d_table 改用换算到 x̂0 空间的读数（cos 计分的伴生诊断）
                    ac_b = float(self.sampler.alphas_cumprod[int(self.bin_rep_t[bi])])
                    sq_a, sq_1a = ac_b ** 0.5, (1.0 - ac_b) ** 0.5
                    x0_c = (x_t - sq_1a * p_c) / sq_a
                    x0_z = (x_t - sq_1a * p_z) / sq_a
                    d_new[bi] = (x0_z - x0_c).pow(2).flatten(1).mean(1).median()
            if self.t_weight_joint:
                # 1 - cos(ε̂_cond, ε̂_zero)：仿射不变 => 与参数化(ε/x0)无关；
                # 尺度不变 => 免疫两种增益爆炸；语义 = z_sem 对预测方向的相对影响
                a = p_c.flatten(1)
                b = p_z.flatten(1)
                cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)
                d_cos_new[bi] = (1.0 - cos).median()

        if was_training:
            self.diffae_unet.train()

        # 收益 EMA（非平稳折扣近似）
        m = self.t_weight_ema
        self.d_table.mul_(m).add_(d_new, alpha=1.0 - m)
        self.d_eps_table.mul_(m).add_(d_eps_new, alpha=1.0 - m)
        self.d_cos_table.mul_(m).add_(d_cos_new, alpha=1.0 - m)
        self.d_count.add_(1.0)

        # 预热期只累计收益；结束后按指数加权(Hedge/EXP3)更新权重
        if self._probe_step >= self.t_weight_warmup_iters:
            if self.t_weight_joint:
                # cos 计分：增益不变、有界、参数化无关，形状天然中段峰（见 d_cos_table 曲线）
                score = self.d_cos_table
            else:
                score = self.d_table
            K = self.t_weight_bins
            cap = self.t_weight_z_cap

            # 参与竞争的桶 = 定义域内 且 score > 0（正值 = z_sem 对该 t 的去噪有正贡献）。
            # 其余（负贡献、零贡献、域外）一律给 -cap：真抑制（拿探索地板），不靠符号运气。
            ls_bin = self.logsnr_per_t.to(score.device)[self.bin_rep_t]   # 每桶代表 t 的 logSNR
            inside = torch.ones(K, dtype=torch.bool, device=score.device)
            if self.t_weight_domain is not None:
                lo, hi = self.t_weight_domain
                inside = (ls_bin >= lo) & (ls_bin <= hi)
            if self.t_weight_pos_only:
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
                    z = torch.where(pos, self.t_weight_eta * (score - mu) / sd, z)
            else:
                # 【默认，对照实验做法】全局带符号 softmax + z_cap（同已完成那轮一致）
                std = score.std().clamp_min(1e-12)
                z = self.t_weight_eta * (score - score.mean()) / std
            z = z.clamp(-cap, cap)
            w = torch.exp(z - z.max())
            w = w / w.sum()
            self.t_weight.mul_(0.0).add_(
                (1.0 - self.t_weight_eps) * w + self.t_weight_eps / K)

    # ------------------------------------------------------------------
    # 训练接口
    # ------------------------------------------------------------------
    def forward_diffae(self, x_start: torch.Tensor, cond: torch.Tensor,
                       input_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        DiffAE 训练接口。

        Args:
            x_start: [B, C, H, W]，已归一化到 [-1, 1] 的真实图像
            cond:    [B, cond_dim]，已压缩的语义条件（由子类通过 encode_diffae + diffae_enc_compress 得到）
            input_mask: [B,1,H,W] 互补蒙块（1=保留，0=蒙住），由模型侧从 seg 输入梯度热力图生成；
                        None = 不启用（默认，行为不变）。传入时：先蒙 x0 再扩散，
                        并追加蒙区 x0 重构辅助 loss（系数 diffae_mask_aux_weight）。
        Returns:
            dict: {
                'loss_diffae': scalar Tensor（t 加权后）,
                'loss_diffae_per_sample': [B] Tensor（未加权，供 PCGrad batch 模式使用）,
                'z_sem': [B, cond_dim],
                'tw_ent': t_weight 归一化熵（监控用，无 'loss' 字样不会进 loss 求和）,
                'tw_max': t_weight 最大桶权重（监控用）,
            }
        """
        if self.training and self.auto_ema:
            self.ema.step()

        # 探针（每 t_weight_interval 步触发一次，开销 < 训练 1%）
        self._tweight_probe(x_start, cond)

        # ---- 互补蒙块。两种应用方式：
        #   软蒙（v3）：加噪【前】把蒙区向数据集均值色混合（α 控强度），分布连续无跳变；
        #   硬蒙（v2）：加噪【后】乘 0/1 蒙块，蒙区输入为 0。
        #   两者目标都是完整 x_start（x0 模式）/ 噪声（ε 模式）。
        mask = input_mask
        # t_weight 的两种应用：'sample' = 作为 t 的采样分布（每步满强度）；
        # 'loss' = 按桶加权 loss（期望等价，但有限 batch 下强度不均）
        if self.t_weight_enabled and self.t_weight_apply == 'sample':
            probs = self.t_weight.to(x_start.device)[self.grid_bin]
            probs = probs / probs.sum().clamp_min(1e-12)
        else:
            probs = None
        if mask is not None and self.diffae_mask_soft:
            w_keep = 1.0 - self.diffae_mask_alpha * (1.0 - mask)     # 蒙块处 = 1-α
            x_start_in = w_keep * x_start + (1.0 - w_keep) * self.mask_mean.to(x_start.device)
            losses = self.sampler.training_losses(
                model=self.diffae_unet,
                x_start=x_start_in,                # 输入：带均值色块的混合图
                model_kwargs={"cond": cond},
                target_override=x_start,           # 【修复】目标 = 原始干净图（蒙区内容只能靠 z_sem）
                t_probs=probs,
            )
        else:
            losses = self.sampler.training_losses(
                model=self.diffae_unet,
                x_start=x_start,
                model_kwargs={"cond": cond},
                post_mask=mask,
                t_probs=probs,
            )

        loss_per_sample = losses["loss"]  # [B]（未加权的主 epsilon-loss）
        t = losses["t"]                   # [B] long，sampler 已返回
        
        with torch.no_grad():
            self._ts_counts += torch.bincount(
                self._t_to_bin(t).cpu(), minlength=self.t_weight_bins).to(self._ts_counts.device)

        # ---- 主 loss（t 加权，仅 apply='loss' 时启用；'sample' 时均匀聚合）----
        if self.t_weight_enabled and self.t_weight_apply == 'loss':
            w = self.t_weight.to(loss_per_sample.device)[self._t_to_bin(t)]
            loss = (loss_per_sample * w).sum() / w.sum().clamp_min(1e-12)
        else:
            loss = loss_per_sample.mean()


        w = self.t_weight
        tw_ent = -(w * torch.log(w.clamp_min(1e-12))).sum() / np.log(self.t_weight_bins)

        return {
            "loss_diffae": loss * self.lambda_diff,
            "loss_diffae_per_sample": loss_per_sample,
            "z_sem": cond,
            "tw_ent": tw_ent,
            "tw_max": w.max(),
            "zsem_std": cond.std(dim=0).mean().detach(),   # z_sem 存活监控（判据：从 1e-5 回升）
            "ts_mean": t.float().mean().detach(),
        }

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_stochastic(self, x: torch.Tensor) -> torch.Tensor:
        if self._ema_enabled:
            self.ema.apply_shadow()
        try:
            z_sem = self.encode_diffae(x)
            z_sem = self.diffae_enc_compress(z_sem)
            x_T = self.sampler.ddim_reverse_sample_loop(
                model=self.diffae_unet,
                x0=x,
                model_kwargs={"cond": z_sem},
                device=x.device,
            )
        finally:
            if self._ema_enabled:
                self.ema.restore()
        return x_T

    @torch.no_grad()
    def decode(
        self,
        z_sem: torch.Tensor,
        x_T: Optional[torch.Tensor] = None,
        shape: Optional[tuple] = None,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """从语义码解码图像：x_T -> x_0。使用 EMA 参数。"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._ema_enabled:
            self.ema.apply_shadow()
        try:
            if self.enc_output_dim != self.cond_dim:
                z_sem = self.diffae_enc_compress(z_sem)
            if x_T is None:
                assert shape is not None
                x_T = torch.randn(*shape, device=device)
            x_0 = self.sampler.ddim_sample_loop(
                model=self.diffae_unet, noise=x_T,
                model_kwargs={"cond": z_sem}, device=device,
            )
        finally:
            if self._ema_enabled:
                self.ema.restore()
        return x_0

    # ------------------------------------------------------------------
    # EMA / Checkpoint
    # ------------------------------------------------------------------
    def diffae_ema_step(self):
        self.ema.step()

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["diffae_ema_shadow"] = self.ema.state_dict()
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        ema_shadow = state_dict.pop("diffae_ema_shadow", None)
        super().load_state_dict(state_dict, strict=strict)
        if ema_shadow is not None:
            self.ema.load_state_dict(ema_shadow)

    @torch.no_grad()
    def diffae_reconstruct(self, x_start: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        端到端 DiffAE 重建：原图 -> 反演 -> 重建。
        """
        # 1. DDIM 反演: x_0 -> x_T
        x_T = self.encode_stochastic(x_start)

        # 2. 语义编码（不要压缩！让 decode 自己处理）
        z_sem = self.encode_diffae(x_start)

        # 3. EMA 去噪重建: x_T -> x_0
        x_rec = self.decode(z_sem, x_T=x_T, shape=x_start.shape, device=x_start.device)

        # 4. 映射到 [0, 1] 用于可视化
        x_orig = x_start * 0.5 + 0.5
        x_noise = torch.clamp(x_T, -1.0, 1.0) * 0.5 + 0.5
        x_rec = torch.clamp(x_rec, -1.0, 1.0) * 0.5 + 0.5

        return x_orig, x_noise, x_rec

    def save_diffae_visualization(
        self,
        x_start: torch.Tensor,
        save_path: str,
        max_samples: int = 4,
    ):
        """
        保存 [原图 | 反演噪声 | 重建图] 对比。
        每行一个样本，三列横向排列。
        """
        import os
        from PIL import Image
        import numpy as np

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        x_orig, x_noise, x_rec = self.diffae_reconstruct(x_start[:max_samples])
        B = x_orig.shape[0]

        # 第四列：只用 z_sem、从【所有样本共享的同一份固定噪声】生成。
        # 样本间输出差异的唯一来源是 z_sem —— "z_sem 携带了什么"的直接可视化
        # （对 mask 复活故事是决定性的：全糊 = 没携带；能认出对应图 = 携带了）。
        with torch.no_grad():
            z_all = self.encode_diffae(x_start[:max_samples])  # 原始 2048 维；decode 内部自行压缩
            x_T_shared = torch.randn(1, *x_start.shape[1:], device=x_start.device) \
                .expand(B, -1, -1, -1)
            x_gen = self.decode(z_all, x_T=x_T_shared,
                                shape=tuple(x_start.shape[1:]), device=x_start.device)
            x_gen = torch.clamp(x_gen, -1.0, 1.0) * 0.5 + 0.5

        def _to_pil(x):
            arr = x.permute(1, 2, 0).cpu().numpy()
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            return Image.fromarray(arr)

        rows = []
        for i in range(B):
            orig = _to_pil(x_orig[i])
            noise = _to_pil(x_noise[i])
            rec = _to_pil(x_rec[i])
            gen = _to_pil(x_gen[i])
            # 横向拼：原图 | 反演 x_T | 重建 | 仅 z_sem 生成（共享噪声）
            row = Image.new('RGB', (orig.width * 4, orig.height))
            row.paste(orig, (0, 0))
            row.paste(noise, (orig.width, 0))
            row.paste(rec, (orig.width * 2, 0))
            row.paste(gen, (orig.width * 3, 0))
            rows.append(row)

        # 纵向拼所有样本
        total_h = sum(r.height for r in rows)
        grid = Image.new('RGB', (rows[0].width, total_h))
        y = 0
        for r in rows:
            grid.paste(r, (0, y))
            y += r.height

        grid.save(save_path)