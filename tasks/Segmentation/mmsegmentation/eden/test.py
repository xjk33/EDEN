"""
DiffAE Mixin + EMA Manager + 动态 t 加权（discounted-EXP3 / Hedge）

设计原则：
  - 只提供扩散预训练的"能力"和"工具函数"。
  - 不假设任何下游框架的 forward / train_step 签名。

v2：x0 预测 + 空间蒙块（单次 loss 全包，无辅助 loss）
  - diffae_pred_x0=True 时 UNet 直接预测 x0：可见区=去噪，蒙区=内容重构（唯一信息源 z_sem）
  - 蒙块模式：'random' 均匀 | 'hot' 蒙 seg 高敏感区 | 'cold' 蒙低敏感区（后两者需 seg 热力图）
  - 软蒙 diffae_mask_soft：加噪前把蒙区向数据集均值色混合；硬蒙：加噪后乘 0/1 蒙块
  - 动态幅度 diffae_mask_dynamic：每图每步 r~U[0, ratio_max]；False = 固定 mask_ratio
t 加权（Hedge/EXP3）：
  - 探针每 t_weight_interval 步测 per-桶 D（同图同噪声同 t，中位数，EMA 进 d_table）
  - 更新：定义域内且 score>0 的桶参加 softmax 竞争，其余（负贡献/零贡献/域外）给 -z_cap 地板
  - 定义域 t_weight_domain=[lo,hi]（logSNR）：域外恒地板（SODA 先验截两端）
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
        t_weight_static_window: Optional[List[int]] = None,
        t_weight_joint: bool = False,   # True = cos 计分（1-cos）
        t_weight_z_cap: float = 1.0,    # 标准化分数截断
        t_weight_domain: Optional[List[float]] = None,  # logSNR 定义域 [lo,hi]；域外恒地板
        # ---- v2/v3：x0 预测 + 空间蒙块 ----
        diffae_pred_x0: bool = False,
        diffae_mask_ratio: float = 0.0,
        diffae_mask_patch: int = 16,
        diffae_mask_mode: str = "random",
        diffae_mask_temp: float = 1.0,
        diffae_mask_soft: bool = False,       # True = 加噪前向均值色混合；False = 加噪后硬置零
        diffae_mask_alpha: float = 1.0,       # 软蒙混合强度
        diffae_mask_dynamic: bool = False,    # True = 每图每步 r~U[0, ratio_max]
        diffae_mask_ratio_max: float = 0.3,
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
        self._probe_step = 0

        # v2/v3 配置
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
        logsnr = np.log(ac / (1.0 - ac)).astype(np.float32)
        grid = np.array(self.sampler.use_timesteps, dtype=np.int64)
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
        self.register_buffer('d_eps_table', torch.zeros(t_weight_bins))
        self.register_buffer('d_cos_table', torch.zeros(t_weight_bins))
        self.register_buffer('d_count', torch.zeros(t_weight_bins))
        self.register_buffer('t_weight', torch.full((t_weight_bins,), 1.0 / t_weight_bins))

        # 静态窗口模式
        if t_weight_static_window is not None:
            lo, hi = t_weight_static_window
            in_win = (rep_t >= lo) & (rep_t <= hi)
            w = torch.full((t_weight_bins,), t_weight_eps / t_weight_bins)
            w[in_win] = (1.0 - t_weight_eps) / in_win.sum() + t_weight_eps / t_weight_bins
            self.t_weight.copy_(w)

    def encode_diffae(self, x: torch.Tensor) -> torch.Tensor:
        """语义编码器：将输入图像映射为语义向量 [B, enc_output_dim]。"""
        raise NotImplementedError("子类必须重写 encode_diffae(self, x) 方法。")

    # ------------------------------------------------------------------
    # 梯度热力图 -> 蒙块选址（0/1 keep-mask；软/硬在 forward_diffae 里区分）
    # ------------------------------------------------------------------
    def _heatmap_to_mask(self, heatmap: torch.Tensor) -> torch.Tensor:
        """选址蒙块：返回 0/1 keep-mask [B,1,H,W]（1=保留，0=被选中蒙住）。
        - 动态幅度 dynamic=True：每图每步 r~U[0, ratio_max]，k_i=round(r*n)；False：固定 mask_ratio。
        - 每图独立无放回采样（argsort + rank 技巧，支持每行不同 k）。
        """
        B = heatmap.shape[0]
        p = self.diffae_mask_patch
        H, W = heatmap.shape[-2:]
        g_h, g_w = max(1, H // p), max(1, W // p)
        n = g_h * g_w

        if self.diffae_mask_dynamic:
            r = torch.rand(B, device=heatmap.device) * self.diffae_mask_ratio_max
            k = (r * n).round().long().clamp(min=1)
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
            score = torch.log(patch.clamp_min(1e-12)) + torch.log(-torch.log(u))

        order = score.argsort(dim=1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(n, device=heatmap.device).unsqueeze(0).expand(B, n))
        sel = rank < k.view(B, 1)

        keep = (~sel).float().view(B, 1, g_h, g_w)
        mask = keep.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
        return mask[:, :, :H, :W]

    # ------------------------------------------------------------------
    # t 加权工具
    # ------------------------------------------------------------------
    def _t_to_bin(self, t: torch.Tensor) -> torch.Tensor:
        ls = self.logsnr_per_t.to(t.device)[t]
        return torch.bucketize(ls, self.bin_edges.to(t.device)[1:-1].contiguous())

    @torch.no_grad()
    def _tweight_probe(self, x_start: torch.Tensor, cond: torch.Tensor) -> None:
        """轮换探针：每桶测 D（同图同噪声同 t，中位数），EMA 进表；
        预热后按"正值竞争 + 域外地板"更新 t_weight。"""
        if (not self.t_weight_enabled) or (not self.training):
            return
        if self.t_weight_static_window is not None:
            return
        self._probe_step += 1
        if self._probe_step % self.t_weight_interval != 0:
            return

        B = min(self.t_weight_probe_batch, x_start.shape[0])
        x, c = x_start[:B], cond[:B]
        dev = x.device
        K = self.t_weight_bins

        was_training = self.diffae_unet.training
        self.diffae_unet.eval()

        d_new = torch.zeros(K, device=dev)
        d_eps_new = torch.zeros(K, device=dev)
        d_cos_new = torch.zeros(K, device=dev)
        for bi in range(K):
            t = torch.full((B,), int(self.bin_rep_t[bi]), device=dev, dtype=torch.long)
            noise = torch.randn_like(x)
            x_t = self.sampler.q_sample(x, t, noise)
            p_c = self.diffae_unet(x_t, t, c)
            p_z = self.diffae_unet(x_t, t, torch.zeros_like(c))
            if self.sampler.predict_x0:
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
                d_med = (l_z - l_c).median()
                d_eps_new[bi] = d_med
                d_new[bi] = d_med              # 非 joint：d_table = deps（原行为）
                if self.t_weight_joint:
                    ac_b = float(self.sampler.alphas_cumprod[int(self.bin_rep_t[bi])])
                    sq_a, sq_1a = ac_b ** 0.5, (1.0 - ac_b) ** 0.5
                    x0_c = (x_t - sq_1a * p_c) / sq_a
                    x0_z = (x_t - sq_1a * p_z) / sq_a
                    d_new[bi] = (x0_z - x0_c).pow(2).flatten(1).mean(1).median()
            if self.t_weight_joint:
                a = p_c.flatten(1)
                b = p_z.flatten(1)
                cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)
                d_cos_new[bi] = (1.0 - cos).median()

        if was_training:
            self.diffae_unet.train()

        m = self.t_weight_ema
        self.d_table.mul_(m).add_(d_new, alpha=1.0 - m)
        self.d_eps_table.mul_(m).add_(d_eps_new, alpha=1.0 - m)
        self.d_cos_table.mul_(m).add_(d_cos_new, alpha=1.0 - m)
        self.d_count.add_(1.0)

        # 更新：正值竞争 + 域外地板
        if self._probe_step >= self.t_weight_warmup_iters:
            score = self.d_cos_table if self.t_weight_joint else self.d_table
            cap = self.t_weight_z_cap

            ls_bin = self.logsnr_per_t.to(score.device)[self.bin_rep_t]
            inside = torch.ones(K, dtype=torch.bool, device=score.device)
            if self.t_weight_domain is not None:
                lo, hi = self.t_weight_domain
                inside = (ls_bin >= lo) & (ls_bin <= hi)
            pos = inside & (score > 0)

            if not pos.any():
                if not getattr(self, '_warned_flat', False):
                    self._warned_flat = True
                    print('[t_weight] WARNING: no positive bin (all D<=0 or outside domain), '
                          'weights stay at floor.')
                z = torch.zeros(K, device=score.device)
            else:
                mu = score[pos].mean()
                sd = score[pos].std().clamp_min(1e-12)
                z = torch.full_like(score, -cap)
                z = torch.where(pos, self.t_weight_eta * (score - mu) / sd, z)
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
        if self.training and self.auto_ema:
            self.ema.step()

        self._tweight_probe(x_start, cond)

        # 互补蒙块：软蒙 = 加噪前向均值色混合；硬蒙 = 加噪后乘 0/1 蒙块
        mask = input_mask
        if mask is not None and self.diffae_mask_soft:
            w_keep = 1.0 - self.diffae_mask_alpha * (1.0 - mask)
            x_start_in = w_keep * x_start + (1.0 - w_keep) * self.mask_mean.to(x_start.device)
            losses = self.sampler.training_losses(
                model=self.diffae_unet,
                x_start=x_start_in,
                model_kwargs={"cond": cond},
            )
        else:
            losses = self.sampler.training_losses(
                model=self.diffae_unet,
                x_start=x_start,
                model_kwargs={"cond": cond},
                post_mask=mask,
            )

        loss_per_sample = losses["loss"]
        t = losses["t"]

        if self.t_weight_enabled:
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
            "zsem_std": cond.std(dim=0).mean().detach(),
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
        x_T = self.encode_stochastic(x_start)
        z_sem = self.encode_diffae(x_start)
        x_rec = self.decode(z_sem, x_T=x_T, shape=x_start.shape, device=x_start.device)
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
        import os
        from PIL import Image
        import numpy as np

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        x_orig, x_noise, x_rec = self.diffae_reconstruct(x_start[:max_samples])
        B = x_orig.shape[0]

        def _to_pil(x):
            arr = x.permute(1, 2, 0).cpu().numpy()
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            return Image.fromarray(arr)

        rows = []
        for i in range(B):
            orig = _to_pil(x_orig[i])
            noise = _to_pil(x_noise[i])
            rec = _to_pil(x_rec[i])
            row = Image.new('RGB', (orig.width * 3, orig.height))
            row.paste(orig, (0, 0))
            row.paste(noise, (orig.width, 0))
            row.paste(rec, (orig.width * 2, 0))
            rows.append(row)

        total_h = sum(r.height for r in rows)
        grid = Image.new('RGB', (rows[0].width, total_h))
        y = 0
        for r in rows:
            grid.paste(r, (0, y))
            y += r.height
        grid.save(save_path)