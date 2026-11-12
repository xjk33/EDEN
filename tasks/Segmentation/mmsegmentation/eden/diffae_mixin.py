"""
DiffAE Mixin + EMA Manager

设计原则：
  - 只提供扩散预训练的"能力"和"工具函数"。
  - 不假设任何下游框架的 forward / train_step 签名。
  - t 自适应（Dt 探针 + 权重调度）与结构化加噪（选址 + 施加）分别独立在
    t_adaptive.py 和 masking.py，本文件只做组合。

x0 预测 + 空间蒙块（单次 loss 全包，无辅助 loss）：
  - diffae_pred_x0=True 时 UNet 直接预测 x0：可见区=去噪，蒙区=内容重构（唯一信息源 z_sem）
  - 蒙块选址见 masking.MaskSelector；软/硬施加见 masking.MaskApplier
  - loss 全程在图像空间（x0 模式），任何 t 有界

对外暴露的训练接口只有 forward_diffae；监控钩子需要的内部状态通过
本类的委托属性（t_weight / d_table / bin_rep_t / _ts_counts / _heatmap_to_mask）
访问，实现细节均在子模块里。
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List

from .sampler import DDIMSampler, space_timesteps
from .unet import ConditionalUNet
from .vit_denoiser import ConditionalDiT
from .t_adaptive import TAdaptive
from .masking import MaskSelector, MaskApplier
from .beta_sampler import ANTSampler


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
        # ---- 去噪器选择：'dit'（弱仪器，i-DAE 路线，默认）| 'unet'（强仪器）----
        # 注意：属性名保持 diffae_unet 不变（EMA / paramwise custom_keys / 探针 notebook 均引用该名）
        denoiser_type: str = 'dit',
        dit_size: str = 'ti',             # 'ti'(~6.7M) | 's'(~25.8M) | 'b'(~100M)；表征已解耦到 encoder，仪器只做小
        dit_dim: Optional[int] = None,    # 显式给定则覆盖预设（逐项）
        dit_depth: Optional[int] = None,
        dit_heads: Optional[int] = None,
        dit_patch: int = 16,              # i-DAE：patch 16 直接吃像素
        dit_mlp_ratio: float = 4.0,
        adaln_hidden_mult: float = 0.25,  # i-DAE：条件 MLP 隐层 = 1/4
        # ---- t 自适应（实现见 t_adaptive.TAdaptive，参数原样透传）----
        t_weight_enabled: bool = False,
        t_weight_warmup_iters: int = 5000,
        t_weight_eta: float = 2.0,
        t_weight_eps: float = 0.1,
        t_weight_bins: int = 16,
        t_weight_interval: int = 500,
        t_weight_ema: float = 0.95,
        t_weight_probe_batch: int = 16,
        t_weight_static_window: Optional[List[int]] = None,  # 如 [100,500]，设置后禁用动态更新
        t_weight_joint: bool = False,     # True = cos 计分：1-cos(ε̂_cond, ε̂_zero)
        t_weight_z_cap: float = 1.0,      # 标准化分数截断（防单桶权重劫持）
        t_weight_domain: Optional[List[float]] = None,  # logSNR 定义域 [lo, hi]
        t_weight_apply: str = 'loss',     # 'loss' = 加权 loss；'sample' = 作为 t 的采样分布
        t_weight_pos_only: bool = False,  # True = 正值竞争 + domain
        # ---- 结构化加噪（选址见 masking.MaskSelector，施加见 masking.MaskApplier）----
        diffae_pred_x0: bool = False,     # True = 网络直接预测 x0（可见区去噪 + 蒙区靠 z_sem 重构）
        diffae_mask_ratio: float = 0.0,   # 蒙块比例 0~1，0 = 关闭
        diffae_mask_patch: int = 16,      # 蒙块边长（像素）
        diffae_mask_mode: str = "random", # 'random' | 'hot' | 'cold'（后两者需下游热力图）
        diffae_mask_temp: float = 1.0,    # 热力采样温度：<1 更平，>1 更尖
        diffae_mask_soft: bool = False,   # True = 软蒙（加噪前向均值色混合）；False = 硬蒙（加噪后置零）
        diffae_mask_alpha: float = 1.0,   # 软蒙混合强度：0=不蒙, 1=完全均值色
        diffae_mask_dynamic: bool = False,# True = 每图每步 r~U[0, ratio_max]；False = 固定 mask_ratio
        diffae_mask_ratio_max: float = 0.3,
        # ---- ANTS：可学习 Beta t 采样器（实现见 beta_sampler.py，与 t_weight 互斥）----
        ants_enabled: bool = False,
        ants_lr: float = 1e-3,                 # 采样器 Adam lr（原文 1e-3）
        ants_entropy_coef: float = 1e-2,       # 熵正则（原文 1e-2）
        ants_interval: int = 40,               # reward 测量间隔 f_s（原文 40）
        ants_queue: int = 20,                  # reward 队列长度 |Q|（原文 20）
        ants_rep_timesteps: Tuple[int, ...] = (200, 500, 800),  # 代表时间步集 S（简化：固定 3 点）
        ants_hidden: int = 256,
        # ---- Khungurn 两阶段（arXiv:2504.21368，训练侧 t 课程；与 ants/t_weight/窗口互斥）----
        khun_enabled: bool = False,
        khun_phase1_iters: int = 20000,   # 阶段一长度（原文：总长的 1/4，未调优；80k -> 20k）
        khun_phase1_t: int = 990,         # 阶段一恒定 t（须为网格点；ddim100 最大点 990 ≈ 原文 t=1 纯噪声）
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

        # ---- Denoising Network：'dit'（弱仪器，默认）或 'unet' ----
        if denoiser_type == 'dit':
            self.diffae_unet = ConditionalDiT(
                in_channels=in_channels,
                image_size=image_size,
                patch_size=dit_patch,
                size=dit_size,
                dim=dit_dim,
                depth=dit_depth,
                num_heads=dit_heads,
                mlp_ratio=dit_mlp_ratio,
                cond_dim=cond_dim,
                adaln_hidden_mult=adaln_hidden_mult,
            )
        elif denoiser_type == 'unet':
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
        else:
            raise ValueError(f"unknown denoiser_type: {denoiser_type}")

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

        # ---- t 自适应：Dt 探针 + 权重调度（实现见 t_adaptive.py）----
        self.t_adaptive = TAdaptive(
            enabled=t_weight_enabled,
            num_timesteps=num_diffusion_timesteps,
            beta_schedule=beta_schedule,
            use_timesteps=self.sampler.use_timesteps,
            bins=t_weight_bins,
            interval=t_weight_interval,
            ema=t_weight_ema,
            probe_batch=t_weight_probe_batch,
            warmup_iters=t_weight_warmup_iters,
            eta=t_weight_eta,
            eps=t_weight_eps,
            z_cap=t_weight_z_cap,
            domain=t_weight_domain,
            pos_only=t_weight_pos_only,
            joint=t_weight_joint,
            apply=t_weight_apply,
            static_window=t_weight_static_window,
        )

        # ---- 结构化加噪：选址 + 施加（实现见 masking.py）----
        self.mask_selector = MaskSelector(
            patch=diffae_mask_patch,
            mode=diffae_mask_mode,
            temp=diffae_mask_temp,
            ratio=diffae_mask_ratio,
            dynamic=diffae_mask_dynamic,
            ratio_max=diffae_mask_ratio_max,
        )
        self.mask_applier = MaskApplier(soft=diffae_mask_soft, alpha=diffae_mask_alpha)

        # ---- ANTS：可学习 Beta t 采样器（Kim et al. CVPR 2025 简化复现）----
        # 与 t_weight 互斥（两者都决定 t 的采样分布）；与 custom_timestep_sampler
        # 也互斥（t_override 优先级最高，窗口会被静默忽略——这里显式拦住）。
        self.ants_enabled = ants_enabled
        self._ants_pending = None           # reward 簿记（去噪器 step 后由 train_step 消费）
        self._ants_monitor = {}             # 最近一次 REINFORCE 的监控读数（一步延迟）
        if ants_enabled:
            assert not t_weight_enabled, \
                'ANTS 与 t_weight 互斥：两者都决定 t 的采样分布'
            assert custom_timestep_sampler is None, \
                'ANTS 与 custom_timestep_sampler 互斥（t_override 会覆盖窗口采样）'
            self.ants_sampler = ANTSampler(
                in_channels=in_channels, hidden=ants_hidden,
                num_timesteps=num_diffusion_timesteps,
                use_timesteps=self.sampler.use_timesteps,
                lr=ants_lr, entropy_coef=ants_entropy_coef,
                interval=ants_interval, queue_size=ants_queue,
                rep_timesteps=ants_rep_timesteps)
            self._ants_step = 0

        # ---- Khungurn 两阶段：训练侧 t 课程（arXiv:2504.21368 核心机制复现）----
        # 阶段一（前 khun_phase1_iters 步）恒 t=khun_phase1_t：输入近似纯噪声，
        # 去噪器从输入拿不到任何信息，被迫把结构信息压进 z_sem（≈ vanilla AE 状态），
        # 这是原文"强迫 z 携带结构信息"的机制。阶段二恢复随机采样，但密度偏低噪区
        # （t = T_max·(1-√u)，u~U(0,1)），对应原文 Sc-4 schedule"更多时间花在
        # 低噪区"的效果（补细节）。原文还验证了 z 的线性探针精度在两阶段下几乎不掉。
        # 有意偏差（写入注释以备查）：为与其他臂保持单变量可比，维持 EDEN 的
        # 线性-β schedule + x0 预测不变，不引入原文的 Sc-4 噪声 schedule 与
        # v-prediction——只复现"两阶段课程"这一核心机制（原文 Table 1 显示
        # x-pred 在低噪 schedule 下同样可行）。
        self.khun_enabled = khun_enabled
        self.khun_phase1_iters = khun_phase1_iters
        self.khun_phase1_t = khun_phase1_t
        if khun_enabled:
            assert not ants_enabled, \
                'Khungurn 与 ANTS 互斥：两者都独占 t_override 通道'
            assert not t_weight_enabled, \
                'Khungurn 与 t_weight 互斥：两者都决定 t 的采样分布'
            assert custom_timestep_sampler is None, \
                'Khungurn 与 custom_timestep_sampler 互斥（t_override 会覆盖窗口采样）'
            assert khun_phase1_t in self.sampler.use_timesteps, \
                f'khun_phase1_t={khun_phase1_t} 不在采样网格上（如 ddim100: 0,10,...,990）'
            self._khun_step = 0
            tm = self.sampler.timestep_map
            self._khun_grid_step = tm[1] - tm[0] if len(tm) > 1 else 1

        # 配置镜像（adapter / 监控钩子读取用；实现细节都在子模块里）
        self.diffae_pred_x0 = diffae_pred_x0
        self.diffae_mask_ratio = diffae_mask_ratio
        self.diffae_mask_patch = diffae_mask_patch
        self.diffae_mask_mode = diffae_mask_mode
        self.diffae_mask_temp = diffae_mask_temp
        self.diffae_mask_soft = diffae_mask_soft
        self.diffae_mask_alpha = diffae_mask_alpha
        self.diffae_mask_dynamic = diffae_mask_dynamic
        self.diffae_mask_ratio_max = diffae_mask_ratio_max

    # ------------------------------------------------------------------
    # 子类必须实现的接口
    # ------------------------------------------------------------------
    def encode_diffae(self, x: torch.Tensor) -> torch.Tensor:
        """语义编码器：将输入图像映射为语义向量 [B, enc_output_dim]。"""
        raise NotImplementedError("子类必须重写 encode_diffae(self, x) 方法。")

    # ------------------------------------------------------------------
    # 委托属性：监控钩子 / adapter 的访问路径（实现已拆到子模块）
    # ------------------------------------------------------------------
    @property
    def t_weight(self) -> torch.Tensor:
        return self.t_adaptive.scheduler.t_weight

    @property
    def d_table(self) -> torch.Tensor:
        return self.t_adaptive.probe.d_table

    @property
    def bin_rep_t(self) -> torch.Tensor:
        return self.t_adaptive.probe.bin_rep_t

    @property
    def _ts_counts(self) -> torch.Tensor:
        return self.t_adaptive._ts_counts

    def _heatmap_to_mask(self, heatmap: torch.Tensor) -> torch.Tensor:
        """委托给 mask_selector.select（保留旧接口名）。"""
        return self.mask_selector.select(heatmap)

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
            input_mask: [B,1,H,W] 互补蒙块（1=保留，0=蒙住），由子类从下游任务敏感度
                        热力图经 mask_selector 生成；None = 不启用。施加方式（软/硬）
                        由 mask_applier 决定，loss 目标始终是完整 x0/ε，
                        蒙区内容的唯一信息源是 cond（z_sem）。
        Returns:
            dict: {
                'loss_diffae': scalar Tensor（t 加权后）,
                'z_sem': [B, cond_dim],
                'tw_ent': t_weight 归一化熵（监控用，无 'loss' 字样不会进 loss 求和）,
                'tw_max': t_weight 最大桶权重（监控用）,
                'zsem_std': z_sem 跨图 std（存活监控）,
                'ts_mean': 本步采样 t 均值（监控用）,
            }
        """
        if self.training and self.auto_ema:
            self.ema.step()

        # t 自适应：探针测 Dt -> 调度器更新 wt ->（apply='sample' 时）给出 t 采样分布
        # （探针开销 < 训练 1%，且始终用未蒙块的原始 x_start 测量）
        t_probs = self.t_adaptive.on_train_step(
            self.diffae_unet, self.sampler, x_start, cond, training=self.training)

        # ANTS：每步从 Beta(a,b) 采 t（覆盖内部采样）；到 interval 时先测
        # 更新前 reward 基线（同批图、同 cond、固定代表 t、CRN 噪声配对），
        # 簿记挂到 self._ants_pending，等去噪器 step 后由 train_step 消费。
        t_override = None
        if self.ants_enabled and self.training:
            t_override, logp, ent, ab = self.ants_sampler.sample(x_start)
            self._ants_step += 1
            if self._ants_step % self.ants_sampler.interval == 0:
                before, noises = self._ants_measure(x_start, cond)
                self._ants_pending = dict(
                    x=x_start.detach(), cond=cond.detach(),
                    logp=logp, ent=ent, ab=ab, before=before, noises=noises)

        # Khungurn 两阶段：阶段一恒 t=khun_phase1_t（纯噪声课程，逼 z_sem 扛结构）；
        # 阶段二偏低噪密度 t = T_max·(1-√u)，吸附到采样网格后与 t_override 通道对接。
        khun_phase = 0.0
        if self.khun_enabled and self.training:
            self._khun_step += 1
            if self._khun_step <= self.khun_phase1_iters:
                khun_phase = 1.0
                t_override = torch.full(
                    (x_start.shape[0],), self.khun_phase1_t,
                    device=x_start.device, dtype=torch.long)
            else:
                khun_phase = 2.0
                u = torch.rand(x_start.shape[0], device=x_start.device)
                t_f = self.khun_phase1_t * (1.0 - u.sqrt())
                step = self._khun_grid_step
                t_override = (torch.round(t_f / step) * step).long().clamp_(
                    0, self.khun_phase1_t)

        # 结构化加噪：MaskApplier 把蒙块翻译成 sampler 的输入（软蒙/硬蒙/无蒙）
        loss_kwargs = self.mask_applier.transform(x_start, input_mask)
        losses = self.sampler.training_losses(
            model=self.diffae_unet,
            model_kwargs={"cond": cond},
            t_probs=t_probs,
            t_override=t_override,
            **loss_kwargs,
        )

        loss_per_sample = losses["loss"]  # [B]（未加权的主 loss）
        t = losses["t"]                   # [B] long，sampler 已返回

        self.t_adaptive.record_samples(t)

        # ---- 主 loss 聚合（t 加权仅 apply='loss' 时启用；'sample' 时均匀聚合）----
        loss = self.t_adaptive.weight_loss(loss_per_sample, t)

        w = self.t_adaptive.scheduler.t_weight
        tw_ent = -(w * torch.log(w.clamp_min(1e-12))).sum() / np.log(self.t_adaptive.probe.num_bins)

        out = {
            "loss_diffae": loss * self.lambda_diff,
            "z_sem": cond,
            "tw_ent": tw_ent,
            "tw_max": w.max(),
            "zsem_std": cond.std(dim=0).mean().detach(),   # z_sem 存活监控（判据：从 1e-5 回升）
            "ts_mean": t.float().mean().detach(),
        }
        if self.ants_enabled:
            # ANTS 监控：上一次 REINFORCE 的读数（一步延迟，键名无 'loss' 不求和）
            out.update(self._ants_monitor)
        if self.khun_enabled:
            # Khungurn 监控：当前所处阶段（1=纯噪声课程，2=偏低噪采样）
            out['khun_phase'] = torch.tensor(khun_phase)
        return out

    # ------------------------------------------------------------------
    # ANTS：reward 测量与 REINFORCE 更新（实现语义见 beta_sampler.py 头注释）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ants_measure(self, x: torch.Tensor, cond: torch.Tensor,
                      noises: Optional[List[torch.Tensor]] = None):
        """在代表时间步集 S 上测 per-sample 扩散 MSE；noises 给定则复用（CRN 配对）。
        与 DtProbe 同一约定：始终用未蒙块的原始 x_start 测量。
        Returns: (loss [B, |S|], noises list)"""
        ls, ns = [], []
        for i, tau in enumerate(self.ants_sampler.rep_timesteps):
            t = torch.full((x.shape[0],), int(tau), device=x.device, dtype=torch.long)
            n = torch.randn_like(x) if noises is None else noises[i]
            x_t = self.sampler.q_sample(x, t, n)
            pred = self.diffae_unet(x_t, t, cond)
            ls.append((pred - x).pow(2).flatten(1).mean(1))
            ns.append(n)
        return torch.stack(ls, dim=1), ns

    def ants_after_update(self) -> None:
        """
        去噪器优化器 step 之后由 train_step 调用（仅在有 pending 簿记时生效）：
        测更新后 loss -> reward = L_before - L_after -> REINFORCE 更新采样器。
        """
        p = self._ants_pending
        if p is None:
            return
        self._ants_pending = None
        after, _ = self._ants_measure(p['x'], p['cond'], noises=p['noises'])
        reward = float((p['before'] - after).mean())
        self.ants_sampler.reinforce_update(p['logp'], p['ent'], reward)
        a, b = p['ab']
        w = self.ants_sampler.reward_queue
        self._ants_monitor = {
            'ants_a': a.mean().detach(),
            'ants_b': b.mean().detach(),
            'ants_reward': torch.tensor(sum(w) / len(w)),
            'ants_ent': p['ent'].mean().detach(),
        }

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_stochastic(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """DDIM 反演 x_0 -> x_T。cond（压缩后）可外部传入；None 时走 encode_diffae 现算。"""
        if self._ema_enabled:
            self.ema.apply_shadow()
        try:
            if cond is None:
                cond = self.diffae_enc_compress(self.encode_diffae(x))
            x_T = self.sampler.ddim_reverse_sample_loop(
                model=self.diffae_unet,
                x0=x,
                model_kwargs={"cond": cond},
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
    def diffae_reconstruct(
        self,
        x_start: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        端到端 DiffAE 重建：原图 -> 反演 -> 重建。
        cond（压缩后的语义条件）可外部传入——监控/离线探针应传从【训练同分布】
        路径（如归一化大图过 backbone）算出的 cond；None 时退回 encode_diffae(x_start)。
        """
        if cond is None:
            cond = self.diffae_enc_compress(self.encode_diffae(x_start))

        if self._ema_enabled:
            self.ema.apply_shadow()
        try:
            # 1. DDIM 反演: x_0 -> x_T
            x_T = self.sampler.ddim_reverse_sample_loop(
                model=self.diffae_unet, x0=x_start,
                model_kwargs={"cond": cond}, device=x_start.device,
            )
            # 2. EMA 去噪重建: x_T -> x_0
            x_rec = self.sampler.ddim_sample_loop(
                model=self.diffae_unet, noise=x_T,
                model_kwargs={"cond": cond}, device=x_start.device,
            )
        finally:
            if self._ema_enabled:
                self.ema.restore()

        # 3. 映射到 [0, 1] 用于可视化
        x_orig = x_start * 0.5 + 0.5
        x_noise = torch.clamp(x_T, -1.0, 1.0) * 0.5 + 0.5
        x_rec = torch.clamp(x_rec, -1.0, 1.0) * 0.5 + 0.5

        return x_orig, x_noise, x_rec

    def save_diffae_visualization(
        self,
        x_start: torch.Tensor,
        save_path: str,
        max_samples: int = 4,
        cond: Optional[torch.Tensor] = None,
    ):
        """
        保存 [原图 | 反演噪声 | 重建图 | 仅 z_sem 生成] 对比。
        每行一个样本，四列横向排列。cond 语义同 diffae_reconstruct。
        """
        import os
        from PIL import Image
        import numpy as np

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        x_start = x_start[:max_samples]
        if cond is None:
            cond = self.diffae_enc_compress(self.encode_diffae(x_start))
        else:
            cond = cond[:max_samples]

        x_orig, x_noise, x_rec = self.diffae_reconstruct(x_start, cond=cond)
        B = x_orig.shape[0]

        # 第四列：只用 z_sem、从【所有样本共享的同一份固定噪声】生成。
        # 样本间输出差异的唯一来源是 z_sem —— "z_sem 携带了什么"的直接可视化
        # （对 mask 复活故事是决定性的：全糊 = 没携带；能认出对应图 = 携带了）。
        with torch.no_grad():
            if self._ema_enabled:
                self.ema.apply_shadow()
            try:
                x_T_shared = torch.randn(1, *x_start.shape[1:], device=x_start.device) \
                    .expand(B, -1, -1, -1)
                x_gen = self.sampler.ddim_sample_loop(
                    model=self.diffae_unet, noise=x_T_shared,
                    model_kwargs={"cond": cond}, device=x_start.device,
                )
            finally:
                if self._ema_enabled:
                    self.ema.restore()
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
