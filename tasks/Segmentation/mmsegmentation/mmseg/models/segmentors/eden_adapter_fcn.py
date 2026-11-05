"""
mmseg 适配层：把 eden 的 DiffAEMixin 挂到 EncoderDecoder 上（FCN 语义分割）。

唯一注册类 FCNWithDiffAE：
  - backbone 单次前向，seg loss 与 DiffAE 辅助 loss 复用同一份 feats；
  - 结构化噪声选址（mask_selector）需要 seg loss 对输入的梯度热力图，
    用 autograd.grad 只取输入梯度、不写参数梯度；
  - 训练期零监控：分析全部离线（visual/probe_*.ipynb），tw_*/zsem_std 等
    标量挂在 losses 里随训练日志记录（键名不含 'loss'，不参与求和）。

包名说明：部署到 mmsegmentation 时 eden 包以 mydiffae 为名挂在 PYTHONPATH 上
（各 config 的 custom_imports 也是 mydiffae.*），本地开发回退到 eden。
"""
from typing import List, Optional

import torch
import torch.nn.functional as F
from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS

try:
    from mydiffae.diffae_mixin import DiffAEMixin
except ImportError:  # 本地仓库内开发（包名 eden）
    from eden.diffae_mixin import DiffAEMixin


def _add_monitor_keys(losses, diff_out):
    """把监控标量挂到 losses 上。键名不含 'loss'，不参与 loss 求和，只会被记录。
    tw_*: t 权重熵 / 最大桶权重；zsem_std: z_sem 跨图 std（存活判据）；ts_mean: 采样 t 均值。"""
    for k in ("tw_ent", "tw_max", "zsem_std", "ts_mean"):
        v = diff_out.get(k)
        if v is not None:
            losses[k] = v


# force=True：服务器若已通过其他途径注册同名类（如 __init__ 导入），静默覆盖而不是报错
@MODELS.register_module(force=True)
class FCNWithDiffAE(DiffAEMixin, EncoderDecoder):

    def __init__(
        self,
        backbone,
        decode_head,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        image_size: int = 128,
        enc_output_dim: int = 2048,
        cond_dim: int = 512,
        unet_channels: int = 128,
        unet_channel_mult: tuple = (1, 1, 2, 3),
        num_diffusion_timesteps: int = 1000,
        beta_schedule: str = "linear",
        spaced_timesteps: str = "ddim100",
        ema_decay: float = 0.9999,
        lambda_diff: float = 1.0,
        use_zero_module: bool = True,
        auto_ema: bool = True,
        num_res_blocks: int = 1,
        custom_timestep_sampler: Optional[List[int]] = None,
        # ---- t 自适应（透传给 DiffAEMixin -> TAdaptive）----
        t_weight_enabled: bool = False,
        t_weight_warmup_iters: int = 5000,
        t_weight_eta: float = 2.0,
        t_weight_eps: float = 0.1,
        t_weight_bins: int = 16,
        t_weight_interval: int = 500,
        t_weight_ema: float = 0.95,
        t_weight_probe_batch: int = 16,
        t_weight_static_window=None,
        t_weight_joint: bool = False,
        t_weight_z_cap: float = 1.0,
        t_weight_domain=None,
        t_weight_apply: str = 'sample',
        t_weight_pos_only: bool = True,
        # ---- 结构化噪声（透传给 DiffAEMixin -> MaskSelector / MaskApplier）----
        diffae_pred_x0: bool = False,
        diffae_mask_ratio: float = 0.0,
        diffae_mask_patch: int = 16,
        diffae_mask_mode: str = "random",
        diffae_mask_temp: float = 1.0,
        diffae_mask_soft: bool = True,
        diffae_mask_alpha: float = 1.0,
        diffae_mask_dynamic: bool = True,
        diffae_mask_ratio_max: float = 0.3,
        diffae_pretrain: bool = False,   # True = 预训练模式：跳过 seg 分支，只训扩散 loss
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            image_size=image_size,
            enc_output_dim=enc_output_dim,
            cond_dim=cond_dim,
            unet_channels=unet_channels,
            unet_channel_mult=unet_channel_mult,
            num_diffusion_timesteps=num_diffusion_timesteps,
            beta_schedule=beta_schedule,
            spaced_timesteps=spaced_timesteps,
            ema_decay=ema_decay,
            lambda_diff=lambda_diff,
            use_zero_module=use_zero_module,
            auto_ema=auto_ema,
            num_res_blocks=num_res_blocks,
            custom_timestep_sampler=custom_timestep_sampler,
            t_weight_enabled=t_weight_enabled,
            t_weight_warmup_iters=t_weight_warmup_iters,
            t_weight_eta=t_weight_eta,
            t_weight_eps=t_weight_eps,
            t_weight_bins=t_weight_bins,
            t_weight_interval=t_weight_interval,
            t_weight_ema=t_weight_ema,
            t_weight_probe_batch=t_weight_probe_batch,
            t_weight_static_window=t_weight_static_window,
            t_weight_joint=t_weight_joint,
            t_weight_z_cap=t_weight_z_cap,
            t_weight_domain=t_weight_domain,
            t_weight_apply=t_weight_apply,
            t_weight_pos_only=t_weight_pos_only,
            diffae_pred_x0=diffae_pred_x0,
            diffae_mask_ratio=diffae_mask_ratio,
            diffae_mask_patch=diffae_mask_patch,
            diffae_mask_mode=diffae_mask_mode,
            diffae_mask_temp=diffae_mask_temp,
            diffae_mask_soft=diffae_mask_soft,
            diffae_mask_alpha=diffae_mask_alpha,
            diffae_mask_dynamic=diffae_mask_dynamic,
            diffae_mask_ratio_max=diffae_mask_ratio_max,
            **kwargs,
        )
        self.image_size = image_size
        self.diffae_pretrain = diffae_pretrain
        if diffae_pretrain:
            # 预训练没有 seg 分支，hot/cold 选址拿不到任务梯度，只允许 random / 关闭
            assert diffae_mask_mode == 'random', (
                'diffae_pretrain=True 时无 seg 梯度，hot/cold 蒙块选址不可用，'
                '请用 random 或 diffae_mask_ratio=0 关闭蒙块')

    def _to_diffae_range(self, x: torch.Tensor) -> torch.Tensor:
        """ImageNet 归一化空间 -> [-1,1] @ image_size²（扩散输入）。"""
        mean = torch.tensor([123.675, 116.28, 103.53], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device=x.device).view(1, 3, 1, 1)
        x = (x * std + mean) / 255.0 * 2.0 - 1.0
        target_size = (self.image_size, self.image_size)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

    # ------------------------------------------------------------------
    # 推理/分析路径：encode_diffae 独立过 backbone
    # 训练时 loss() 不走这里，直接复用 feats[-1]
    # ------------------------------------------------------------------
    def encode_diffae(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        z = F.adaptive_avg_pool2d(feats[-1], 1)
        return z.flatten(1)

    # ------------------------------------------------------------------
    # 训练核心：backbone 只前向一次
    # ------------------------------------------------------------------
    def loss(self, inputs, data_samples):
        # 0) 结构化噪声选址需要 seg 输入梯度：本 batch 的 inputs 挂 requires_grad
        #    （叶子张量，迭代结束即弃；不影响参数，参数梯度仍由联合 backward 唯一写入）
        want_mask = self.training and self.mask_selector.enabled
        if want_mask:
            inputs = inputs.detach().requires_grad_(True)

        # 1) Backbone 前向一次
        feats = self.backbone(inputs)

        # 2) 分割分支（预训练模式跳过：encoder 只接收未被任务梯度稀释的去噪信号）
        losses = dict()
        if not self.diffae_pretrain:
            seg_feats = self.neck(feats) if self.with_neck else feats
            losses.update(self._decode_head_forward_train(seg_feats, data_samples))

            if self.with_auxiliary_head:
                losses.update(self._auxiliary_head_forward_train(seg_feats, data_samples))

        # 2.5) 梯度热力图 -> 结构化噪声选址（每步、逐样本；autograd.grad 只取输入梯度、
        #      不写参数梯度、retain 图给后续联合 backward）
        mask = None
        if want_mask:
            if self.mask_selector.mode == 'random':
                # 均匀蒙块不需要 seg 梯度：省掉整次反传
                heatmap = torch.zeros(inputs.shape[0], 1, self.image_size, self.image_size,
                                      device=inputs.device)
            else:
                task = sum(
                    v for k, v in losses.items()
                    if isinstance(v, torch.Tensor) and "loss" in k
                )
                gx = torch.autograd.grad(task, inputs, retain_graph=True)[0]     # ∂L_seg/∂x
                heatmap = gx.abs().mean(dim=1, keepdim=True)                     # [B,1,512,512]
                heatmap = F.adaptive_avg_pool2d(heatmap, self.image_size)        # [B,1,128,128]
            mask = self.mask_selector.select(heatmap)

        # 3) DiffAE 分支：直接复用 feats[-1]，不走 encode_diffae
        z_sem = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)
        cond = self.diffae_enc_compress(z_sem)

        x_diffae = self._to_diffae_range(inputs.detach())
        diff_out = self.forward_diffae(x_diffae, cond, input_mask=mask)
        losses['loss_diffae'] = diff_out['loss_diffae']
        _add_monitor_keys(losses, diff_out)

        return losses
