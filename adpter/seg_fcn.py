import torch
import torch.nn.functional as F
import os
from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS

from mydiffae.diffae_mixin import DiffAEMixin
from mydiffae.pcgrad_mixin import PCGradMixin


def _add_tweight_keys(losses, diff_out):
    """把监控标量挂到 losses 上（不含 'loss' 字样，不参与 loss 求和，只会被记录）。
    tw_*: t 权重熵/最大桶权重；zsem_std: z_sem 跨图 std（复活判据）；mask_aux: 蒙区辅助 loss 值。
    值为 None（如未开蒙块）时跳过。"""
    for k in ("tw_ent", "tw_max", "zsem_std", "ts_mean"):
        v = diff_out.get(k)
        if v is not None:
            losses[k] = v


# =====================================================================
# Baseline
# =====================================================================
@MODELS.register_module()
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
        unet_channel_mult: tuple = (1, 2, 3, 4),
        num_diffusion_timesteps: int = 1000,
        beta_schedule: str = "linear",
        spaced_timesteps: str = "ddim100",
        ema_decay: float = 0.9999,
        lambda_diff: float = 1.0,
        use_zero_module: bool = True,
        auto_ema: bool = True,
        num_res_blocks: int = 2,
        # ---- 动态 t 加权 ----
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
        # ---- v2：x0 预测 + 空间蒙块 ----
        diffae_pred_x0: bool = False,
        diffae_mask_ratio: float = 0.0,
        diffae_mask_patch: int = 16,
        diffae_mask_mode: str = "random",
        diffae_mask_temp: float = 1.0,
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
            diffae_pred_x0=diffae_pred_x0,
            diffae_mask_ratio=diffae_mask_ratio,
            diffae_mask_patch=diffae_mask_patch,
            diffae_mask_mode=diffae_mask_mode,
            diffae_mask_temp=diffae_mask_temp,
            **kwargs,
        )
        self.image_size = image_size

    def _to_diffae_range(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([123.675, 116.28, 103.53], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device=x.device).view(1, 3, 1, 1)
        x = x * std + mean
        x = x / 255.0
        x = x * 2.0 - 1.0
        target_size = (self.image_size, self.image_size)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

    # ------------------------------------------------------------------
    # 推理/可视化路径：encode_diffae 独立过 backbone
    # 训练时 loss() 不走这里，直接复用 feats[-1]
    # ------------------------------------------------------------------
    def encode_diffae(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feat = feats[-1]
        z = F.adaptive_avg_pool2d(feat, 1)
        return z.flatten(1)

    # ------------------------------------------------------------------
    # 训练核心：backbone 只前向一次
    # ------------------------------------------------------------------
    def loss(self, inputs, data_samples):
        # 0) 互补蒙块需要 seg 输入梯度：本 batch 的 inputs 挂 requires_grad
        #    （叶子张量，迭代结束即弃；不影响参数，参数梯度仍由联合 backward 唯一写入）
        if self.training and self.diffae_mask_ratio > 0:
            inputs = inputs.detach().requires_grad_(True)

        # 1) Backbone 前向一次
        feats = self.backbone(inputs)

        # 2) 分割分支
        if self.with_neck:
            seg_feats = self.neck(feats)
        else:
            seg_feats = feats

        losses = dict()
        losses.update(self._decode_head_forward_train(seg_feats, data_samples))

        if self.with_auxiliary_head:
            losses.update(self._auxiliary_head_forward_train(seg_feats, data_samples))

        # 2.5) 梯度热力图 -> 互补蒙块（每步、逐样本；autograd.grad 只取输入梯度、
        #      不写参数梯度、retain 图给后续联合 backward；参数梯度仍由那一次 backward 唯一决定）
        mask = None
        if self.training and self.diffae_mask_ratio > 0:
            if self.diffae_mask_mode == 'random':
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
            mask = self._heatmap_to_mask(heatmap)

        # 3) DiffAE 分支：直接复用 feats[-1]，不走 encode_diffae
        feat = feats[-1]
        z_sem = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        cond = self.diffae_enc_compress(z_sem)

        x_diffae = self._to_diffae_range(inputs)
        diff_out = self.forward_diffae(x_diffae, cond, input_mask=mask)
        losses['loss_diffae'] = diff_out['loss_diffae']
        _add_tweight_keys(losses, diff_out)
        self._diff_loss_per_sample = diff_out['loss_diffae_per_sample']

        # 可视化（放在 loss() 里：PCGrad 开关两条路径都执行；验证走 predict 不会触发）
        self._vis_counter += 1
        if self.vis_interval > 0 and (
            self._vis_counter == 1
            or self._vis_counter % self.vis_interval == 0
        ):
            with torch.no_grad():
                x_vis = inputs[:self.vis_max_samples]
                x_diffae_vis = self._to_diffae_range(x_vis)
                os.makedirs(self.vis_dir, exist_ok=True)
                save_path = os.path.join(
                    self.vis_dir,
                    f"step_{self._vis_counter:06d}.png"
                )
                self.save_diffae_visualization(x_diffae_vis, save_path, self.vis_max_samples)

        return losses


from typing import Dict, Any, Optional, Tuple, List
# =====================================================================
# PCGrad 版本
# =====================================================================
@MODELS.register_module()
class FCNWithDiffAEPlusPCGrad(PCGradMixin, DiffAEMixin, EncoderDecoder):

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
        unet_channel_mult: tuple = (1, 2, 3, 4),
        num_diffusion_timesteps: int = 1000,
        beta_schedule: str = "linear",
        spaced_timesteps: str = "ddim100",
        ema_decay: float = 0.9999,
        lambda_diff: float = 1.0,
        use_zero_module: bool = True,
        auto_ema: bool = True,
        num_res_blocks: int = 2,
        custom_timestep_sampler: Optional[List[int]] = None,
        pcgrad_enabled: bool = True,
        pcgrad_lambda_d: float = 1.0,
        pcgrad_encoder_names: tuple = ("backbone",),
        pcgrad_mode: str = "global",
        vis_interval: int = 8000,
        vis_dir: str = './vis_diffae',
        vis_max_samples: int = 4,
        # ---- 动态 t 加权 ----
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
        # ---- v2：x0 预测 + 空间蒙块 ----
        diffae_pred_x0: bool = False,
        diffae_mask_ratio: float = 0.0,
        diffae_mask_patch: int = 16,
        diffae_mask_mode: str = "random",
        diffae_mask_temp: float = 1.0,
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
            pcgrad_enabled=pcgrad_enabled,
            pcgrad_lambda_d=pcgrad_lambda_d,
            pcgrad_encoder_names=pcgrad_encoder_names,
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
            diffae_pred_x0=diffae_pred_x0,
            diffae_mask_ratio=diffae_mask_ratio,
            diffae_mask_patch=diffae_mask_patch,
            diffae_mask_mode=diffae_mask_mode,
            diffae_mask_temp=diffae_mask_temp,
            **kwargs,
        )
        self.image_size = image_size
        self.pcgrad_mode = pcgrad_mode
        self.vis_interval = vis_interval
        self.vis_dir = vis_dir
        self.vis_max_samples = vis_max_samples
        self._vis_counter = 0

    def _to_diffae_range(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([123.675, 116.28, 103.53], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device=x.device).view(1, 3, 1, 1)
        x = x * std + mean
        x = x / 255.0
        x = x * 2.0 - 1.0
        target_size = (self.image_size, self.image_size)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

    def encode_diffae(self, x: torch.Tensor) -> torch.Tensor:
        """推理/可视化路径：独立过 backbone"""
        feats = self.backbone(x)
        feat = feats[-1]
        z = F.adaptive_avg_pool2d(feat, 1)
        return z.flatten(1)

    def loss(self, inputs, data_samples):
        # 0) 互补蒙块需要 seg 输入梯度：本 batch 的 inputs 挂 requires_grad
        #    （叶子张量，迭代结束即弃；不影响参数，参数梯度仍由联合 backward 唯一写入）
        if self.training and self.diffae_mask_ratio > 0:
            inputs = inputs.detach().requires_grad_(True)

        # 1) Backbone 只前向一次
        feats = self.backbone(inputs)

        # 2) 分割分支
        if self.with_neck:
            seg_feats = self.neck(feats)
        else:
            seg_feats = feats

        losses = dict()
        losses.update(self._decode_head_forward_train(seg_feats, data_samples))

        if self.with_auxiliary_head:
            losses.update(self._auxiliary_head_forward_train(seg_feats, data_samples))

        # 2.5) 梯度热力图 -> 互补蒙块（每步、逐样本；autograd.grad 只取输入梯度、
        #      不写参数梯度、retain 图给后续联合 backward；参数梯度仍由那一次 backward 唯一决定）
        mask = None
        if self.training and self.diffae_mask_ratio > 0:
            if self.diffae_mask_mode == 'random':
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
            mask = self._heatmap_to_mask(heatmap)

        # 3) DiffAE 分支：复用 feats[-1]
        feat = feats[-1]
        z_sem = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        cond = self.diffae_enc_compress(z_sem)

        x_diffae = self._to_diffae_range(inputs)
        diff_out = self.forward_diffae(x_diffae, cond, input_mask=mask)
        losses['loss_diffae'] = diff_out['loss_diffae']
        _add_tweight_keys(losses, diff_out)
        self._diff_loss_per_sample = diff_out['loss_diffae_per_sample']

        # 可视化（放在 loss() 里：PCGrad 开关两条路径都执行；验证走 predict 不会触发）
        self._vis_counter += 1
        if self.vis_interval > 0 and (
            self._vis_counter == 1
            or self._vis_counter % self.vis_interval == 0
        ):
            with torch.no_grad():
                x_vis = inputs[:self.vis_max_samples]
                x_diffae_vis = self._to_diffae_range(x_vis)
                os.makedirs(self.vis_dir, exist_ok=True)
                save_path = os.path.join(
                    self.vis_dir,
                    f"step_{self._vis_counter:06d}.png"
                )
                self.save_diffae_visualization(x_diffae_vis, save_path, self.vis_max_samples)

        return losses

    def train_step(self, data, optim_wrapper):
        if not self.pcgrad_enabled or not self.training:
            return super().train_step(data, optim_wrapper)

        # 1) 前向
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, training=True)
            losses = self.loss(data["inputs"], data["data_samples"])

        # 2) 分离 task loss 和 diff loss
        task_loss = sum(
            v for k, v in losses.items()
            if isinstance(v, torch.Tensor)
            and "loss" in k
            and k != "loss_diffae"
        )
        diff_loss = losses["loss_diffae"]
        diff_loss_per_sample = self._diff_loss_per_sample

        # 3) 按 mode 调用 PCGrad
        if self.pcgrad_mode == "global":
            self.pcgrad_backward(task_loss, diff_loss, optim_wrapper, mode="global")

        elif self.pcgrad_mode == "batch":
            self.pcgrad_backward(
                task_loss, diff_loss, optim_wrapper,
                mode="batch",
                diff_loss_per_sample=diff_loss_per_sample,
            )

        elif self.pcgrad_mode == "task":
            task_losses = {
                k: v for k, v in losses.items()
                if isinstance(v, torch.Tensor)
                and "loss" in k
                and k != "loss_diffae"
            }
            self.pcgrad_backward(
                task_loss, diff_loss, optim_wrapper,
                mode="task",
                task_losses=task_losses,
            )

        elif self.pcgrad_mode == "batch_task":
            B = diff_loss_per_sample.shape[0]
            task_losses_per_sample = {}
            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and "loss" in k and k != "loss_diffae":
                    if v.ndim == 0:
                        task_losses_per_sample[k] = v.unsqueeze(0).expand(B)
                    else:
                        task_losses_per_sample[k] = v
            self.pcgrad_backward(
                task_loss, diff_loss, optim_wrapper,
                mode="batch_task",
                task_losses_per_sample=task_losses_per_sample,
                diff_loss_per_sample=diff_loss_per_sample,
            )

        else:
            raise ValueError(f"Unknown pcgrad_mode: {self.pcgrad_mode}")

        # 4) 返回 log_vars（tw_* 不含 'loss'，不会被算进 task_loss，但会被记录）
        _, log_vars = self.parse_losses(losses)

        return log_vars