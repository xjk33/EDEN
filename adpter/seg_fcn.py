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
from mmengine.optim import OptimWrapperDict
from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import (MODELS, OPTIM_WRAPPER_CONSTRUCTORS,
                            OPTIM_WRAPPERS, OPTIMIZERS)
from mmseg.utils import add_prefix

try:
    from mydiffae.diffae_mixin import DiffAEMixin
except ImportError:  # 本地仓库内开发（包名 eden）
    from eden.diffae_mixin import DiffAEMixin


@OPTIM_WRAPPER_CONSTRUCTORS.register_module(force=True)
class SplitOptimWrapperConstructor:
    """
    双优化器构造器：按参数名前缀把模型参数切给 task / diffae 两个子优化器。
    diffae_patterns 命中的参数进 'diffae'（AdamW 独立调度），其余全部进
    'task'（SGD + PolyLR，与 baseline 完全一致）。配合 config 里
    param_scheduler 的 dict 写法（key 与优化器同名）实现两套独立 lr 体制。

    config 写法：
        optim_wrapper = dict(
            constructor='SplitOptimWrapperConstructor',
            task=dict(optimizer=dict(type='SGD', ...)),
            diffae=dict(optimizer=dict(type='AdamW', ...)),
        )
    """

    def __init__(self, optim_wrapper_cfg: dict, paramwise_cfg=None,
                 diffae_key: str = 'diffae',
                 diffae_patterns=('diffae_unet', 'diffae_enc_compress')):
        assert isinstance(optim_wrapper_cfg, dict), 'optim_wrapper_cfg 必须是 dict'
        assert diffae_key in optim_wrapper_cfg, \
            f'optim_wrapper 缺少 {diffae_key!r} 子优化器'
        others = [k for k in optim_wrapper_cfg if k != diffae_key]
        assert len(others) == 1, \
            f'除 {diffae_key} 外应恰有一个 task 子优化器，得到 {others}'
        self.task_key = others[0]
        self.diffae_key = diffae_key
        self.patterns = tuple(diffae_patterns)
        self.wrapper_cfgs = {k: v.copy() for k, v in optim_wrapper_cfg.items()}
        if paramwise_cfg:
            raise NotImplementedError(
                'SplitOptimWrapperConstructor 不支持 paramwise_cfg：'
                '参数归属已由名称前缀决定，各分支超参写在各自 optimizer 里')

    def __call__(self, model) -> OptimWrapperDict:
        if hasattr(model, 'module'):
            model = model.module
        splits = {self.task_key: [], self.diffae_key: []}
        n_task = n_diff = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(pat in name for pat in self.patterns):
                splits[self.diffae_key].append(p)
                n_diff += p.numel()
            else:
                splits[self.task_key].append(p)
                n_task += p.numel()
        print(f'[SplitOptimWrapperConstructor] task={n_task / 1e6:.1f}M params, '
              f'{self.diffae_key}={n_diff / 1e6:.1f}M params '
              f'(patterns={self.patterns})')

        wrappers = {}
        for key, cfg in self.wrapper_cfgs.items():
            cfg = cfg.copy()
            cfg.setdefault('type', 'OptimWrapper')
            opt_cfg = cfg.pop('optimizer').copy()
            with OPTIMIZERS.switch_scope_and_registry(None) as registry:
                opt_cls = registry.get(opt_cfg.pop('type'))
            optimizer = opt_cls(params=splits[key], **opt_cfg)
            wrappers[key] = OPTIM_WRAPPERS.build(
                cfg, default_args=dict(optimizer=optimizer))
        return OptimWrapperDict(**wrappers)


def _add_monitor_keys(losses, diff_out):
    """把监控标量挂到 losses 上。键名不含 'loss'，不参与 loss 求和，只会被记录。
    tw_*: t 权重熵 / 最大桶权重；zsem_std: z_sem 跨图 std（存活判据）；ts_mean: 采样 t 均值；
    ants_*: Beta 采样器 (a, b) 均值 / 队列平滑 reward / 采样熵。"""
    for k in ("tw_ent", "tw_max", "zsem_std", "ts_mean",
              "ants_a", "ants_b", "ants_reward", "ants_ent"):
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
        enc_output_dim: int = 512,   # 须等于 decode_head.channels（z_sem 取分类器输入特征）
        cond_dim: int = 512,
        unet_channels: int = 128,
        unet_channel_mult: tuple = (1, 1, 2, 3),
        # ---- 去噪器选择：'dit'（弱仪器，i-DAE 路线，默认）| 'unet' ----
        denoiser_type: str = 'dit',
        dit_size: str = 'ti',   # 'ti'(~6.7M) | 's'(~25.8M) | 'b'(~100M)；表征已解耦到 encoder，仪器只做小
        dit_dim=None,           # 显式给定则覆盖预设（逐项）
        dit_depth=None,
        dit_heads=None,
        dit_patch: int = 16,
        dit_mlp_ratio: float = 4.0,
        adaln_hidden_mult: float = 0.25,
        num_diffusion_timesteps: int = 1000,
        beta_schedule: str = "linear",
        spaced_timesteps: str = "ddim100",
        ema_decay: float = 0.9999,
        lambda_diff: float = 1.0,
        use_zero_module: bool = True,
        auto_ema: bool = True,
        num_res_blocks: int = 1,
        custom_timestep_sampler: Optional[List[int]] = None,
        # ---- ANTS（透传给 DiffAEMixin -> ANTSampler，与 t_weight/窗口互斥）----
        ants_enabled: bool = False,
        ants_lr: float = 1e-3,
        ants_entropy_coef: float = 1e-2,
        ants_interval: int = 40,
        ants_queue: int = 20,
        ants_rep_timesteps: tuple = (200, 500, 800),
        ants_hidden: int = 256,
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
            denoiser_type=denoiser_type,
            dit_size=dit_size,
            dit_dim=dit_dim,
            dit_depth=dit_depth,
            dit_heads=dit_heads,
            dit_patch=dit_patch,
            dit_mlp_ratio=dit_mlp_ratio,
            adaln_hidden_mult=adaln_hidden_mult,
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
            ants_enabled=ants_enabled,
            ants_lr=ants_lr,
            ants_entropy_coef=ants_entropy_coef,
            ants_interval=ants_interval,
            ants_queue=ants_queue,
            ants_rep_timesteps=ants_rep_timesteps,
            ants_hidden=ants_hidden,
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

    # ------------------------------------------------------------------
    # 双优化器训练步：task（SGD）+ diffae（AdamW）共享一次前向/反传
    # ------------------------------------------------------------------
    def train_step(self, data, optim_wrapper):
        if not isinstance(optim_wrapper, OptimWrapperDict):
            # 单优化器配置（纯 baseline / 旧实验复现）：走 mmengine 原生路径
            return super().train_step(data, optim_wrapper)
        # OptimWrapperDict 没有 update_params/backward，手动编排：
        # 单次前向、单次反传（seg 与扩散分支共享 backbone，计算图是一张），
        # 两个子优化器各自 step / zero_grad（各自只持有自己的参数）。
        data = self.data_preprocessor(data, True)
        losses = self._run_forward(data, mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses)
        next(iter(optim_wrapper.values())).backward(parsed_losses)
        for wrapper in optim_wrapper.values():
            wrapper.step()
            wrapper.zero_grad()
        # ANTS：去噪器已更新，消费 reward 簿记、REINFORCE 更新 Beta 采样器
        # （未启用或未到测量间隔时为空操作）
        self.ants_after_update()
        return log_vars

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
    # 推理/分析路径：encode_diffae 独立过 backbone + decode head
    # 与训练同分布：z_sem = decode head 分类器输入特征（_forward_feature 输出）
    # 的全局池化——下游任务真正使用的特征，而非 backbone 原始输出
    # ------------------------------------------------------------------
    def encode_diffae(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        seg_feats = self.neck(feats) if self.with_neck else feats
        head_feat = self.decode_head._forward_feature(seg_feats)
        z = F.adaptive_avg_pool2d(head_feat, 1)
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

        # 2) 分割分支：手动走 decode head，截获分类器输入特征 head_feat
        #    （_forward_feature = convs + conv_cat，cls_seg = dropout + conv_seg 1x1；
        #      与 _decode_head_forward_train 完全等价，loss 键名经 add_prefix 不变）
        losses = dict()
        seg_feats = self.neck(feats) if self.with_neck else feats
        head_feat = self.decode_head._forward_feature(seg_feats)      # [B, 512, h, w]
        seg_logits = self.decode_head.cls_seg(head_feat)
        losses.update(add_prefix(
            self.decode_head.loss_by_feat(seg_logits, data_samples), 'decode'))

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

        # 3) DiffAE 分支：z_sem = head_feat 的全局池化——seg loss 与 diff loss
        #    在同一表示（下游真正使用的特征）上交汇；enc_output_dim=512 时
        #    diffae_enc_compress 为 Identity
        z_sem = F.adaptive_avg_pool2d(head_feat, 1).flatten(1)
        cond = self.diffae_enc_compress(z_sem)

        x_diffae = self._to_diffae_range(inputs.detach())
        diff_out = self.forward_diffae(x_diffae, cond, input_mask=mask)
        losses['loss_diffae'] = diff_out['loss_diffae']
        _add_monitor_keys(losses, diff_out)

        return losses
