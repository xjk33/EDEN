"""
mmdet 适配层：把 eden 的 DiffAEMixin 挂到 MaskRCNN 上（COCO 检测 / 实例分割）。

与 seg 适配层（mmseg/seg_fcn.py）同一结构——全部交互逻辑集中在本文件：
  - backbone+neck 单次前向，RPN/RoI 损失与 DiffAE 辅助损失复用同一份 FPN 特征；
  - z_sem = FPN 最高层（P5，256ch）的全局平均池化——RPN 与 RoI 头共享的颈部
    特征（与 cls/seg 同一原则：取任务头真正使用的特征，而非 backbone 原始输出）；
  - 结构化噪声选址（mask_selector）需要检测损失对输入的梯度热力图，用
    autograd.grad 只取输入梯度、不写参数梯度；
  - 训练期零监控：tw_*/zsem_std 等标量挂在 losses 里随训练日志记录
    （键名不含 'loss'，不参与求和）。

包名说明：部署到 mmdetection 时 eden 包以 mydiffae 为名挂在 PYTHONPATH 上
（各 config 的 custom_imports 也是 mydiffae.*），本地开发回退到 eden。
部署：本文件拷为 mmdet/models/detectors/eden_adapter_maskrcnn.py，
eden 包拷到 mmdetection 仓库根目录（与 seg/cls 两臂相同的部署方式）。
"""
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from mmengine.model import MMDistributedDataParallel
from mmengine.optim import OptimWrapperDict
from mmdet.models.detectors.mask_rcnn import MaskRCNN
from mmdet.registry import (MODEL_WRAPPERS, MODELS, OPTIM_WRAPPER_CONSTRUCTORS,
                            OPTIM_WRAPPERS, OPTIMIZERS)

try:
    from mydiffae.diffae_mixin import DiffAEMixin
except ImportError:  # 本地仓库内开发（包名 eden）
    from eden.diffae_mixin import DiffAEMixin


@OPTIM_WRAPPER_CONSTRUCTORS.register_module(force=True)
class SplitOptimWrapperConstructor:
    """
    双优化器构造器：按参数名前缀把模型参数切给 task / diffae 两个子优化器。
    diffae_patterns 命中的参数进 'diffae'（AdamW 独立调度），其余全部进
    'task'（SGD + 1x 调度，与 baseline 完全一致）。配合 config 里
    param_scheduler 的 dict 写法（key 与优化器同名）实现两套独立 lr 体制。
    （与 seg 臂同名同实现；各框架部署副本独立，避免跨仓库 import。）

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


@MODEL_WRAPPERS.register_module(force=True)
class EDENDistributedDataParallel(MMDistributedDataParallel):
    """
    支持 OptimWrapperDict（task + diffae 双优化器）的 DDP 包装。

    原生 MMDDP.train_step 强制调用 optim_wrapper.optim_context()，而
    OptimWrapperDict 未实现该方法（NotImplementedError）——单卡时走
    BaseModel.train_step -> MaskRCNNWithDiffAE.train_step，不触发；
    多卡 DDP 才会走到。本类对 OptimWrapperDict 复刻模型侧 train_step
    的手动编排（单次前向、单次反传、各子优化器分别 step/zero_grad），
    唯一区别是前向经 self._run_forward —— 走 DDP.__call__，reducer 的
    bucket 标记与梯度同步行为与原生多卡路径一致。无梯度累积、无 AMP 时
    optim_context 只是计数器重置，跳过无影响。单优化器时回退原生实现。

    config 写法：model_wrapper_cfg = dict(type='EDENDistributedDataParallel')
    """

    def train_step(self, data, optim_wrapper):
        if not isinstance(optim_wrapper, OptimWrapperDict):
            return super().train_step(data, optim_wrapper)
        data = self.module.data_preprocessor(data, True)
        losses = self._run_forward(data, mode='loss')
        parsed_loss, log_vars = self.module.parse_losses(losses)
        next(iter(optim_wrapper.values())).backward(parsed_loss)
        for wrapper in optim_wrapper.values():
            wrapper.step()
            wrapper.zero_grad()
        return log_vars


def _add_monitor_keys(losses, diff_out):
    """把监控标量挂到 losses 上。键名不含 'loss'，不参与 loss 求和，只会被记录。
    tw_*: t 权重熵 / 最大桶权重；zsem_std: z_sem 跨图 std（存活判据）；ts_mean: 采样 t 均值。"""
    for k in ("tw_ent", "tw_max", "zsem_std", "ts_mean"):
        v = diff_out.get(k)
        if v is not None:
            losses[k] = v


# force=True：服务器若已通过其他途径注册同名类（如 __init__ 导入），静默覆盖而不是报错
@MODELS.register_module(force=True)
class MaskRCNNWithDiffAE(DiffAEMixin, MaskRCNN):

    def __init__(
        self,
        backbone,
        rpn_head,
        roi_head,
        train_cfg,
        test_cfg,
        neck=None,
        data_preprocessor=None,
        init_cfg=None,
        image_size: int = 128,
        enc_output_dim: int = 256,   # FPN out_channels（z_sem 取 P5 全局池化）
        cond_dim: int = 256,
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
        # ---- t 自适应（透传给 DiffAEMixin -> TAdaptive）----
        t_weight_enabled: bool = False,
        t_weight_warmup_iters: int = 5000,
        t_weight_bins: int = 16,
        t_weight_interval: int = 500,
        t_weight_ema: float = 0.95,
        t_weight_probe_batch: int = 16,
        t_weight_k: float = 0.5,        # 贡献度 μ 的不确定性惩罚系数（D − k·σ_D）
        t_weight_delta: float = 1e-3,   # 贡献度 μ 的量纲归一保护常数（E[L_0] + δ）
        t_weight_eps=None,              # 采样分布的均匀保底总份额；None = 1/K
        # ---- 结构化掩码（透传给 DiffAEMixin -> MaskSelector / MaskApplier）----
        diffae_pred_x0: bool = False,
        diffae_mask_enabled: bool = False,
        diffae_mask_patch: int = 16,
        diffae_mask_ratio: float = 0.25,  # 数量限制：每图蒙 round(ratio·n) 块
        diffae_mask_alpha: float = 1.0,   # 掩码强度：1=完全替换为均值色，0=不蒙
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            neck=neck,
            data_preprocessor=data_preprocessor,
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
            t_weight_enabled=t_weight_enabled,
            t_weight_warmup_iters=t_weight_warmup_iters,
            t_weight_bins=t_weight_bins,
            t_weight_interval=t_weight_interval,
            t_weight_ema=t_weight_ema,
            t_weight_probe_batch=t_weight_probe_batch,
            t_weight_k=t_weight_k,
            t_weight_delta=t_weight_delta,
            t_weight_eps=t_weight_eps,
            diffae_pred_x0=diffae_pred_x0,
            diffae_mask_enabled=diffae_mask_enabled,
            diffae_mask_patch=diffae_mask_patch,
            diffae_mask_ratio=diffae_mask_ratio,
            diffae_mask_alpha=diffae_mask_alpha,
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
        # 单次前向、单次反传（检测与扩散分支共享 backbone+neck，计算图是一张），
        # 两个子优化器各自 step / zero_grad（各自只持有自己的参数）。
        data = self.data_preprocessor(data, True)
        losses = self._run_forward(data, mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses)
        next(iter(optim_wrapper.values())).backward(parsed_losses)
        for wrapper in optim_wrapper.values():
            wrapper.step()
            wrapper.zero_grad()
        return log_vars

    def _to_diffae_range(self, x: torch.Tensor) -> torch.Tensor:
        """mmdet 归一化空间（ImageNet mean/std）-> [-1,1] @ image_size²（扩散输入）。"""
        mean = torch.tensor([123.675, 116.28, 103.53], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device=x.device).view(1, 3, 1, 1)
        x = (x * std + mean) / 255.0 * 2.0 - 1.0
        target_size = (self.image_size, self.image_size)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

    # ------------------------------------------------------------------
    # 推理/分析路径：encode_diffae 独立过 backbone + neck
    # 与训练同分布：z_sem = FPN 最高层（P5）的全局池化
    # ------------------------------------------------------------------
    def encode_diffae(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_feat(x)
        z = F.adaptive_avg_pool2d(feats[-1], 1)
        return z.flatten(1)

    # ------------------------------------------------------------------
    # 训练核心：backbone+neck 只前向一次（与 TwoStageDetector.loss 完全等价，
    # 仅复用特征 x 并追加 DiffAE 分支）
    # ------------------------------------------------------------------
    def loss(self, batch_inputs: torch.Tensor, batch_data_samples) -> dict:
        # 0) 结构化噪声选址需要检测损失对输入的梯度：本 batch 的 inputs 挂
        #    requires_grad（叶子张量，迭代结束即弃；参数梯度仍由联合 backward 唯一写入）
        want_mask = self.training and self.mask_selector.enabled
        if want_mask:
            batch_inputs = batch_inputs.detach().requires_grad_(True)

        # 1) backbone + FPN 前向一次
        x = self.extract_feat(batch_inputs)

        losses = dict()

        # 2) RPN forward and loss（与 TwoStageDetector.loss 逐行一致）
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        # 2.5) 梯度热力图 G = |∂L_task/∂x₀| -> 结构化掩码选址（autograd.grad 只取
        #      输入梯度、不写参数梯度、retain 图给后续联合 backward；
        #      预热期内 selector 自动退化为均匀随机采样，G 照算但不生效）
        mask = None
        if want_mask:
            task = sum(
                v for k, v in losses.items()
                if isinstance(v, torch.Tensor) and "loss" in k
            )
            gx = torch.autograd.grad(task, batch_inputs, retain_graph=True)[0]
            heatmap = gx.abs().mean(dim=1, keepdim=True)
            heatmap = F.adaptive_avg_pool2d(heatmap, self.image_size)
            mask = self.mask_selector.select(heatmap)

        # 3) DiffAE 分支：z_sem = FPN 最高层（P5）的全局池化——RPN 与 RoI 头
        #    共享的颈部特征；enc_output_dim == cond_dim 时 enc_compress 为 Identity
        z_sem = F.adaptive_avg_pool2d(x[-1], 1).flatten(1)
        cond = self.diffae_enc_compress(z_sem)

        x_diffae = self._to_diffae_range(batch_inputs.detach())
        diff_out = self.forward_diffae(x_diffae, cond, input_mask=mask)
        losses['loss_diffae'] = diff_out['loss_diffae']
        _add_monitor_keys(losses, diff_out)

        return losses
