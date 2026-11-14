"""
SwinV2 适配层：把 eden 的 DiffAEMixin 挂到 SwinTransformerV2 上（ImageNet 分类）。

与 seg 适配层（mmseg/eden_adapter_fcn.py）同一结构——全部交互逻辑集中在本文件：

  SwinV2WithDiffAE      —— 模型封装：backbone 单次前向，分类 loss 与 DiffAE
                           辅助 loss 复用同一份特征；z_sem = forward_features
                           输出（分类器 head 的输入特征，tiny 为 768 维全局
                           向量）——下游任务真正使用的特征，而非中间层输出。
                           结构化掩码选址需要 G = |∂L_task/∂x₀|，用
                           autograd.grad 只取输入梯度、不写参数梯度。
  build_eden_optimizers —— 双优化器：task 分支完全复用仓库原生
                           build_optimizer（与 baseline 逐位一致，仅过滤掉
                           diffae 参数）；diffae 分支独立 AdamW(lr=1e-4, wd=0)。
  build_diffae_scheduler—— diffae 分支独立调度：LinearLR warmup + 余弦退火
                           到 0（与 seg 臂同一形状；task 分支沿用 baseline 调度）。
  eden_train_one_epoch  —— 训练循环：单次前向、单次反传、双优化器各自 step。
                           AMP 下分类分支随 autocast，扩散分支强制 fp32。
  eden_save/load_checkpoint —— 双优化器 / 双调度器状态的保存与恢复
                           （模型权重含 EMA shadow，由 DiffAEMixin 的
                           state_dict/load_state_dict 处理）。

包名说明：部署时 eden 包可以 mydiffae 为名挂在 PYTHONPATH 上，本地开发回退到 eden。
"""
import datetime
import time

import torch
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import AverageMeter

from optimizer import build_optimizer
from utils import ampscaler_get_grad_norm

try:
    from mydiffae.diffae_mixin import DiffAEMixin
except ImportError:  # 本地仓库内开发（包名 eden）
    from eden.diffae_mixin import DiffAEMixin

# 扩散分支参数名前缀（与 seg 的 SplitOptimWrapperConstructor 同一约定）
DIFFAE_PATTERNS = ('diffae_unet', 'diffae_enc_compress')

# ImageNet 归一化常数（data/build.py 的 timm 默认变换同值）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _is_diffae_param(name: str) -> bool:
    return any(pat in name for pat in DIFFAE_PATTERNS)


def eden_kwargs_from_config(config) -> dict:
    """yacs config -> SwinV2WithDiffAE(**kwargs)。键名与 EDEN 配置节点一一对应。"""
    e = config.EDEN
    return dict(
        image_size=e.IMAGE_SIZE,
        enc_output_dim=e.ENC_OUTPUT_DIM,
        cond_dim=e.COND_DIM,
        denoiser_type='dit',
        dit_size=e.DIT_SIZE,
        dit_patch=e.DIT_PATCH,
        dit_mlp_ratio=e.DIT_MLP_RATIO,
        adaln_hidden_mult=e.ADALN_HIDDEN_MULT,
        num_diffusion_timesteps=e.NUM_DIFFUSION_TIMESTEPS,
        beta_schedule=e.BETA_SCHEDULE,
        spaced_timesteps=e.SPACED_TIMESTEPS,
        ema_decay=e.EMA_DECAY,
        lambda_diff=e.LAMBDA_DIFF,
        diffae_pred_x0=e.PRED_X0,
        diffae_mask_enabled=e.MASK_ENABLED,
        diffae_mask_patch=e.MASK_PATCH,
        diffae_mask_ratio=e.MASK_RATIO,
        diffae_mask_alpha=e.MASK_ALPHA,
        t_weight_enabled=e.T_WEIGHT_ENABLED,
        t_weight_warmup_iters=e.T_WEIGHT_WARMUP_ITERS,
        t_weight_bins=e.T_WEIGHT_BINS,
        t_weight_interval=e.T_WEIGHT_INTERVAL,
        t_weight_ema=e.T_WEIGHT_EMA,
        t_weight_probe_batch=e.T_WEIGHT_PROBE_BATCH,
        t_weight_k=e.T_WEIGHT_K,
        t_weight_delta=e.T_WEIGHT_DELTA,
    )


class SwinV2WithDiffAE(DiffAEMixin):
    """
    封装 SwinTransformerV2 的 EDEN 模型。task 参数 = 内部裸模型全部参数
    （名字带 'model.' 前缀），diffae 参数 = diffae_unet / diffae_enc_compress。

    forward(samples)                    -> logits（验证/推理路径，与裸模型一致）
    forward(samples, targets, criterion)-> losses dict（训练路径，DDP 兼容：
                                           dict 输出经 DDP forward 正常返回）
    """

    def __init__(self, model: torch.nn.Module, **diffae_kwargs):
        super().__init__(in_channels=3, **diffae_kwargs)
        self.model = model
        self.image_size = diffae_kwargs.get('image_size', 128)

    # ------------------------------------------------------------------
    # 推理路径：与裸模型完全等价（去噪网络仅存在于训练阶段）
    # ------------------------------------------------------------------
    def forward(self, samples, targets=None, criterion=None):
        if targets is None:
            return self.model(samples)
        return self.eden_loss(samples, targets, criterion)

    def flops(self):
        return self.model.flops()

    # 仓库原生 build_optimizer 会读这两个接口决定 weight decay 分组；
    # 名称加 'model.' 前缀对齐 wrapper 视角
    def no_weight_decay(self):
        inner = self.model.no_weight_decay() if hasattr(self.model, 'no_weight_decay') else set()
        return {'model.' + n for n in inner}

    def no_weight_decay_keywords(self):
        if hasattr(self.model, 'no_weight_decay_keywords'):
            return self.model.no_weight_decay_keywords()
        return set()

    # ------------------------------------------------------------------
    # DiffAE 接口：z_sem = 分类器输入特征（forward_features 输出）
    # 与训练同分布——下游任务真正使用的特征
    # ------------------------------------------------------------------
    def encode_diffae(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)              # [B, num_features]

    def _to_diffae_range(self, x: torch.Tensor) -> torch.Tensor:
        """ImageNet 归一化空间 -> [-1,1] @ image_size²（扩散输入）。"""
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x = (x * std + mean) * 2.0 - 1.0
        target_size = (self.image_size, self.image_size)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

    # ------------------------------------------------------------------
    # 训练核心：backbone 只前向一次，分类 loss 与扩散 loss 共享特征图
    # ------------------------------------------------------------------
    def eden_loss(self, samples, targets, criterion):
        """
        Returns: dict ——
          loss_task / loss_diffae（参与求和；loss_diffae 已乘 lambda_diff）/
          tw_ent / tw_max / zsem_std / ts_mean（监控标量，只记录）
        """
        # 0) 结构化掩码选址需要任务 loss 对输入的梯度：本 batch 的 samples 挂
        #    requires_grad（mixup 后的张量，迭代结束即弃；参数梯度仍由联合
        #    backward 唯一写入）
        want_mask = self.training and self.mask_selector.enabled
        if want_mask:
            samples = samples.detach().requires_grad_(True)

        # 1) Backbone 前向一次：z_sem 与 logits 共享同一特征
        feats = self.model.forward_features(samples)       # [B, 768]
        logits = self.model.head(feats)
        task_loss = criterion(logits, targets)

        # 2) 梯度热力图 G = |∂L_task/∂x₀| -> 掩码选址（预热期内 selector
        #    自动退化为均匀随机采样，G 照算但不生效；retain 图给联合 backward）
        mask = None
        if want_mask:
            gx = torch.autograd.grad(task_loss, samples, retain_graph=True)[0]
            heatmap = gx.abs().mean(dim=1, keepdim=True)            # [B,1,H,W]
            heatmap = F.adaptive_avg_pool2d(heatmap, self.image_size)
            mask = self.mask_selector.select(heatmap)

        # 3) DiffAE 分支：AMP 下强制 fp32（探针配对测量要求数值稳定）；
        #    enc_output_dim == cond_dim 时 diffae_enc_compress 为 Identity
        with torch.cuda.amp.autocast(enabled=False):
            cond = self.diffae_enc_compress(feats.float())
            x_diffae = self._to_diffae_range(samples.detach()).float()
            diff_out = self.forward_diffae(x_diffae, cond, input_mask=mask)

        losses = {'loss_task': task_loss, 'loss_diffae': diff_out['loss_diffae']}
        for k in ('tw_ent', 'tw_max', 'zsem_std', 'ts_mean'):
            if diff_out.get(k) is not None:
                losses[k] = diff_out[k]
        return losses


# ----------------------------------------------------------------------
# 双优化器：task（仓库原生 build_optimizer，与 baseline 逐位一致）
#          + diffae（AdamW 独立调度）
# ----------------------------------------------------------------------
class _TaskParamView:
    """把 wrapper 中非 diffae 的参数伪装成整个模型，喂给仓库原生
    build_optimizer——分组规则（decay / no_decay）与 baseline 完全一致。"""

    def __init__(self, model):
        self._model = model

    def named_parameters(self):
        for n, p in self._model.named_parameters():
            if p.requires_grad and not _is_diffae_param(n):
                yield n, p

    def no_weight_decay(self):
        return self._model.no_weight_decay()

    def no_weight_decay_keywords(self):
        return self._model.no_weight_decay_keywords()


def build_eden_optimizers(config, model):
    """返回 (optimizer_task, optimizer_diffae)。model 为未包 DDP 的 wrapper。"""
    optimizer_task = build_optimizer(config, _TaskParamView(model))
    diffae_params = [p for n, p in model.named_parameters()
                     if p.requires_grad and _is_diffae_param(n)]
    n_task = sum(p.numel() for _, p in model.named_parameters()
                 if p.requires_grad and not _is_diffae_param(_))
    print(f'[build_eden_optimizers] task={n_task / 1e6:.1f}M params, '
          f'diffae={sum(p.numel() for p in diffae_params) / 1e6:.1f}M params '
          f'(patterns={DIFFAE_PATTERNS})')
    optimizer_diffae = torch.optim.AdamW(
        diffae_params, lr=config.EDEN.DIFF_LR, betas=(0.9, 0.999), weight_decay=0.0)
    return optimizer_task, optimizer_diffae


def build_diffae_scheduler(config, optimizer_diffae, n_iter_per_epoch):
    """diffae 分支调度：LinearLR warmup + 余弦退火到 0（与 seg 臂同一形状）。"""
    total = config.TRAIN.EPOCHS * n_iter_per_epoch
    warmup = config.EDEN.DIFF_WARMUP_ITERS
    return CosineLRScheduler(
        optimizer_diffae,
        t_initial=total - warmup,
        lr_min=0.0,
        warmup_lr_init=config.EDEN.DIFF_LR * 1e-3,
        warmup_t=warmup,
        cycle_limit=1,
        t_in_epochs=False,
        warmup_prefix=True,
    )


# ----------------------------------------------------------------------
# 训练循环：单次前向、单次反传、双优化器各自 step
# ----------------------------------------------------------------------
def eden_train_one_epoch(config, model, criterion, data_loader,
                         optimizer_task, optimizer_diffae, epoch, mixup_fn,
                         lr_scheduler_task, lr_scheduler_diffae, loss_scaler, logger):
    model.train()
    optimizer_task.zero_grad()
    optimizer_diffae.zero_grad()

    num_steps = len(data_loader)
    accum = config.TRAIN.ACCUMULATION_STEPS
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_diff_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            losses = model(samples, targets, criterion)     # DDP forward -> eden_loss
        loss = (losses['loss_task'] + losses['loss_diffae']) / accum

        # 单次反传 + 双优化器各自 step（task 参数按 baseline 裁剪，
        # diffae 参数不裁剪——与 seg 臂 clip_grad=None 一致）
        scaler = loss_scaler._scaler
        scaler.scale(loss).backward()
        if (idx + 1) % accum == 0:
            scaler.unscale_(optimizer_task)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer_task.param_groups for p in g['params']],
                config.TRAIN.CLIP_GRAD)
            scaler.unscale_(optimizer_diffae)
            scaler.step(optimizer_task)
            scaler.step(optimizer_diffae)
            scaler.update()
            optimizer_task.zero_grad()
            optimizer_diffae.zero_grad()
            num_updates = (epoch * num_steps + idx) // accum
            lr_scheduler_task.step_update(num_updates)
            lr_scheduler_diffae.step_update(num_updates)
        else:
            grad_norm = None
        loss_scale_value = loss_scaler.state_dict()['scale']

        torch.cuda.synchronize()

        loss_meter.update(losses['loss_task'].item(), targets.size(0))
        loss_diff_meter.update(losses['loss_diffae'].item(), targets.size(0))
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer_task.param_groups[0]['lr']
            lr_diff = optimizer_diffae.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'lr_diff {lr_diff:.2e}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'loss_diffae {loss_diff_meter.val:.4f} ({loss_diff_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'tw_ent {_v(losses, "tw_ent")}\t'
                f'tw_max {_v(losses, "tw_max")}\t'
                f'zsem_std {_v(losses, "zsem_std")}\t'
                f'ts_mean {_v(losses, "ts_mean")}\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def _v(losses, key):
    """监控标量格式化；未启用对应模块时显示 '-'。"""
    v = losses.get(key)
    return f'{v.item():.4f}' if isinstance(v, torch.Tensor) else '-'


# ----------------------------------------------------------------------
# Checkpoint：双优化器 / 双调度器状态（模型权重含 EMA shadow，
# 由 DiffAEMixin 的 state_dict/load_state_dict 处理）
# ----------------------------------------------------------------------
def eden_save_checkpoint(config, epoch, model, max_accuracy,
                         optimizer_task, optimizer_diffae,
                         lr_scheduler_task, lr_scheduler_diffae, loss_scaler, logger):
    import os
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer_task.state_dict(),
                  'optimizer_diffae': optimizer_diffae.state_dict(),
                  'lr_scheduler': lr_scheduler_task.state_dict(),
                  'lr_scheduler_diffae': lr_scheduler_diffae.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def eden_load_checkpoint(config, model, optimizer_task, optimizer_diffae,
                         lr_scheduler_task, lr_scheduler_diffae, loss_scaler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'epoch' in checkpoint:
        optimizer_task.load_state_dict(checkpoint['optimizer'])
        lr_scheduler_task.load_state_dict(checkpoint['lr_scheduler'])
        if 'optimizer_diffae' in checkpoint:
            optimizer_diffae.load_state_dict(checkpoint['optimizer_diffae'])
        if 'lr_scheduler_diffae' in checkpoint:
            lr_scheduler_diffae.load_state_dict(checkpoint['lr_scheduler_diffae'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' "
                    f"(epoch {checkpoint['epoch']})")
    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy
