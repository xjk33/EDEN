"""
TransReID 适配层：把 eden 的 DiffAEMixin 挂到 build_transformer 上（MSMT17 ReID）。

与 cls/seg 适配层同一结构——全部交互逻辑集中在本文件：

  TransReIDWithDiffAE   —— 模型封装：backbone 单次前向，ID/triplet 损失与 DiffAE
                           辅助损失复用同一份特征；z_sem = global_feat（bottleneck
                           之前的全局特征，即分类器与 triplet 损失共同的输入特征，
                           ViT-S 为 384 维）——下游任务真正使用的特征。
                           结构化掩码选址需要 G = |∂L_task/∂x₀|，用
                           autograd.grad 只取输入梯度、不写参数梯度。
  build_eden_optimizers —— 双优化器：task 分支复用仓库原生 make_optimizer
                           （与 baseline 逐位一致，仅过滤掉 diffae 参数）；
                           diffae 分支独立 AdamW(lr=1e-4, wd=0)。
  build_diffae_scheduler—— diffae 分支独立调度：线性预热 + 余弦退火到 0
                           （按 epoch step，与仓库原生 scheduler 节奏一致）。

注意：ReID 输入为 256×128（H×W），PIXEL_MEAN/STD = 0.5 已把像素映射到 [-1,1]；
扩散分支输入统一双线性缩到 128×128（长宽比不保留——扩散分支只是训练期的约束
仪器，与其他臂的 128² 输入约定保持一致）。

包名说明：部署时 eden 包可以 mydiffae 为名挂在 PYTHONPATH 上，本地开发回退到 eden。
"""
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

from solver import make_optimizer

try:
    from mydiffae.diffae_mixin import DiffAEMixin
except ImportError:  # 本地仓库内开发（包名 eden）
    from eden.diffae_mixin import DiffAEMixin

# 扩散分支参数名前缀（与 cls/seg 两臂同一约定）
DIFFAE_PATTERNS = ('diffae_unet', 'diffae_enc_compress')


def _is_diffae_param(name: str) -> bool:
    return any(pat in name for pat in DIFFAE_PATTERNS)


def eden_kwargs_from_config(cfg) -> dict:
    """yacs config -> TransReIDWithDiffAE(**kwargs)。键名与 EDEN 配置节点一一对应。"""
    e = cfg.EDEN
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
        # yml 里写 1e-3 会被 YAML 1.1 解析成字符串（缺小数点）；
        # 旧版 yacs merge 时会 literal_eval 解码回 float，此处 float() 是双保险
        t_weight_delta=float(e.T_WEIGHT_DELTA),
    )


class TransReIDWithDiffAE(DiffAEMixin):
    """
    封装 build_transformer 的 EDEN 模型。task 参数 = 内部裸模型全部参数
    （名字带 'model.' 前缀），diffae 参数 = diffae_unet / diffae_enc_compress。

    forward(img)                                   -> 推理路径（去噪网络仅存在于训练阶段）
    forward(img, label, cam_label, view_label)     -> (score, feat)，与裸模型一致
    forward(..., criterion=loss_fn)                -> (losses dict, score)，训练路径
    """

    def __init__(self, model: torch.nn.Module, **diffae_kwargs):
        super().__init__(in_channels=3, **diffae_kwargs)
        self.model = model
        self.image_size = diffae_kwargs.get('image_size', 128)

    def forward(self, x, label=None, cam_label=None, view_label=None, criterion=None):
        if criterion is None:
            return self.model(x, label, cam_label=cam_label, view_label=view_label)
        return self.eden_loss(x, label, cam_label, view_label, criterion)

    def load_param(self, trained_path):
        """委托给内部裸模型（兼容仓库原生加载逻辑）。"""
        return self.model.load_param(trained_path)

    # ------------------------------------------------------------------
    # DiffAE 接口：z_sem = global_feat（bottleneck 之前的全局特征）
    # 与训练同分布——分类器与 triplet 损失共同的输入特征
    # ------------------------------------------------------------------
    def encode_diffae(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.base(x)                    # [B, 384]（ViT-S）

    def _to_diffae_range(self, x: torch.Tensor) -> torch.Tensor:
        """TransReID 归一化（mean/std=0.5）已把像素映射到 [-1,1]，只需缩到
        image_size²。输入 256×128 -> 128×128 不保留长宽比：扩散分支只是训练期
        的约束仪器，与其他臂的 128² 输入约定保持一致。"""
        return F.interpolate(x, size=(self.image_size, self.image_size),
                             mode='bilinear', align_corners=False)

    # ------------------------------------------------------------------
    # 训练核心：backbone 只前向一次，任务损失与扩散损失共享特征
    # ------------------------------------------------------------------
    def eden_loss(self, img, target, cam_label, view_label, criterion):
        """
        Returns: (losses, score) ——
          losses: loss_task / loss_diffae（参与求和；loss_diffae 已乘 lambda_diff）/
                  tw_ent / tw_max / zsem_std / ts_mean（监控标量，只记录）
          score:  分类 logits（处理器算训练精度用）
        """
        # 0) 结构化掩码选址需要任务损失对输入的梯度：本 batch 的 img 挂
        #    requires_grad（迭代结束即弃；参数梯度仍由联合 backward 唯一写入）
        want_mask = self.training and self.mask_selector.enabled
        if want_mask:
            img = img.detach().requires_grad_(True)

        # 1) Backbone 前向一次：z_sem 与 score 共享同一特征
        score, feat = self.model(img, target, cam_label=cam_label,
                                 view_label=view_label)          # feat: [B, 384]
        task_loss = criterion(score, feat, target, cam_label)

        # 2) 梯度热力图 G = |∂L_task/∂x₀| -> 掩码选址（预热期内 selector
        #    自动退化为均匀随机采样，G 照算但不生效；retain 图给联合 backward）
        mask = None
        if want_mask:
            gx = torch.autograd.grad(task_loss, img, retain_graph=True)[0]
            heatmap = gx.abs().mean(dim=1, keepdim=True)            # [B,1,256,128]
            heatmap = F.adaptive_avg_pool2d(heatmap, self.image_size)
            mask = self.mask_selector.select(heatmap)

        # 3) DiffAE 分支：AMP 下强制 fp32（探针配对测量要求数值稳定）；
        #    enc_output_dim == cond_dim 时 diffae_enc_compress 为 Identity
        with torch.cuda.amp.autocast(enabled=False):
            cond = self.diffae_enc_compress(feat.float())
            x_diffae = self._to_diffae_range(img.detach()).float()
            diff_out = self.forward_diffae(x_diffae, cond, input_mask=mask)

        losses = {'loss_task': task_loss, 'loss_diffae': diff_out['loss_diffae']}
        for k in ('tw_ent', 'tw_max', 'zsem_std', 'ts_mean'):
            if diff_out.get(k) is not None:
                losses[k] = diff_out[k]
        return losses, score


# ----------------------------------------------------------------------
# 双优化器：task（仓库原生 make_optimizer，与 baseline 逐位一致）
#          + diffae（AdamW 独立调度）
# ----------------------------------------------------------------------
class _TaskParamView:
    """把 wrapper 中非 diffae 的参数伪装成整个模型，喂给仓库原生
    make_optimizer——分组规则（bias lr 倍数等）与 baseline 完全一致。"""

    def __init__(self, model):
        self._model = model

    def named_parameters(self):
        for n, p in self._model.named_parameters():
            if p.requires_grad and not _is_diffae_param(n):
                yield n, p


def build_eden_optimizers(cfg, model, center_criterion):
    """返回 (optimizer_task, optimizer_diffae, optimizer_center)。
    model 为未包 DDP 的 wrapper。"""
    optimizer_task, optimizer_center = make_optimizer(
        cfg, _TaskParamView(model), center_criterion)
    diffae_params = [p for n, p in model.named_parameters()
                     if p.requires_grad and _is_diffae_param(n)]
    n_task = sum(p.numel() for n, p in model.named_parameters()
                 if p.requires_grad and not _is_diffae_param(n))
    print(f'[build_eden_optimizers] task={n_task / 1e6:.1f}M params, '
          f'diffae={sum(p.numel() for p in diffae_params) / 1e6:.1f}M params '
          f'(patterns={DIFFAE_PATTERNS})')
    optimizer_diffae = torch.optim.AdamW(
        diffae_params, lr=float(cfg.EDEN.DIFF_LR), betas=(0.9, 0.999),
        weight_decay=0.0)  # float()：yml 的 1e-4 可能是字符串（YAML 1.1）
    return optimizer_task, optimizer_diffae, optimizer_center


def build_diffae_scheduler(cfg, optimizer_diffae):
    """diffae 分支调度：线性预热 DIFF_WARMUP_EPOCHS 个 epoch + 余弦退火到 0
    （按 epoch step，与仓库原生 scheduler 节奏一致；与 cls/seg 臂同一形状）。"""
    warmup = cfg.EDEN.DIFF_WARMUP_EPOCHS
    total = cfg.SOLVER.MAX_EPOCHS
    schedulers = [
        lr_scheduler.LinearLR(optimizer_diffae, start_factor=1e-3,
                              total_iters=warmup),
        lr_scheduler.CosineAnnealingLR(optimizer_diffae,
                                       T_max=total - warmup, eta_min=0.0),
    ]
    return lr_scheduler.SequentialLR(optimizer_diffae, schedulers,
                                     milestones=[warmup])
