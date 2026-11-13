"""
结构化掩码：任务引导的选址 + 替换型施加（定稿方案）

拆分自 diffae_mixin.py，两个组件各司其职：
  MaskSelector —— 选址：从下游任务敏感度 G 得到逐块掩码概率并独立采样
  MaskApplier  —— 施加：x̃₀ = m⊙c + (1−m)⊙x₀（替换型退化，c 为数据集均值色），
                  输出 sampler.training_losses 的输入参数

方法学映射（与论文 3.3 节一一对应）：
  - 空间敏感度 G = |∂L_task/∂x₀|（在 adapter 中计算，此处只消费热力图）
  - 块级敏感度 = 划分为 b×b 块、块内取平均、逐图归一化（min-max 到 [0,1]）
  - 掩码概率   = p_i = σ((h_i − μ_h)/s_h)：μ_h 控触发阈值，s_h 控过渡带软硬；
    逐图归一化使 h_i 不随任务损失量级漂移；sigmoid 软门控而非硬阈值
  - 掩码采样   = 各块以 p_i 独立采样二值掩码（每图蒙块数量随机，期望 Σp_i）
  - 预热期     = 前 warmup_iters 步 p_i 恒为 warmup_ratio（均匀随机采样，
    不启用 G 引导），预热结束与 t 自适应同时启用引导
  - 施加       = 掩码只作用于扩散分支输入；loss 目标始终是完整 x₀，
    无掩码时自然退化为标准扩散训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskSelector:
    """
    选址：敏感度热力图 -> 0/1 keep-mask [B,1,H,W]（1=保留，0=被选中蒙住）。

    每块独立 Bernoulli：p_i = σ((h_i − μ_h)/s_h)，h_i 为逐图 min-max 归一化的
    块级敏感度；预热期内 p_i 恒为 warmup_ratio（均匀随机采样）。
    """

    def __init__(
        self,
        enabled: bool = False,     # 总开关
        patch: int = 16,           # 蒙块边长（像素；128 图上 16px = 8×8 块）
        mu_h: float = 0.5,         # sigmoid 触发阈值（归一化敏感度坐标）
        s_h: float = 0.1,          # sigmoid 过渡带软硬（越小越接近硬阈值）
        warmup_iters: int = 5000,  # 预热期：均匀随机采样，不启用 G 引导
        warmup_ratio: float = 0.1, # 预热期均匀掩码概率（每块独立）
    ):
        self.enabled = enabled
        self.patch = patch
        self.mu_h = mu_h
        self.s_h = s_h
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self._step = 0             # 已选址次数（训练步），预热计时用

    def select(self, heatmap: torch.Tensor, guided: bool = None) -> torch.Tensor:
        """
        Args:
            heatmap: [B,1,H,W] 敏感度热力图 G
            guided:  None = 按内部计数器自动判定（前 warmup_iters 步为均匀采样）；
                     True/False = 强制引导/均匀（离线可视化用）
        Returns:
            keep-mask [B,1,H,W]，1=保留，0=蒙住
        """
        self._step += 1
        if guided is None:
            guided = self._step > self.warmup_iters

        B = heatmap.shape[0]
        p = self.patch
        H, W = heatmap.shape[-2:]
        g_h, g_w = max(1, H // p), max(1, W // p)

        if guided:
            # 块内均值 -> 逐图 min-max 归一化 -> sigmoid 伪概率
            h = F.adaptive_avg_pool2d(heatmap, (g_h, g_w))           # [B,1,g_h,g_w]
            hmin = h.amin(dim=(2, 3), keepdim=True)
            hmax = h.amax(dim=(2, 3), keepdim=True)
            h = (h - hmin) / (hmax - hmin).clamp_min(1e-8)
            prob = torch.sigmoid((h - self.mu_h) / self.s_h)
        else:
            # 预热期：均匀随机采样（不启用 G 引导）
            prob = torch.full((B, 1, g_h, g_w), self.warmup_ratio,
                              device=heatmap.device)

        sel = torch.bernoulli(prob)                                  # 1=蒙住
        keep = 1.0 - sel
        mask = keep.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
        return mask[:, :, :H, :W]


class MaskApplier(nn.Module):
    """
    施加：替换型退化 x̃₀ = m⊙c + (1−m)⊙x₀（keep-mask m：1=保留），
    c 为数据集均值色；loss 目标始终是完整原图（target_override），
    蒙区内容的唯一信息源是 cond（z_sem）。
    """

    def __init__(self):
        super().__init__()
        # 替换色 c：ImageNet 均值在 [-1,1] 空间的取值（恰好等于 padding 的颜色）
        _mu = [(123.675 / 255.0) * 2 - 1, (116.28 / 255.0) * 2 - 1, (103.53 / 255.0) * 2 - 1]
        self.register_buffer('mask_mean', torch.tensor(_mu).view(1, 3, 1, 1))

    def transform(self, x_start: torch.Tensor,
                  mask: "torch.Tensor | None") -> dict:
        """
        返回 sampler.training_losses 的输入 kwargs。
        mask=None（未启用掩码）时原样透传，退化为标准扩散训练。
        """
        if mask is None:
            return {"x_start": x_start}
        x_in = mask * x_start + (1.0 - mask) * self.mask_mean.to(x_start.device)
        return {
            "x_start": x_in,                 # 输入：蒙区替换为均值色
            "target_override": x_start,      # 目标：原始干净图（蒙区内容只能靠 z_sem）
        }
