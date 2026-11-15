"""
结构化掩码：任务引导的选址 + 软替换施加

拆分自 diffae_mixin.py，两个组件各司其职：
  MaskSelector —— 选址：从下游任务敏感度 G 得到逐块概率，按概率定量采样
  MaskApplier  —— 施加：软替换退化（蒙区向均值色按 alpha 混合），
                  输出 sampler.training_losses 的输入参数

机制：
  - 空间敏感度 G = |∂L_task/∂x₀|（在 adapter 中计算，此处只消费热力图）
  - 块级概率   = G 划块取均值得 h，逐图均值归一后 softmax((h/h̄)/τ) 得逐块
    概率 p（尺度不变化：引导强度与 G 的绝对量纲无关，τ 控软硬）
  - 掩码采样   = 每图按 p 无放回采固定数量 k = round(ratio·n) 个块
    （Gumbel-top-k 技巧：log p + Gumbel 噪声取前 k，等价于按 p 无放回采样）
  - 预热期     = 前 warmup_iters 步退化为均匀采样（同样 k 块、位置随机，
    不启用 G 引导），预热结束与 t 自适应同时启用引导
  - 施加       = x̃₀ = (1−α(1−m))⊙x₀ + α(1−m)⊙c（m：keep-mask，c 为该图
    自己的均值色）；alpha=1 完全替换，alpha=0 退化为标准扩散；loss 目标
    始终是完整 x₀，蒙区内容的唯一信息源是 cond（z_sem）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskSelector:
    """
    选址：敏感度热力图 -> 0/1 keep-mask [B,1,H,W]（1=保留，0=被选中蒙住）。

    每图按 softmax((h/h̄)/τ) 概率无放回采固定 k = round(ratio·n) 个块
    （h̄ 为逐图块均值——尺度不变化：logits 是"相对敏感度"，与 G 的绝对
    量纲无关；τ 控采样软硬：τ→0 趋近硬 top-k，τ→∞ 趋近均匀）；
    预热期内概率退化为均匀（同样 k 块、位置随机）。
    """

    def __init__(
        self,
        enabled: bool = False,     # 总开关
        patch: int = 16,           # 蒙块边长（像素；128 图上 16px = 8×8 = 64 块）
        ratio: float = 0.25,       # 掩码量：每图蒙块比例（块数 = round(ratio·n)）
        warmup_iters: int = 5000,  # 预热期：均匀随机采样，不启用 G 引导
        tau: float = 1.0,          # 采样温度 τ：logits = (h/h̄)/τ
        delta: float = 1e-8,       # 归一化保护常数（h̄=0 时退化为均匀，不除零）
    ):
        self.enabled = enabled
        self.patch = patch
        self.ratio = ratio
        self.warmup_iters = warmup_iters
        self.tau = tau
        self.delta = delta
        self._step = 0             # 已选址次数（训练步），预热计时用

    def select(self, heatmap: torch.Tensor, guided: bool = None,
               ratio: float = None) -> torch.Tensor:
        """
        Args:
            heatmap: [B,1,H,W] 敏感度热力图 G
            guided:  None = 按内部计数器自动判定（前 warmup_iters 步为均匀采样）；
                     True/False = 强制引导/均匀（离线可视化用）
            ratio:   本次调用的掩码量覆盖；None = 用 self.ratio
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
        n = g_h * g_w
        r = self.ratio if ratio is None else ratio
        k = max(1, int(round(r * n)))                          # 每图固定蒙块数

        if guided:
            # 块内均值 -> 逐图归一化 -> softmax 概率。
            # 除以逐图均值 h̄（尺度不变化）：logits = 相对敏感度，与 G 的
            # 绝对量纲无关——否则小梯度任务（如 det，G~1e-4）softmax 退化
            # 为均匀分布，G 引导失效。G≈0 的块仍有 e⁰=1 的基础概率
            # （全支撑，保留探索）；h̄=0（梯度全零）时 logits 全 0，
            # 自动退化为均匀采样。
            h = F.adaptive_avg_pool2d(heatmap, (g_h, g_w)).reshape(B, n)
            logits = h / (h.mean(dim=1, keepdim=True) + self.delta) / self.tau
            prob = torch.softmax(logits, dim=1)
        else:
            # 预热期：均匀概率（位置随机，数量相同）
            prob = torch.full((B, n), 1.0 / n, device=heatmap.device)

        # Gumbel-top-k：log p + Gumbel 噪声取前 k = 按 p 无放回采样
        u = torch.rand_like(prob).clamp_min(1e-12)
        score = torch.log(prob.clamp_min(1e-12)) - torch.log(-torch.log(u))
        order = score.argsort(dim=1, descending=True)            # [B, n]
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(n, device=heatmap.device)
                      .unsqueeze(0).expand(B, n))
        sel = rank < k                                           # [B, n] 前 k 个被选中

        keep = (~sel).float().view(B, 1, g_h, g_w)
        mask = keep.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
        return mask[:, :, :H, :W]


class MaskApplier(nn.Module):
    """
    施加：软替换退化 x̃₀ = w⊙x₀ + (1−w)⊙c，w = 1−α(1−m)
    （m：keep-mask，1=保留；c 为每张图自己的均值色；alpha 控替换程度——
    1 = 完全替换，0 = 不蒙）。loss 目标始终是完整原图（target_override），
    蒙区内容的唯一信息源是 cond（z_sem）。
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def transform(self, x_start: torch.Tensor,
                  mask: "torch.Tensor | None") -> dict:
        """
        返回 sampler.training_losses 的输入 kwargs。
        mask=None（未启用掩码）时原样透传，退化为标准扩散训练。
        """
        if mask is None:
            return {"x_start": x_start}
        # 替换色 c：逐图均值色（蒙区与原图色调一致，只毁局部内容、不漏全局常量）
        c = x_start.mean(dim=(2, 3), keepdim=True)
        w_keep = 1.0 - self.alpha * (1.0 - mask)       # 蒙块处 = 1−alpha
        x_in = w_keep * x_start + (1.0 - w_keep) * c
        return {
            "x_start": x_in,                 # 输入：蒙区向该图均值色按 alpha 混合
            "target_override": x_start,      # 目标：原始干净图（蒙区内容只能靠 z_sem）
        }
