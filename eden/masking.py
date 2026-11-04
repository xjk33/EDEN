"""
结构化加噪：选址 + 施加

拆分自 diffae_mixin.py，两个组件各司其职：
  MaskSelector —— 控制结构化加噪方法：从下游任务敏感度热力图选址
                  （random 均匀 / hot 蒙高敏感区 / cold 蒙低敏感区，
                  Gumbel 无放回采样，支持动态幅度）
  MaskApplier  —— 控制加噪的施加：软蒙（加噪前把蒙区向数据集均值色混合）
                  或硬蒙（先加噪后把蒙区置零），输出 sampler.training_losses 的输入参数

两种施加方式的 loss 目标始终是完整 x0/ε：蒙区内容的唯一信息源是 z_sem，
以此逼迫 encoder 把蒙区信息压进语义码。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskSelector:
    """
    选址：敏感度热力图 -> 0/1 keep-mask [B,1,H,W]（1=保留，0=被选中蒙住）。
    每张图独立无放回采样（argsort + rank 技巧，支持每行不同 k）。
    """

    def __init__(
        self,
        patch: int = 16,           # 蒙块边长（像素）
        mode: str = "random",      # 'random' | 'hot' | 'cold'（后两者需热力图）
        temp: float = 1.0,         # 热力采样温度：<1 更平，>1 更尖
        ratio: float = 0.0,        # 固定蒙块比例 0~1，0 = 关闭
        dynamic: bool = False,     # True = 每图每步 r~U[0, ratio_max]；False = 固定 ratio
        ratio_max: float = 0.3,    # 动态幅度上限（文献剂量最优区上沿）
    ):
        assert mode in ('random', 'hot', 'cold'), f"unknown mask mode: {mode}"
        self.patch = patch
        self.mode = mode
        self.temp = temp
        self.ratio = ratio
        self.dynamic = dynamic
        self.ratio_max = ratio_max

    @property
    def enabled(self) -> bool:
        return self.ratio > 0 or self.dynamic

    def select(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heatmap: [B,1,H,W] 敏感度热力图（random 模式内容随意，只取形状）
        Returns:
            keep-mask [B,1,H,W]，1=保留，0=蒙住
        """
        B = heatmap.shape[0]
        p = self.patch
        H, W = heatmap.shape[-2:]
        g_h, g_w = max(1, H // p), max(1, W // p)
        n = g_h * g_w

        if self.dynamic:
            r = torch.rand(B, device=heatmap.device) * self.ratio_max   # U[0, rmax]
            k = (r * n).round().long().clamp(min=1)                     # 每图块数
        else:
            k = torch.full((B,), max(1, int(round(self.ratio * n))),
                           device=heatmap.device, dtype=torch.long)

        if self.mode == 'random':
            score = torch.rand(B, n, device=heatmap.device)
        else:
            patch_score = F.adaptive_avg_pool2d(heatmap, (g_h, g_w)).reshape(B, n).clamp_min(0)
            if self.temp != 1.0:
                patch_score = patch_score ** self.temp
            if self.mode == 'cold':
                patch_score = patch_score.amax(dim=1, keepdim=True) - patch_score
            u = torch.rand_like(patch_score).clamp_min(1e-12)
            score = torch.log(patch_score.clamp_min(1e-12)) + torch.log(-torch.log(u))  # Gumbel

        order = score.argsort(dim=1, descending=True)            # [B, n]
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(n, device=heatmap.device).unsqueeze(0).expand(B, n))
        sel = rank < k.view(B, 1)                                # [B, n] 每图前 k_i 个被选中

        keep = (~sel).float().view(B, 1, g_h, g_w)
        mask = keep.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
        return mask[:, :, :H, :W]


class MaskApplier(nn.Module):
    """
    施加：把 keep-mask 翻译成 sampler.training_losses 的输入参数。
      软蒙（soft=True） ：加噪【前】把蒙区向数据集均值色混合（alpha 控强度），
                         输入是混合图、目标仍是原始干净图（target_override），分布连续无跳变；
      硬蒙（soft=False）：完整 x_t 加噪【后】乘 0/1 蒙块，蒙区内容与噪声同时归零。
    """

    def __init__(self, soft: bool = False, alpha: float = 1.0):
        super().__init__()
        self.soft = soft
        self.alpha = alpha
        # 软蒙目标色：ImageNet 均值在 [-1,1] 空间的取值（恰好等于 padding 的颜色）
        _mu = [(123.675 / 255.0) * 2 - 1, (116.28 / 255.0) * 2 - 1, (103.53 / 255.0) * 2 - 1]
        self.register_buffer('mask_mean', torch.tensor(_mu).view(1, 3, 1, 1))

    def transform(self, x_start: torch.Tensor,
                  mask: "torch.Tensor | None") -> dict:
        """
        返回 sampler.training_losses 的输入 kwargs。
        mask=None（未启用蒙块）时原样透传。
        """
        if mask is None:
            return {"x_start": x_start}
        if self.soft:
            w_keep = 1.0 - self.alpha * (1.0 - mask)       # 蒙块处 = 1-alpha
            x_in = w_keep * x_start + (1.0 - w_keep) * self.mask_mean.to(x_start.device)
            return {
                "x_start": x_in,                 # 输入：带均值色块的混合图
                "target_override": x_start,      # 目标：原始干净图（蒙区内容只能靠 z_sem）
            }
        return {
            "x_start": x_start,
            "post_mask": mask,                   # 加噪后蒙：蒙区输入为 0
        }
