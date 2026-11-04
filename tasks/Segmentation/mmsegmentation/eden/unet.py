"""
Conditional UNet (Denoising Network)
包含论文未写明的工程细节：
  1. zero_module：decoder 输出层初始化为 0，训练更稳定
  2. 双条件注入：time_emb + cond_emb 分别映射为 scale/shift
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def zero_module(module: nn.Module) -> nn.Module:
    """将模块的所有可训练参数初始化为 0。训练初期让 decoder 输出接近 0，避免早期梯度爆炸。"""
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """正弦位置编码，把离散的 timestep t 映射为连续向量。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)  # 3x3 conv 平滑插值后的伪影
        return x


class ConditionalResBlock(nn.Module):
    """
    带双条件注入的 ResBlock。
    time_emb 和 cond_emb 分别通过独立线性层映射为 (scale, shift)，
    在 GroupNorm 之后、SiLU 之前施加。
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        cond_dim: int,
        dropout: float = 0.1,
        use_zero_module: bool = True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        # 第一层
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # Time 条件投影 -> (scale, shift)
        self.time_emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch * 2))

        # Cond 条件投影 -> (scale, shift)
        self.cond_emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_ch * 2))

        # 第二层
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        if use_zero_module:
            self.conv2 = zero_module(self.conv2)

        # Skip
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Time 条件注入
        time_out = self.time_emb_proj(time_emb)
        time_scale, time_shift = time_out.chunk(2, dim=1)
        time_scale = time_scale[:, :, None, None]
        time_shift = time_shift[:, :, None, None]

        # Cond 条件注入
        cond_out = self.cond_emb_proj(cond_emb)
        cond_scale, cond_shift = cond_out.chunk(2, dim=1)
        cond_scale = cond_scale[:, :, None, None]
        cond_shift = cond_shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1 + time_scale) + time_shift
        h = h * (1 + cond_scale) + cond_shift
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.skip(x) + h


class AttentionBlock(nn.Module):
    """简化版空间 Self-Attention。proj 用 zero_module。"""
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_reshape = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_reshape))
        q, k, v = qkv.chunk(3, dim=1)
        q, k, v = q.permute(0, 2, 1), k.permute(0, 2, 1), v.permute(0, 2, 1)
        attn_out, _ = self.attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1)
        attn_out = self.proj(attn_out)
        return x + attn_out.reshape(b, c, h, w)


class ConditionalUNet(nn.Module):
    """
    条件 UNet，作为 DiffAE 的 Decoder（Denoising Network）。
    接受 (x_t, t, cond) 三个输入，输出预测噪声 eps。
    """
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        channel_mult: Tuple[int, ...] = (1, 2, 3, 4),
        cond_dim: int = 512,
        dropout: float = 0.1,
        use_zero_module: bool = True,
        image_size: int = 128,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.channel_mult = channel_mult
        
        n_downsample = len(channel_mult) - 1
        min_divisible = 2 ** n_downsample
        if image_size % min_divisible != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by 2^{n_downsample} = {min_divisible}, "
                f"got remainder {image_size % min_divisible}. "
                f"Either adjust image_size or channel_mult."
            )

        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 输入投影
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        self.down_block_types = []
        skip_chs = [model_channels]
        ch = model_channels
        now_res = image_size

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ConditionalResBlock(
                    ch, out_ch, time_emb_dim, cond_dim, dropout, use_zero_module
                ))
                self.down_block_types.append("res")
                ch = out_ch
                skip_chs.append(ch)
                if now_res in attention_resolutions:
                    self.down_blocks.append(AttentionBlock(ch))
                    self.down_block_types.append("attn")

            if level != len(channel_mult) - 1:
                self.down_blocks.append(Downsample(ch))
                self.down_block_types.append("down")
                skip_chs.append(ch)
                now_res //= 2

        # Middle
        self.mid_block1 = ConditionalResBlock(ch, ch, time_emb_dim, cond_dim, dropout, use_zero_module)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ConditionalResBlock(ch, ch, time_emb_dim, cond_dim, dropout, use_zero_module)

        # 上采样路径
        self.up_blocks = nn.ModuleList()
        self.up_block_types = []
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ConditionalResBlock(
                    ch + skip_chs.pop(), out_ch, time_emb_dim, cond_dim, dropout, use_zero_module
                ))
                self.up_block_types.append('res')
                ch = out_ch
                if now_res in attention_resolutions:
                    self.up_blocks.append( AttentionBlock(ch))
                    self.up_block_types.append('attn')

            if level != len(channel_mult) - 1:
                self.up_blocks.append( Upsample(ch))
                self.up_block_types.append('up')
                now_res *= 2

        # 输出层（zero_module）
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, out_channels, 3, padding=1)) if use_zero_module
            else nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]  噪声图像 x_t
            t: [B]            离散 timestep
            cond: [B, cond_dim] 语义条件
        Returns:
            [B, C, H, W] 预测噪声 eps
        """
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        h = self.input_conv(x)
        hs = [h]

        # Down
        for block_type, block in zip(self.down_block_types, self.down_blocks):
            if block_type == "res":
                h = block(h, t_emb, cond)
                hs.append(h)
            elif block_type == "attn":
                h = block(h)
            elif block_type == "down":
                h = block(h)
                hs.append(h)

        # Middle
        h = self.mid_block1(h, t_emb, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb, cond)

        # Up
        for block_type, block in zip(self.up_block_types, self.up_blocks):
            if block_type == "res":
                h = block(torch.cat([h, hs.pop()], dim=1), t_emb, cond)
            elif block_type == "attn":
                h = block(h)
            elif block_type == "up":
                h = block(h)

        return self.out(h)