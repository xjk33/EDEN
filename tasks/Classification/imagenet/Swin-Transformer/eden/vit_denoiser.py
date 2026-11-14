"""
DiT 式条件 ViT 去噪器（i-DAE 路线）：替代强 UNet 的弱仪器。

动机：DiffAE 框架已把表征解耦到任务 encoder（z_sem），denoiser 不承担
表征学习，只是施加去噪约束的仪器——因此可以单纯地把仪器做小，把约束压力
定向转移到 z_sem 上（MAE 轻量解码器同族设计）。强 UNet 能在低中 t 区间
绕过条件单走去噪，D_t 被压没；弱仪器离开条件做不好，D_t 大且干净。
注意甜区是倒 U：仪器太弱时残差全是它自身的无能，D_t 变成噪声——
换尺寸后对照 D_t 曲线（整体抬升 = 压力转移成功；塌平/噪声化 = 太弱）。

设计来源（全部为 DiT / i-DAE 的标准做法，无自造成分）：
  - 骨架 = 标准 ViT：patchify（patch 16 直接吃像素）+ 可学习 pos emb
    + N 个 pre-LN block + unpatchify
  - 条件注入 = adaLN-Zero（DiT）：c = t_emb + z_sem_emb，每个 block 的
    scale/shift/gate 由 c 经两层 MLP 回归，gate 与输出头零初始化
    （等价于 UNet 版的 zero_module：训练初期输出 ≈ 0，zero 条件 ≈ 去条件）
  - 条件 MLP 隐层压到 1/4（i-DAE 明确报告无精度损失的减参技巧）
  - 预测目标 = x0（i-DAE：predict clean data；与结构化掩码的"单次 loss
    全包"直接兼容：可见区去噪 + 蒙区靠 z_sem 重构，同一参数化）

规模预设（i-DAE 的规模是对照 ViT-L 来的，这里不做对齐——表征已在 encoder，
仪器只做小；显式传 dim/depth/num_heads 可覆盖预设）：
  'ti' = ViT-Ti/16（dim 192, 12 层, 3 头）≈ 6.7M   —— 默认，比原 UNet(38.2M) 弱约 5.7×
  's'  = ViT-S/16 （dim 384, 12 层, 6 头）≈ 25.8M —— 对照档
  'b'  = ViT-B/16 （dim 768, 12 层, 12 头）≈ 100M —— 对照上限（一般不需要）

接口与 ConditionalUNet 完全一致：forward(x[B,C,H,W], t[B], cond[B,cond_dim])
-> [B,C,H,W]，输出语义（eps 还是 x0）由 sampler.predict_x0 决定。
"""
import torch
import torch.nn as nn

from .unet import timestep_embedding

DIT_PRESETS = {
    'ti': dict(dim=192, depth=12, num_heads=3),    # ~6.7M
    's':  dict(dim=384, depth=12, num_heads=6),    # ~25.8M
    'b':  dict(dim=768, depth=12, num_heads=12),   # ~100M
}


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN 调制：x * (1 + scale) + shift，shift/scale 为 [B, dim]。"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _gelu_tanh() -> nn.Module:
    """GELU(tanh 近似)；approximate 参数为 torch 1.12 新增，
    旧环境（如分类任务的 py3.7 + torch≤1.11）退化为精确 GELU——
    两者最大偏差约 1e-3，对去噪仪器无实质影响。"""
    try:
        return nn.GELU(approximate="tanh")
    except TypeError:
        return nn.GELU()


class DiTBlock(nn.Module):
    """标准 ViT block + adaLN-Zero 条件注入（DiT 做法）。"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 adaln_hidden: int = None):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            _gelu_tanh(),
            nn.Linear(hidden, dim),
        )
        # 条件 MLP：两层，隐层压到 1/4（i-DAE 减参技巧），输出 6 路调制量
        adaln_hidden = adaln_hidden or dim
        self.adaln = nn.Sequential(
            nn.Linear(dim, adaln_hidden),
            nn.SiLU(),
            nn.Linear(adaln_hidden, dim * 6),
        )
        # adaLN-Zero：调制输出层零初始化 -> 训练初期 gate=0，block 恒等
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaln(c).chunk(6, dim=1)

        h = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.qkv(h).reshape(B, N, 3, self.num_heads, D // self.num_heads) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        h = nn.functional.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, N, D)
        x = x + gate_msa.unsqueeze(1) * self.attn_proj(h)

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ConditionalDiT(nn.Module):
    """
    条件 DiT 去噪器（ViT-S 级），作为 DiffAE 的 Denoising Network。
    接受 (x_t, t, cond) 三个输入，输出与输入同形的预测（x0 或 eps，
    由外层 sampler.predict_x0 决定）。
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 128,
        patch_size: int = 16,
        size: str = 'ti',             # DIT_PRESETS 预设档；显式传 dim/depth/num_heads 可逐项覆盖
        dim: int = None,
        depth: int = None,
        num_heads: int = None,
        mlp_ratio: float = 4.0,
        cond_dim: int = 512,
        adaln_hidden_mult: float = 0.25,   # i-DAE：条件 MLP 隐层 = 1/4
    ):
        super().__init__()
        if size not in DIT_PRESETS:
            raise ValueError(f"unknown DiT size preset: {size!r}, "
                             f"expected one of {list(DIT_PRESETS)}")
        preset = DIT_PRESETS[size]
        dim = dim or preset['dim']
        depth = depth if depth is not None else preset['depth']
        num_heads = num_heads or preset['num_heads']
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size}).")
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.out_channels = in_channels
        grid = image_size // patch_size
        self.num_patches = grid * grid

        # patchify / unpatchify
        self.patch_embed = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 条件嵌入：c = t_emb + cond_emb（DiT 的 t + label 求和做法）
        self.t_embedder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.cond_embedder = nn.Linear(cond_dim, dim)

        adaln_hidden = int(dim * adaln_hidden_mult)
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio, adaln_hidden) for _ in range(depth)
        ])

        # 输出头：adaLN（无 gate）+ 线性到 patch 像素，两者均零初始化
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )
        self.final_proj = nn.Linear(dim, patch_size * patch_size * self.out_channels)
        nn.init.zeros_(self.final_adaln[-1].weight)
        nn.init.zeros_(self.final_adaln[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]  噪声图像 x_t（可能已被掩码替换退化）
            t: [B]            离散 timestep
            cond: [B, cond_dim] 语义条件 z_sem（零向量 = 去条件探针）
        Returns:
            [B, C, H, W] 预测（x0 或 eps，由 sampler.predict_x0 决定）
        """
        B, C, H, W = x.shape
        p = self.patch_size
        if H != self.image_size or W != self.image_size:
            raise ValueError(
                f"ConditionalDiT expects {self.image_size}x{self.image_size} input, "
                f"got {H}x{W}.")

        c = self.t_embedder(timestep_embedding(t, self.dim)) + self.cond_embedder(cond)

        h = self.patch_embed(x).flatten(2).transpose(1, 2)   # [B, N, dim]
        h = h + self.pos_embed
        for block in self.blocks:
            h = block(h, c)

        shift, scale = self.final_adaln(c).chunk(2, dim=1)
        h = modulate(self.final_norm(h), shift, scale)
        h = self.final_proj(h)                               # [B, N, p*p*C]

        # unpatchify: [B, N, p*p*C] -> [B, C, H, W]
        g = H // p
        h = h.reshape(B, g, g, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return h
