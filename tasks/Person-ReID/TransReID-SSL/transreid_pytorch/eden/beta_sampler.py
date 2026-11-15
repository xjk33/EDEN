"""
ANTS 复现：可学习 Beta 时间步采样器 + REINFORCE
（Kim et al., "Adaptive Non-uniform Timestep Sampling for Accelerating
 Diffusion Model Training", CVPR 2025, arXiv:2411.09998 的简化复现）

机制（与原文一致）：
  - 采样器 π_φ：小 CNN 吃 x0，输出两个正标量 (a, b)，t ~ Beta(a, b)；
  - reward：本次去噪器更新前后，扩散 loss 在少量代表时间步上的下降量
    （Δ = L_before - L_after，同批图、同 cond、同噪声 CRN 配对）；
  - 更新：REINFORCE（likelihood-ratio）梯度上升 + Beta 熵正则，
    队列平滑 reward（|Q|=20），Adam(lr=1e-3)；
  - 初始化 a=b=1，Beta(1,1) 即均匀分布 —— 与均匀采样基线从同一点出发。

与原文的偏差（有意为之，config 注释同样声明）：
  1. 代表时间步集 S 固定（默认 [200,500,800]，均在 ddim100 网格上），
     不做原文基于队列的特征选择；
  2. reward 每 interval 步测一次（原文 f_s=40，沿用该值）；
  3. 训练目标保持 EDEN 的 x0-MSE 不变（原文用 c_t 加权的 L_VLB 训练），
     reward 也在同一 x0-MSE 上测 —— 保证"采样分布"是与基线的唯一变量；
  4. φ 优化器状态与 reward 队列不进 checkpoint（resume 后自动重新热启动）。
"""
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class ANTSampler(nn.Module):
    """Beta(a, b) 时间步采样器：CNN(x0) -> (a, b) -> t（吸附到训练网格）。"""

    def __init__(self, in_channels: int = 3, hidden: int = 256,
                 num_timesteps: int = 1000, use_timesteps=None,
                 lr: float = 1e-3, entropy_coef: float = 1e-2,
                 interval: int = 40, queue_size: int = 20,
                 rep_timesteps=(200, 500, 800)):
        super().__init__()
        self.entropy_coef = entropy_coef
        self.interval = interval
        self.rep_timesteps = list(rep_timesteps)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.SiLU(),      # /2
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),              # /4
            nn.Conv2d(128, hidden, 4, 2, 1), nn.SiLU(),          # /8
            nn.Conv2d(hidden, hidden, 4, 2, 1), nn.SiLU(),       # /16
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        # 末层小权重 + bias=softplusinv(1)：初始 a=b=1 -> Beta(1,1)=均匀分布
        head = self.net[-1]
        nn.init.normal_(head.weight, std=0.01)
        nn.init.constant_(head.bias, 0.5413)

        grid = list(range(num_timesteps)) if use_timesteps is None \
            else sorted(int(t) for t in use_timesteps)
        self.register_buffer('grid', torch.tensor(grid, dtype=torch.float32))
        self.register_buffer('grid_long', torch.tensor(grid, dtype=torch.long))

        # REINFORCE 的簿记（非 nn 状态，不进 ckpt）
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.reward_queue = deque(maxlen=queue_size)

    def forward(self, x: torch.Tensor):
        a, b = F.softplus(self.net(x)).unbind(-1)
        return a + 1e-3, b + 1e-3

    def sample(self, x: torch.Tensor):
        """
        采一个 batch 的 t。前向带图（log_prob 要给 REINFORCE 回传），
        采样与网格吸附无梯度。
        Returns: t [B] long / logp [B] / ent [B] / (a,b) 监控用
        """
        a, b = self.forward(x)
        dist = torch.distributions.Beta(a, b)
        u = dist.sample()                                   # (0,1)，无梯度
        logp = dist.log_prob(u.clamp(1e-6, 1.0 - 1e-6))
        ent = dist.entropy()
        with torch.no_grad():
            g = self.grid
            t_cont = u.to(g.device) * float(g[-1])
            idx = torch.searchsorted(g, t_cont).clamp(max=len(g) - 1)
            left = (idx - 1).clamp(min=0)
            nearer_left = (t_cont - g[left]).abs() < (g[idx] - t_cont).abs()
            idx = torch.where(nearer_left, left, idx)
            t = self.grid_long.to(x.device)[idx]
        return t, logp, ent, (a.detach(), b.detach())

    # ------------------------------------------------------------------
    # REINFORCE 更新（由 mixin 在去噪器 step 之后调用）
    # ------------------------------------------------------------------
    def reinforce_update(self, logp: torch.Tensor, ent: torch.Tensor,
                         reward: float) -> None:
        self.reward_queue.append(float(reward))
        D = sum(self.reward_queue) / len(self.reward_queue)   # 队列平滑
        loss_phi = -(D * logp.mean()) - self.entropy_coef * ent.mean()
        self.opt.zero_grad()
        loss_phi.backward()
        self.opt.step()
