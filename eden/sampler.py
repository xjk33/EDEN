"""
DDIM / DDPM Sampler
包含论文未写明的工程细节：
  1. Spaced Diffusion（跳步采样）
  2. clip_denoised（限制 x0 预测范围）
  3. pred_xstart_detach（训练时 detach x0 分支）
"""
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union


def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int) -> np.ndarray:
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        return np.linspace(scale * 0.0001, scale * 0.02, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise ValueError(f"Unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar, max_beta: float = 0.999) -> np.ndarray:
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


def space_timesteps(num_timesteps: int, section_counts: Union[str, List[int]]) -> List[int]:
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            section_counts = [int(section_counts[4:])]
        else:
            raise ValueError(f"Unsupported spaced format: {section_counts}")

    if len(section_counts) == 1:
        num_sections = section_counts[0]
        stride = num_timesteps // num_sections
        used = set(range(0, num_timesteps, stride))
        if (num_timesteps - 1) not in used:
            used.add(num_timesteps - 1)
        return sorted(list(used))
    else:
        raise NotImplementedError("Only uniform spaced timesteps are supported now.")


class DDIMSampler:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        use_timesteps: Optional[List[int]] = None,
        clip_denoised: bool = True,
        pred_xstart_detach: bool = True,
        device: str = None,
        custom_timestep_sampler: Optional[List[int]] = None,
        predict_x0: bool = False,   # True = 网络直接预测 x0（v2：单次 loss 全包）
    ):
        self.num_timesteps = num_timesteps
        self.predict_x0 = predict_x0
        self.clip_denoised = clip_denoised
        self.pred_xstart_detach = pred_xstart_detach
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- 自定义 timestep 范围（原始 timestep 坐标）----
        self.use_custom_timestep_sampler = False
        self.custom_timestep_min = 0
        self.custom_timestep_max = num_timesteps
        self.custom_timestep_indices = None   # 存储有效的 spaced 索引
        self.num_custom_timesteps = 0

        if custom_timestep_sampler is not None:
            assert len(custom_timestep_sampler) == 2, "custom_timestep_sampler must be [min, max]"
            self.custom_timestep_min = custom_timestep_sampler[0]
            self.custom_timestep_max = custom_timestep_sampler[1]
            self.use_custom_timestep_sampler = True

        # 预计算所有扩散系数
        betas = get_named_beta_schedule(beta_schedule, num_timesteps)
        self.betas = torch.from_numpy(betas).float().to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )
        self.alphas_cumprod_next = torch.cat(
            [self.alphas_cumprod[1:], torch.tensor([0.0], device=device)]
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Spaced Diffusion 映射
        if use_timesteps is None:
            use_timesteps = list(range(num_timesteps))
        self.use_timesteps = sorted(list(use_timesteps))
        self.num_sample_steps = len(self.use_timesteps)
        self.timestep_map = {i: self.use_timesteps[i] for i in range(self.num_sample_steps)}

        # ---- 关键：在 spaced 映射后，筛选落在 [min, max] 内的索引 ----
        if self.use_custom_timestep_sampler:
            self.custom_timestep_indices = [
                i for i, ts in enumerate(self.use_timesteps)
                if self.custom_timestep_min <= ts <= self.custom_timestep_max
            ]
            self.num_custom_timesteps = len(self.custom_timestep_indices)
            if self.num_custom_timesteps == 0:
                raise ValueError(
                    f"No spaced timesteps fall in range "
                    f"[{self.custom_timestep_min}, {self.custom_timestep_max}]. "
                    f"Available spaced timesteps: {self.use_timesteps[:10]}...{self.use_timesteps[-10:]}"
                )

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
        out = arr[t].float()
        while len(out.shape) < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def _predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t: torch.Tensor, t: torch.Tensor, pred_xstart: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_mean_variance(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, model_kwargs: Optional[Dict] = None):
        if model_kwargs is None:
            model_kwargs = {}
        model_output = model(x_t, t, **model_kwargs)
        if self.predict_x0:
            pred_xstart = model_output
            pred_eps = self._predict_eps_from_xstart(x_t, t, pred_xstart)
        else:
            pred_eps = model_output
            pred_xstart = self._predict_xstart_from_eps(x_t, t, pred_eps)

        if self.clip_denoised:
            pred_xstart = torch.clamp(pred_xstart, -1.0, 1.0)
        if self.pred_xstart_detach:
            pred_xstart = pred_xstart.detach()

        return pred_eps, pred_xstart

    def ddim_sample(
        self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor,
        model_kwargs: Optional[Dict] = None, eta: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        pred_eps, pred_xstart = self.p_mean_variance(model, x_t, t, model_kwargs)
        alpha_bar = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev, t, x_t.shape)

        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        noise = torch.randn_like(x_t)
        mean_pred = pred_xstart * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * pred_eps
        nonzero_mask = ((t != 0).float()).view(-1, *([1] * (len(x_t.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": pred_xstart}

    def ddim_sample_loop(
        self, model: nn.Module, noise: Optional[torch.Tensor] = None,
        shape: Optional[tuple] = None, model_kwargs: Optional[Dict] = None,
        device: str = None, progress: bool = False,
    ) -> torch.Tensor:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if noise is not None:
            x = noise
        else:
            assert shape is not None
            x = torch.randn(*shape, device=device)

        indices = list(range(self.num_sample_steps))[::-1]
        if progress:
            from tqdm import tqdm
            indices = tqdm(indices, desc="DDIM Sampling")

        for i in indices:
            t = torch.full((x.shape[0],), self.timestep_map[i], device=device, dtype=torch.long)
            with torch.no_grad():
                out = self.ddim_sample(model, x, t, model_kwargs=model_kwargs)
                x = out["sample"]
        return x

    def ddim_reverse_sample(
        self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor,
        model_kwargs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        pred_eps, pred_xstart = self.p_mean_variance(model, x_t, t, model_kwargs)
        alpha_bar_next = self._extract(self.alphas_cumprod_next, t, x_t.shape)
        mean_pred = pred_xstart * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * pred_eps
        return {"sample": mean_pred, "pred_xstart": pred_xstart}

    def ddim_reverse_sample_loop(
        self, model: nn.Module, x0: torch.Tensor,
        model_kwargs: Optional[Dict] = None, device: str = None,
    ) -> torch.Tensor:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        x = x0
        indices = list(range(self.num_sample_steps))
        for i in indices:
            t = torch.full((x.shape[0],), self.timestep_map[i], device=device, dtype=torch.long)
            with torch.no_grad():
                out = self.ddim_reverse_sample(model, x, t, model_kwargs=model_kwargs)
                x = out["sample"]
        return x

    def training_losses(
        self, model: nn.Module, x_start: torch.Tensor,
        model_kwargs: Optional[Dict] = None,
        post_mask: Optional[torch.Tensor] = None,   # v2：加噪后再蒙（蒙区输入为 0）
        target_override: Optional[torch.Tensor] = None,  # 软蒙：输入是混合图，目标仍是原始干净图
        t_probs: Optional[torch.Tensor] = None,     # t 采样分布（use_timesteps 上的概率）；None = 均匀
    ) -> Dict[str, Any]:
        """
        训练损失。返回 loss / noise / x_t / t / pred。
        predict_x0=False 时 pred = ε̂；predict_x0=True 时 pred = x̂0（loss 目标）。
        post_mask 给定（v2 空间蒙块）：完整 x_t 加噪后再乘 mask，蒙区输入为 0，
        目标仍是完整 x_start —— 蒙区唯一信息源是 cond（z_sem）。
        """
        B = x_start.shape[0]
        device = x_start.device

        if t_probs is not None:
            # 按 t_weight 推导的分布采样（与 loss 权重化期望等价，但每步满强度）
            t_idx = torch.multinomial(t_probs.to(device), B, replacement=True)
        elif self.use_custom_timestep_sampler:
            # 在筛选后的 spaced 索引中均匀采样
            local_idx = torch.randint(0, self.num_custom_timesteps, (B,), device=device)
            t_idx = torch.tensor(
                [self.custom_timestep_indices[i] for i in local_idx],
                device=device, dtype=torch.long,
            )
        else:
            t_idx = torch.randint(0, self.num_sample_steps, (B,), device=device)

        t = torch.tensor(
            [self.timestep_map[idx.item()] for idx in t_idx],
            device=device, dtype=torch.long,
        )

        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        if post_mask is not None:
            x_t = x_t * post_mask          # v2：先加噪后蒙，蒙区内容与噪声同时归零
        model_output = model(x_t, t, **(model_kwargs or {}))

        # predict_x0=True 时目标是干净图 x0（v2），否则是噪声 ε（v1）。
        # 软蒙时输入是混合图、目标用 target_override 传原始干净图（蒙区内容只能靠 z_sem）。
        target = target_override if target_override is not None else \
            (x_start if self.predict_x0 else noise)
        loss = nn.functional.mse_loss(model_output, target, reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        return {"loss": loss, "noise": noise, "x_t": x_t, "t": t, "pred": model_output}