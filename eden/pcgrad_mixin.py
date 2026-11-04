"""
PCGrad Mixin（重构版）

支持四种投影粒度：
  - global:   全局投影（默认）
  - batch:    每个 sample 独立投影（利用 per-sample 扩散 loss）
  - task:     每个下游任务独立投影（多任务场景）
  - batch_task: 每个 (sample, task) 对独立投影（最细粒度）

设计原则：
  - 只提供梯度投影的数学工具和 backward 流程。
  - 不假设任何框架的 train_step 签名。
  - 当前为串行实现，接口预留并行化扩展。
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from collections import defaultdict


class PCGradMixin(nn.Module):
    """
    PCGrad 功能 Mixin。

    用法：
        class MyModel(PCGradMixin, SomeBaseModel):
            def train_step(self, data, optim_wrapper):
                losses = self.loss(data)
                task_loss = losses['task']
                diff_loss = losses['diff']
                diff_per_sample = losses['diff_per_sample']
                self.pcgrad_backward(
                    task_loss, diff_loss, optim_wrapper,
                    mode='batch',
                    diff_loss_per_sample=diff_per_sample,
                )
                return losses
    """

    def __init__(
        self,
        *args,
        pcgrad_enabled: bool = True,
        pcgrad_lambda_d: float = 1.0,
        pcgrad_encoder_names: Tuple[str, ...] = ("backbone",),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pcgrad_enabled = pcgrad_enabled
        self.pcgrad_lambda_d = pcgrad_lambda_d
        self.pcgrad_encoder_names = pcgrad_encoder_names

    def _is_encoder_param(self, name: str) -> bool:
        """判断参数名是否属于 encoder。"""
        return any(enc in name for enc in self.pcgrad_encoder_names)

    def _pcgrad_project(self, g_task: torch.Tensor, g_diffuse: torch.Tensor) -> torch.Tensor:
        """
        PCGrad 核心投影公式：
            ĝ = g_T + λ_D * (g_D^⊥ + max(0, g_D^∥))
        Inverse test:
            ĝ = g_T + λ_D * (g_D^⊥ + min(0, g_D^∥))
        """
        dot = torch.sum(g_task * g_diffuse)
        if dot >= 0:
            return g_task + self.pcgrad_lambda_d * g_diffuse

        denom = torch.sum(g_task ** 2) + 1e-8
        g_parallel = (dot / denom) * g_task
        g_orthogonal = g_diffuse - g_parallel
        return g_task + self.pcgrad_lambda_d * g_orthogonal

    # ------------------------------------------------------------------
    # 主入口：根据 mode 分发
    # ------------------------------------------------------------------
    def pcgrad_backward(
        self,
        task_loss: torch.Tensor,
        diff_loss: torch.Tensor,
        optim_wrapper,
        mode: str = "global",
        task_losses: Optional[Dict[str, torch.Tensor]] = None,
        diff_loss_per_sample: Optional[torch.Tensor] = None,
        task_losses_per_sample: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        执行 PCGrad 的完整 backward + 投影 + step。

        Args:
            task_loss: 全局下游任务 loss（标量）。mode="global" / "batch" 时使用。
            diff_loss: 全局扩散 loss（标量）。mode="global" / "task" 时使用。
            optim_wrapper: mmengine OptimWrapper。
            mode: "global" | "batch" | "task" | "batch_task"。
            task_losses: Dict[str, Tensor]，每个 task 的标量 loss。mode="task" / "batch_task" 时使用。
            diff_loss_per_sample: [B] Tensor，per-sample 扩散 loss。mode="batch" / "batch_task" 时使用。
            task_losses_per_sample: Dict[str, [B] Tensor]，per-sample task loss。mode="batch_task" 时使用。

        注意：当前为串行实现。未来可替换为并行版（torch.autograd.grad + vmap），
        接口保持不变。
        """
        if not self.pcgrad_enabled:
            optim_wrapper.zero_grad()
            total_loss = task_loss + self.pcgrad_lambda_d * diff_loss
            optim_wrapper.backward(total_loss)
            optim_wrapper.step()
            optim_wrapper.zero_grad()
            return

        if mode == "global":
            self._pcgrad_backward_global(task_loss, diff_loss, optim_wrapper)
        elif mode == "batch":
            assert diff_loss_per_sample is not None, "batch mode 需要 diff_loss_per_sample ([B])"
            self._pcgrad_backward_batch(task_loss, diff_loss_per_sample, optim_wrapper)
        elif mode == "task":
            assert task_losses is not None, "task mode 需要 task_losses (dict of scalars)"
            self._pcgrad_backward_task(task_losses, diff_loss, optim_wrapper)
        elif mode == "batch_task":
            assert task_losses_per_sample is not None and diff_loss_per_sample is not None, \
                "batch_task mode 需要 task_losses_per_sample 和 diff_loss_per_sample"
            self._pcgrad_backward_batch_task(task_losses_per_sample, diff_loss_per_sample, optim_wrapper)
        else:
            raise ValueError(f"Unknown pcgrad mode: {mode}")

    # ------------------------------------------------------------------
    # mode = "global"
    # ------------------------------------------------------------------
    def _pcgrad_backward_global(self, task_loss, diff_loss, optim_wrapper):
        optim_wrapper.zero_grad()
        optim_wrapper.backward(task_loss, retain_graph=True)

        task_grads = {}
        for name, param in self.named_parameters():
            if param.grad is not None and self._is_encoder_param(name):
                task_grads[name] = param.grad.clone()

        optim_wrapper.backward(diff_loss)

        for name, param in self.named_parameters():
            if name in task_grads and param.grad is not None:
                g_task = task_grads[name]
                g_diff = param.grad - g_task
                param.grad.copy_(self._pcgrad_project(g_task, g_diff))

        optim_wrapper.step()
        optim_wrapper.zero_grad()

    # ------------------------------------------------------------------
    # mode = "batch"：每个 sample 独立投影
    # ------------------------------------------------------------------
    def _pcgrad_backward_batch(self, task_loss, diff_loss_per_sample, optim_wrapper):
        """
        分别对每个 sample 的 diff 梯度与全局 task 梯度做投影，最后平均。

        流程：
          1. backward(task_loss) 得到全局 g_task
          2. 对每个 sample i：
               - backward(diff_loss[i]) 累加得到 g_task + g_diff_i
               - 对 encoder 参数：project(g_task, g_diff_i)，保存
               - 恢复 g_task 用于下一次 loop
          3. 所有参数取平均后写回，执行 step

        注意：当前为串行实现（1 + B 次 backward）。
        未来可替换为并行版：torch.autograd.grad 分别求 B 个独立梯度后批量投影。
        """
        B = diff_loss_per_sample.shape[0]

        # 1. 全局 task backward
        optim_wrapper.zero_grad()
        optim_wrapper.backward(task_loss, retain_graph=True)
        task_grads = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                task_grads[name] = param.grad.clone()

        # 2. Loop over batch samples
        encoder_grads = defaultdict(list)
        other_grads = defaultdict(list)

        for i in range(B):
            # 关键修复：除了最后一个 sample，都保留计算图
            optim_wrapper.backward(
                diff_loss_per_sample[i],
                retain_graph=(i < B - 1)
            )

            for name, param in self.named_parameters():
                if param.grad is not None:
                    if self._is_encoder_param(name):
                        g_task = task_grads[name]
                        g_diff = param.grad - g_task
                        g_hat = self._pcgrad_project(g_task, g_diff)
                        encoder_grads[name].append(g_hat)
                    else:
                        other_grads[name].append(param.grad.clone())

            # 恢复 task 梯度用于下一次 loop
            if i < B - 1:
                optim_wrapper.zero_grad()
                for name, param in self.named_parameters():
                    if name in task_grads:
                        param.grad = task_grads[name].clone()

        # 3. 平均后写回所有参数
        optim_wrapper.zero_grad()
        for name, param in self.named_parameters():
            if name in encoder_grads:
                param.grad = torch.stack(encoder_grads[name]).mean(dim=0)
            elif name in other_grads:
                param.grad = torch.stack(other_grads[name]).mean(dim=0)

        # 4. step
        optim_wrapper.step()
        optim_wrapper.zero_grad()

    # ------------------------------------------------------------------
    # mode = "task"：每个下游任务独立投影
    # ------------------------------------------------------------------
    def _pcgrad_backward_task(self, task_losses, diff_loss, optim_wrapper):
        """
        分别对每个 task 的梯度与 diff 梯度做投影，最后平均。

        流程：
          1. 对每个 task k：
               - backward(task_loss_k, retain_graph=True) 得到 g_task_k
               - backward(diff_loss) 累加得到 g_task_k + g_diff
               - 对 encoder 参数：project(g_task_k, g_diff)，保存
          2. 所有参数取平均后写回，执行 step

        注意：当前为串行实现（2K 次 backward）。
        未来可替换为并行版：torch.autograd.grad 分别求 K 个独立 task 梯度后批量投影。
        """
        encoder_grads = defaultdict(list)
        other_grads = defaultdict(list)

        for idx, (task_name, task_loss_k) in enumerate(task_losses.items()):
            # task backward
            optim_wrapper.zero_grad()
            optim_wrapper.backward(task_loss_k, retain_graph=True)
            task_grads_k = {}
            for name, param in self.named_parameters():
                if param.grad is not None:
                    task_grads_k[name] = param.grad.clone()

            # diff backward（累加）
            # 关键修复：除了最后一个 task，都保留计算图
            is_last = (idx == len(task_losses) - 1)
            optim_wrapper.backward(diff_loss, retain_graph=not is_last)

            for name, param in self.named_parameters():
                if param.grad is not None:
                    if self._is_encoder_param(name):
                        g_task = task_grads_k[name]
                        g_diff = param.grad - g_task
                        g_hat = self._pcgrad_project(g_task, g_diff)
                        encoder_grads[name].append(g_hat)
                    else:
                        other_grads[name].append(param.grad.clone())

        # 平均后写回
        optim_wrapper.zero_grad()
        for name, param in self.named_parameters():
            if name in encoder_grads:
                param.grad = torch.stack(encoder_grads[name]).mean(dim=0)
            elif name in other_grads:
                param.grad = torch.stack(other_grads[name]).mean(dim=0)

        optim_wrapper.step()
        optim_wrapper.zero_grad()

    # ------------------------------------------------------------------
    # mode = "batch_task"：每个 (sample, task) 对独立投影
    # ------------------------------------------------------------------
    def _pcgrad_backward_batch_task(
        self, task_losses_per_sample, diff_loss_per_sample, optim_wrapper
    ):
        """
        最细粒度：每个 (sample, task) 对独立投影，最后平均。

        流程：
          1. 对每个 task k：
               - backward(task_loss_k.sum()) 得到全局 g_task_k
               - 对每个 sample i：
                    - backward(diff_loss[i]) 累加
                    - 对 encoder 参数：project(g_task_k, g_diff_i)，保存
                    - 恢复 g_task_k
          2. 所有参数取平均后写回，执行 step

        开销：K * (1 + B) 次 backward。K=task 数，B=batch size。
        注意：当前为串行实现。未来可替换为并行版。
        """
        B = diff_loss_per_sample.shape[0]
        encoder_grads = defaultdict(list)
        other_grads = defaultdict(list)

        for task_name, task_loss_k in task_losses_per_sample.items():
            # 全局 task backward（聚合所有 sample）
            optim_wrapper.zero_grad()
            optim_wrapper.backward(task_loss_k.sum(), retain_graph=True)
            task_grads_k = {}
            for name, param in self.named_parameters():
                if param.grad is not None:
                    task_grads_k[name] = param.grad.clone()

            for i in range(B):
                # 关键修复：除了最后一个 sample，都保留计算图
                optim_wrapper.backward(
                    diff_loss_per_sample[i],
                    retain_graph=(i < B - 1)
                )

                for name, param in self.named_parameters():
                    if param.grad is not None:
                        if self._is_encoder_param(name):
                            g_task = task_grads_k[name]
                            g_diff = param.grad - g_task
                            g_hat = self._pcgrad_project(g_task, g_diff)
                            encoder_grads[name].append(g_hat)
                        else:
                            other_grads[name].append(param.grad.clone())

                # 恢复 task 梯度
                if i < B - 1:
                    optim_wrapper.zero_grad()
                    for name, param in self.named_parameters():
                        if name in task_grads_k:
                            param.grad = task_grads_k[name].clone()

        # 平均后写回
        optim_wrapper.zero_grad()
        for name, param in self.named_parameters():
            if name in encoder_grads:
                param.grad = torch.stack(encoder_grads[name]).mean(dim=0)
            elif name in other_grads:
                param.grad = torch.stack(other_grads[name]).mean(dim=0)

        optim_wrapper.step()
        optim_wrapper.zero_grad()