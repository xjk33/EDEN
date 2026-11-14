# EDEN 单步冒烟脚本：绕过分布式与数据加载，单进程直接跑
#   建模 -> 封装 -> 双优化器 -> 一步 eden_loss -> backward -> step
# 用于在服务器上快速定位旧环境兼容问题。
# 用法：CUDA_VISIBLE_DEVICES=3 python debug_eden_step.py 2>&1 | tee debug.log
import torch

from config import _C, _update_config_from_file
from models import build_model
from eden_adapter_swinv2 import (SwinV2WithDiffAE, eden_kwargs_from_config,
                                 build_eden_optimizers, build_diffae_scheduler)

CFG = 'configs/swinv2/swinv2_tiny_patch4_window8_256_eden_tweight_mask_mini.yaml'

print('== 1. 加载 config ==')
config = _C.clone()
_update_config_from_file(config, CFG)
config.freeze()
print('EDEN.ENABLED =', config.EDEN.ENABLED,
      '| T_WEIGHT =', config.EDEN.T_WEIGHT_ENABLED,
      '| MASK =', config.EDEN.MASK_ENABLED)

print('== 2. 建模 + 封装 ==')
base = build_model(config)
model = SwinV2WithDiffAE(base, **eden_kwargs_from_config(config))
n_task = sum(p.numel() for n, p in model.named_parameters()
             if 'diffae' not in n and p.requires_grad)
print(f'params: total={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M '
      f'task={n_task / 1e6:.1f}M')

print('== 3. 双优化器 + diffae 调度器 ==')
opt_t, opt_d = build_eden_optimizers(config, model)
sch_d = build_diffae_scheduler(config, opt_d, 1000)
print('task lr =', opt_t.param_groups[0]['lr'], '| diffae lr =', opt_d.param_groups[0]['lr'])

print('== 4. 一步训练（fake batch, bs=2, AMP 开） ==')
model.cuda().train()
samples = torch.randn(2, 3, 256, 256).cuda()
targets = torch.randint(0, 1000, (2,)).cuda()
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast(enabled=True):
    losses = model(samples, targets, criterion)
loss = losses['loss_task'] + losses['loss_diffae']
scaler.scale(loss).backward()
scaler.unscale_(opt_t)
torch.nn.utils.clip_grad_norm_(
    [p for g in opt_t.param_groups for p in g['params']], 5.0)
scaler.unscale_(opt_d)
scaler.step(opt_t)
scaler.step(opt_d)
scaler.update()

for k, v in losses.items():
    print(f'  {k}: {v.item() if hasattr(v, "item") else v}')

print('== 5. DDP 包装（单进程 gloo；验证 state_dict 广播兼容） ==')
import os
import torch.distributed as dist
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29517')
dist.init_process_group(backend='gloo', world_size=1, rank=0)
ddp = torch.nn.parallel.DistributedDataParallel(
    model, device_ids=[0], broadcast_buffers=False)
with torch.cuda.amp.autocast(enabled=True):
    losses2 = ddp(samples, targets, criterion)
(losses2['loss_task'] + losses2['loss_diffae']).backward()
print('  DDP forward/backward OK')
dist.destroy_process_group()

print('== 全部通过 ==')
