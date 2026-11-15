"""
EDEN 训练循环（TransReID）：由 processor.do_train 改写，差异点：

  - 去除 DenoiseRep 专有逻辑（freeze/unfreeze denoise layers、ploss、
    fuse_parameters）；
  - 双优化器：task（仓库原生）+ diffae（AdamW 独立调度），单次前向、单次反传、
    各自 step；
  - 任务损失经 wrapper 的 criterion 通道传入（原仓库 loss_func 的 ploss 位补 0），
    返回 (losses dict, score)；loss = loss_task + loss_diffae；
  - 监控标量（tw_*/zsem_std/ts_mean）随日志记录，不参与求和。

eval / inference 与原版完全一致：去噪网络仅存在于训练阶段。
"""
import logging
import os
import time

import torch
import torch.distributed as dist
from torch.cuda import amp

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval


def do_train_eden(cfg,
                  model,
                  center_criterion,
                  train_loader,
                  val_loader,
                  optimizer,
                  optimizer_diffae,
                  optimizer_center,
                  scheduler,
                  scheduler_diffae,
                  loss_fn,
                  num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training (EDEN)')
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    loss_diff_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    # 任务损失适配：原仓库 loss_func(score, feat, target, target_cam, ploss)
    # 的 ploss 位（DenoiseRep 项）补 0，等价于原版任务损失
    def task_criterion(score, feat, target, target_cam):
        return loss_fn(score, feat, target, target_cam, 0)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_diff_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_diffae.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                losses, score = model(img, target, cam_label=target_cam,
                                      view_label=target_view,
                                      criterion=task_criterion)
                loss = losses['loss_task'] + losses['loss_diffae']
            scaler.scale(loss).backward()

            # 单次反传 + 双优化器各自 step（各自只持有自己的参数）
            scaler.step(optimizer)
            scaler.step(optimizer_diffae)
            scaler.update()
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(losses['loss_task'].item(), img.shape[0])
            loss_diff_meter.update(losses['loss_diffae'].item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info(
                        "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, LossDiff: {:.3f}, Acc: {:.3f}, "
                        "Base Lr: {:.2e}, Diff Lr: {:.2e}, tw_ent: {}, zsem_std: {}"
                        .format(epoch, (n_iter + 1), len(train_loader),
                                loss_meter.avg, loss_diff_meter.avg, acc_meter.avg,
                                base_lr, optimizer_diffae.param_groups[0]['lr'],
                                _v(losses, 'tw_ent'), _v(losses, 'zsem_std')))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        # 两个调度器同节奏按 epoch step
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        scheduler_diffae.step()
        if not cfg.MODEL.DIST_TRAIN:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch * (n_iter + 1),
                                train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0:
                # state_dict 含 EMA shadow（摊平键）与 t 自适应 buffer，随 ckpt 保存
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR,
                                        cfg.MODEL.NAME + '_eden_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def _v(losses, key):
    """监控标量格式化；未启用对应模块时显示 '-'。"""
    v = losses.get(key)
    return f'{v.item():.4f}' if isinstance(v, torch.Tensor) else '-'
