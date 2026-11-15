"""
EDEN 训练入口（TransReID，MSMT17）：由 train.py 改写，差异点：

  - 去除 denoiserep 依赖（convert_vit_denoise_linear / ploss 等）；
  - cfg.set_new_allowed(True) 后再 merge，允许 yml 里的 EDEN 配置节点；
  - 模型用 TransReIDWithDiffAE 封装（z_sem = global_feat，384 维）；
  - 双优化器（task 原生 + diffae AdamW 独立调度），训练循环走 do_train_eden。

用法（单卡）：
  python train_eden.py --config_file configs/msmt17/vit_small_eden_tweight_mask.yml
多卡：
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
      --nproc_per_node=4 --master_port 66666 train_eden.py \
      --config_file configs/msmt17/vit_small_eden_tweight_mask.yml
"""
import argparse
import os
import random

import numpy as np
import torch

from config import cfg
from datasets import make_dataloader
from eden_adapter_reid import (TransReIDWithDiffAE, build_eden_optimizers,
                               build_diffae_scheduler, eden_kwargs_from_config)
from loss import make_loss
from model import make_model
from processor.eden_processor import do_train_eden
from solver import WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler
from utils.logger import setup_logger


def _preregister_new_keys(cfg, yml_path):
    """旧版 yacs 兼容：新版可用 cfg.set_new_allowed(True) 放开未知键，
    旧版（无该 API 且 merge 时拒绝未知键）只能在合并前把 yml 中 cfg
    尚未定义的键预登记。yacs merge 时会对字符串做 literal_eval 解码
    （'1e-3' -> 0.001），此处同样解码，保证类型一致不触发类型检查。"""
    import ast
    import yaml
    from yacs.config import CfgNode as CN

    with open(yml_path, encoding='utf-8') as f:
        yml = yaml.safe_load(f)

    def _decode(v):
        if not isinstance(v, str):
            return v
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v

    def _ensure(node, d):
        for k, v in (d or {}).items():
            if isinstance(v, dict):
                if k not in node:
                    node[k] = CN()
                _ensure(node[k], v)
            elif k not in node:
                node[k] = _decode(v)

    _ensure(cfg, yml)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID EDEN Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 允许 yml 中的 EDEN 节点：新版 yacs 显式放开；旧版预登记新键。
    if hasattr(cfg, 'set_new_allowed'):
        cfg.set_new_allowed(True)
    if args.config_file != '':
        _preregister_new_keys(cfg, args.config_file)
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    try:
        os.makedirs(output_dir)
    except:
        pass

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(output_dir))

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # 裸模型（build_transformer 内部按 PRETRAIN_CHOICE 加载预训练权重），
    # 再用 EDEN wrapper 封装（z_sem = global_feat，384 维；enc_compress = Identity）
    base_model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                            view_num=view_num)
    model = TransReIDWithDiffAE(base_model, **eden_kwargs_from_config(cfg))
    model.to(device)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_diffae, optimizer_center = build_eden_optimizers(
        cfg, model, center_criterion)

    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        logger.info('===========using cosine learning rate=======')
        scheduler = create_scheduler(cfg, optimizer)
    else:
        logger.info('===========using normal learning rate=======')
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)
    scheduler_diffae = build_diffae_scheduler(cfg, optimizer_diffae)

    do_train_eden(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_diffae,
        optimizer_center,
        scheduler,
        scheduler_diffae,
        loss_func,
        num_query, args.local_rank
    )
