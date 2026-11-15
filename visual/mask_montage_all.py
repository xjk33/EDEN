"""
结构化掩码四任务拼图（论文用）
==============================

回答的问题：**掩码随训练进程怎么变、不同任务的掩码选址有什么差异。**

与 tweight_all.py 同一哲学：本脚本**纯读 npz 缓存**（numpy + matplotlib，
不 import 任何任务框架、不需要 torch/GPU）。测量由各任务自己的 notebook
（probe_mask_{seg,det,cls,reid}.ipynb，在各自环境里运行）完成——它们调
pc.measure_mask_viz 后用 pc.save_mask_viz_cache 存盘，本脚本只负责拼装：

    各任务 notebook（各自环境）          本脚本（任意环境）
    measure_mask_viz + save cache ──>  读 npz -> 挑早/中/晚 ckpt -> 拼图
    iter_8000_cache.npz 等             montage_masked.png / montage_prob.png

图版式（与 montage_heat 一致的"一张图看懂"风格）：
  行 = 任务；列 = [原图 | 早期 | 中期 | 晚期]。
  每个任务内部用**同一张图**贯穿（image_idx 可配，跨阶段 CRN 同图同噪声），
  列标题标出该阶段对应的 ckpt tag。
  - montage_masked.png：蒙后输入 x~0（模型实际看到的退化图）
  - montage_prob.png：  块级采样概率热力（全部 cell 共享色标，跨任务可比）

用法：
  1. 先在四个任务各自的环境里跑对应的 probe_mask_*.ipynb（产出 cache npz）；
  2. 按需修改下面 TASKS 的 cache_dir / image_idx / stages；
  3. python mask_montage_all.py            # 全部任务
     python mask_montage_all.py seg det    # 只拼指定任务
输出：OUT_DIR/montage_masked.png、OUT_DIR/montage_prob.png
"""
import argparse
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EDEN_ROOT = os.path.abspath(os.path.join(HERE, '..'))

# =====================================================================
# 配置：四个任务的掩码缓存目录（与 probe_mask_*.ipynb 的 OUT 一致）
# =====================================================================
TASKS = {
    'seg': dict(
        name='Seg / ADE20K (FCN-R50)',
        cache_dir='tasks/Segmentation/mmsegmentation/work_dirs/'
                  'offline_probe/mask_viz',
        image_idx=0,            # 用该任务测量批次里的第几张图（跨阶段同一张）
    ),
    'det': dict(
        name='Det / COCO (Mask R-CNN R50)',
        cache_dir='tasks/Detection/mmdetection/work_dirs/'
                  'offline_probe/mask_viz',
        image_idx=0,
    ),
    'cls': dict(
        name='Cls / ImageNet (SwinV2-T)',
        cache_dir='visual/outputs/mask_viz/cls',
        image_idx=0,
    ),
    'reid': dict(
        name='ReID / MSMT17 (TransReID ViT-S)',
        cache_dir='visual/outputs/mask_viz/reid',
        image_idx=0,
    ),
}
OUT_DIR = os.path.join(HERE, 'outputs', 'mask_montage')

# 早/中/晚阶段的挑选：None = 自动（排序后取 首/中/尾 三个 ckpt）；
# 也可以显式指定 tag（cache 文件名去掉 _cache.npz 的部分），例如
# STAGES = {'seg': dict(early='iter_8000', mid='iter_40000', late='iter_80000')}
STAGES = {k: None for k in TASKS}

DPI = 200                     # 论文图分辨率


# =====================================================================
# 缓存读取（与 probe_core.save_mask_viz_cache 的布局对应；不依赖 probe_core，
# 保持本脚本零框架依赖）
# =====================================================================
def load_cache(path):
    """读一个 ckpt 的掩码缓存。Returns: dict ——
    img_s [N,3,S,S] / prob [N,1,g,g] / g_block / mask / masked [N,3,S,S] /
    masked_frac [N] / S / g / k / ratio / alpha"""
    z = np.load(path)
    S, g, k, ratio, alpha = z['meta'].tolist()
    return {'img_s': z['img_s'], 'g_block': z['g_block'],
            'prob': z['prob'], 'mask': z['mask'], 'masked': z['masked'],
            'masked_frac': z['masked_frac'],
            'S': int(S), 'g': int(g), 'k': int(k),
            'ratio': float(ratio), 'alpha': float(alpha),
            'tag': re.sub(r'_cache$', '',
                          os.path.splitext(os.path.basename(path))[0])}


def _stage_key(path):
    nums = re.findall(r'\d+', os.path.splitext(os.path.basename(path))[0])
    return int(nums[-1]) if nums else 0


def pick_stages(cache_dir, stages=None):
    """从缓存目录挑 早/中/晚 三个 ckpt 的 cache 路径。
    stages=None：排序后取 首/中/尾；否则按显式 tag 找。
    Returns: list of (stage_label, path)，长度<=3。"""
    paths = sorted(glob.glob(os.path.join(cache_dir, '*_cache.npz')),
                   key=_stage_key)
    if not paths:
        return []
    if stages is None:
        idx = sorted({0, len(paths) // 2, len(paths) - 1})
        picked = [paths[i] for i in idx]
    else:
        by_tag = {re.sub(r'_cache$', '',
                         os.path.splitext(os.path.basename(p))[0]): p
                  for p in paths}
        picked = []
        for lab in ('early', 'mid', 'late'):
            tag = stages.get(lab)
            if tag is None or tag not in by_tag:
                print(f'[pick_stages] {cache_dir}: 找不到指定阶段 '
                      f'{lab}={tag}，跳过该档位')
                continue
            picked.append(by_tag[tag])
    return [p for p in picked]


# =====================================================================
# 拼图
# =====================================================================
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_mask_montage(task_panels, mode='masked', save_path=None, dpi=DPI):
    """四任务掩码拼图：行 = 任务，列 = [原图 | 早期 | 中期 | 晚期]。

    task_panels: list of dict ——
        {'name': str, 'img': [3,S,S] 原图,
         'cells': list of (stage_label, cell_array)}
        mode='masked' 时 cell_array = [3,S,S] 蒙后输入；
        mode='prob' / 'g' 时 = [g,g] 热力（全部 cell 共享色标）。
    """
    plt = _plt()
    rows = [p for p in task_panels if p is not None and p['cells']]
    if not rows:
        print('[plot_mask_montage] 没有任何任务数据，跳过'
              '（先跑各任务的 probe_mask_*.ipynb 产出 cache）')
        return None
    n_r = len(rows)
    n_c = 1 + max(len(p['cells']) for p in rows)

    vmin = vmax = None
    if mode in ('prob', 'g'):
        allv = np.concatenate([c[1].ravel() for p in rows for c in p['cells']])
        vmin, vmax = float(allv.min()), float(allv.max())

    fig, axes = plt.subplots(n_r, n_c, figsize=(2.4 * n_c, 2.5 * n_r + 0.4),
                             squeeze=False)
    im = None
    for r, p in enumerate(rows):
        # 第 0 列：原图（各阶段共用同一张）
        axes[r, 0].imshow(p['img'].transpose(1, 2, 0))
        axes[r, 0].set_ylabel(p['name'], fontsize=9, rotation=0,
                              ha='right', va='center')
        for c in range(1, n_c):
            ax = axes[r, c]
            if c - 1 < len(p['cells']):
                lab, cell = p['cells'][c - 1]
                if mode == 'masked':
                    ax.imshow(cell.transpose(1, 2, 0))
                else:
                    im = ax.imshow(cell, cmap='turbo', interpolation='nearest',
                                   vmin=vmin, vmax=vmax)
                if r == 0:
                    ax.set_title(lab, fontsize=9)
            ax.axis('off')
        axes[r, 0].axis('off')
    if rows:
        axes[0, 0].set_title('image', fontsize=9)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    supt = {'masked': 'structured mask evolution (masked input)',
            'prob': 'mask sampling probability evolution',
            'g': 'task sensitivity G evolution'}[mode]
    fig.suptitle(supt, fontsize=11)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    return fig


def build_task_panel(key, mode):
    """读一个任务的缓存，挑出 [原图 + 早/中/晚 cell]。"""
    cfg = TASKS[key]
    cache_dir = os.path.join(EDEN_ROOT, cfg['cache_dir'])
    picked = pick_stages(cache_dir, STAGES.get(key))
    if not picked:
        print(f'[{key}] 未找到掩码缓存：{cache_dir}/*_cache.npz'
              f'（先在对应环境跑 probe_mask_{key}.ipynb）')
        return None
    src = {'masked': 'masked', 'prob': 'prob', 'g': 'g_block'}[mode]
    cells = []
    img = None
    for p in picked:
        c = load_cache(p)
        i = min(cfg['image_idx'], c['img_s'].shape[0] - 1)
        img = c['img_s'][i]
        arr = c[src][i]
        cells.append((c['tag'], arr if mode == 'masked' else arr[0]))
    return {'name': cfg['name'], 'img': img, 'cells': cells}


def main():
    ap = argparse.ArgumentParser(description='结构化掩码四任务拼图（论文用）')
    ap.add_argument('tasks', nargs='*', default=None,
                    help=f'要拼的任务（默认全部）：{list(TASKS)}')
    args = ap.parse_args()
    keys = args.tasks or list(TASKS)
    for k in keys:
        if k not in TASKS:
            ap.error(f'未知任务 {k!r}（可选：{list(TASKS)}）')

    for mode in ('masked', 'prob'):
        panels = [build_task_panel(k, mode) for k in keys]
        fig = plot_mask_montage(
            panels, mode=mode,
            save_path=os.path.join(OUT_DIR, f'montage_{mode}.png'))
        if fig is not None:
            print(f'saved {OUT_DIR}/montage_{mode}.png')
    print(f'done -> {OUT_DIR}')


if __name__ == '__main__':
    main()
