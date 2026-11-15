"""
t 自适应采样概率可视化（四任务统一脚本）
==========================================

回答的问题：**t 自适应在各个训练阶段学到了什么样的 t 采样分布。**

t_weight / mu_table / bin_rep_t / grid_bin / logsnr_per_t 都是 TAdaptive 的
buffer，随 checkpoint 保存——因此本脚本**只读 ckpt、不建模型、不取数据、
不用 GPU、不依赖任何任务框架**（mmseg / mmdet / Swin / TransReID 的 ckpt
包装格式都已兼容），四个任务的代码框架差异在这里完全无关。

兼容的 ckpt 包装：
  - mmengine（seg / det）：{'state_dict': ...}
  - Swin eden_save_checkpoint（cls）：{'model': ..., 'epoch': ...}
  - TransReID（reid）：裸 state_dict
键名兼容：重构后 t_adaptive.probe.* / t_adaptive.scheduler.*（任一前缀，含
DDP module.）；重构前的顶层旧键（mu_table 等）也能认，但语义见 probe_core
头注释（旧 run 读出为初始均匀值属正常）。

用法：
  1. 按需修改下面 TASKS 里各任务的 ckpt glob（默认按 终版组合 run 的约定路径，
     相对 EDEN 根目录，即本文件上一级）；
  2. python tweight_all.py            # 跑全部任务
     python tweight_all.py seg det    # 只跑指定任务

输出（OUT_ROOT/<task>/）：
  <tag>_snapshot.png   每个 ckpt 一张：t_weight 柱状 + μ̄ 折线（t / logSNR 两视图）
                       + 训练网格上的采样概率 p(t) 曲线
  evolution.png        跨 ckpt：t_weight 热力图 + 熵 / 最大桶权重曲线
  pt_curves.png        跨 ckpt：p(t) 曲线叠画（颜色 = 训练阶段）
OUT_ROOT/_montage/（四任务拼图，论文用）：
  montage_pt.png       行=任务：p(t) 演化叠画
  montage_heat.png     行=任务：t_weight 热力图
  entropy_all.png      四任务熵演化对比
"""
import argparse
import glob
import os
import re

import numpy as np

import torch  # 仅用于 torch.load；提取与绘图都是纯 numpy

HERE = os.path.dirname(os.path.abspath(__file__))
EDEN_ROOT = os.path.abspath(os.path.join(HERE, '..'))

# =====================================================================
# 配置：四个任务的 ckpt（glob 或显式列表；相对 EDEN 根目录）
# =====================================================================
TASKS = {
    # 分割：ADE20K / FCN-R50（mmsegmentation）
    'seg': dict(
        name='Seg / ADE20K (FCN-R50)',
        ckpts='tasks/Segmentation/mmsegmentation/work_dirs/eden/'
              'fcn_diffae_tweight_mask/iter_*.pth',
    ),
    # 检测：COCO / Mask R-CNN R50-FPN（mmdetection；epoch 制保存）
    'det': dict(
        name='Det / COCO (Mask R-CNN R50)',
        ckpts='tasks/Detection/mmdetection/work_dirs/eden/'
              'maskrcnn_diffae_tweight_mask/epoch_*.pth',
    ),
    # 分类：ImageNet / SwinV2-T（ckpt_epoch_*.pth；output 路径按实际调整）
    'cls': dict(
        name='Cls / ImageNet (SwinV2-T)',
        ckpts='tasks/Classification/imagenet/Swin-Transformer/output/'
              '**/ckpt_epoch_*.pth',
    ),
    # 行人重识别：MSMT17 / TransReID ViT-S（transformer_eden_{epoch}.pth）
    'reid': dict(
        name='ReID / MSMT17 (TransReID ViT-S)',
        ckpts='tasks/Person-ReID/TransReID-SSL/transreid_pytorch/log/'
              'transreid/msmt17/vit_small_eden_tweight_mask/'
              'transformer_eden_*.pth',
    ),
}
OUT_ROOT = os.path.join(HERE, 'outputs', 'tweight')

SAVE_SNAPSHOTS = True     # 每个 ckpt 一张三联快照（关掉可省时间省空间）


# =====================================================================
# ckpt 读取：包装拆解 + t 自适应 buffer 提取
# =====================================================================
_WRAP_KEYS = ('state_dict', 'model', 'model_state_dict', 'net', 'module')

# 目标 buffer 的"末段键名" -> 语义名（任一前缀匹配，兼容 DDP module. 与
# 重构后的 t_adaptive.probe./scheduler. 前缀）
_BUFFER_TAILS = {
    't_weight': 't_weight',           # scheduler：桶采样权重 [K]
    'mu_table': 'mu_table',           # probe：贡献度 EMA [K]
    'bin_rep_t': 'bin_rep_t',         # probe：各桶代表 t [K]
    'bin_edges': 'bin_edges',         # probe：logSNR 桶边界 [K+1]
    'logsnr_per_t': 'logsnr_per_t',   # probe：全 T 网格 logSNR [T]
    'grid_bin': 'grid_bin',           # probe：训练网格点 -> 桶号 [L]
}


def _load_state_any(path):
    """拆各种 ckpt 包装，返回扁平 state_dict。"""
    obj = torch.load(path, map_location='cpu')
    if not isinstance(obj, dict):
        raise ValueError(f'{path}: 无法识别的 ckpt 格式')
    for k in _WRAP_KEYS:
        v = obj.get(k)
        if isinstance(v, dict) and any(
                isinstance(t, torch.Tensor) for t in v.values()):
            return v
    if any(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj                      # 裸 state_dict（reid）
    raise ValueError(f'{path}: 找不到 state_dict（顶层键：{list(obj)[:10]}）')


def read_tweight_from_state(sd, tag=''):
    """从扁平 state_dict（tensor -> array 已完成）提取 t 自适应快照。
    纯 numpy，与 ckpt 包装无关；未找到 t_weight 返回 None。

    Returns: dict ——
      t_weight [K] / mu_table [K] / rep_t [K] / rep_logsnr [K] /
      grid_t [L] / grid_logsnr [L] / grid_prob [L] / entropy / tag
    """
    found = {}
    for k, v in sd.items():
        tail = k.split('.')[-1]
        if tail in _BUFFER_TAILS and tail not in found:
            found[tail] = np.asarray(v, dtype=np.float64)
    if 't_weight' not in found:
        return None

    w = found['t_weight']
    K = len(w)
    mu = found.get('mu_table', np.zeros(K))
    rep_t = found.get('bin_rep_t', np.arange(K)).astype(int)
    logsnr_per_t = found.get('logsnr_per_t')
    rep_logsnr = logsnr_per_t[rep_t] if logsnr_per_t is not None \
        else np.full(K, np.nan)

    # 训练网格上的采样概率：p_t = t_weight[grid_bin] 再归一化
    # （与 TWeightScheduler.sample_probs 同公式）
    grid_bin = found.get('grid_bin')
    if grid_bin is not None:
        L = len(grid_bin)
        T = len(logsnr_per_t) if logsnr_per_t is not None else 1000
        step = max(1, int(round(T / L)))          # ddim100 -> 10；全网格 -> 1
        grid_t = np.arange(L) * step
        p = w[grid_bin.astype(int)]
        p = p / max(p.sum(), 1e-12)
        grid_logsnr = logsnr_per_t[np.clip(grid_t, 0, T - 1)] \
            if logsnr_per_t is not None else np.full(L, np.nan)
    else:
        grid_t, p, grid_logsnr = rep_t, w / w.sum(), rep_logsnr

    ent = float(-(w * np.log(np.clip(w, 1e-12, None))).sum() / np.log(K))
    return {'t_weight': w, 'mu_table': mu, 'rep_t': rep_t,
            'rep_logsnr': rep_logsnr, 'grid_t': grid_t, 'grid_prob': p,
            'grid_logsnr': grid_logsnr, 'entropy': ent, 'tag': tag}


def read_tweight_from_ckpt(path):
    """torch.load + read_tweight_from_state（包装拆解见 _load_state_any）。"""
    sd = _load_state_any(path)
    sd = {k: (v.float().cpu().numpy() if isinstance(v, torch.Tensor) else v)
          for k, v in sd.items()}
    return read_tweight_from_state(
        sd, tag=os.path.splitext(os.path.basename(path))[0])


def _stage_key(path):
    """从文件名提取阶段序号用于排序（iter_20000 -> 20000，epoch_3 -> 3）。"""
    nums = re.findall(r'\d+', os.path.splitext(os.path.basename(path))[0])
    return int(nums[-1]) if nums else 0


# =====================================================================
# 绘图
# =====================================================================
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # CJK 字体回退链：按平台常见字体顺序找，找不到就退回默认（西文照常）
    matplotlib.rcParams['font.sans-serif'] = [
        'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Zen Hei',
        'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return plt


def _stage_colors(n):
    """训练阶段配色（早 -> 晚：蓝 -> 红）。"""
    plt = _plt()
    return plt.cm.turbo(np.linspace(0.08, 0.92, max(n, 2)))


def plot_snapshot(snap, save_path=None):
    """单 ckpt 三联：t_weight 柱状+μ̄（t 视图 / logSNR 视图）+ 网格采样概率 p(t)。"""
    plt = _plt()
    w, mu = snap['t_weight'], snap['mu_table']
    K = len(w)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    mu_n = mu / np.abs(mu).max() * w.max() if np.abs(mu).max() > 0 else mu
    for ax, x, xlab in [(axes[0], snap['rep_t'], 't (bin rep)'),
                        (axes[1], snap['rep_logsnr'], 'log SNR (bin rep)')]:
        width = (np.abs(x[1] - x[0]) * 0.7) if K > 1 else 10
        ax.bar(x, w, width=width, alpha=0.7, label='t_weight')
        ax.plot(x, mu_n, color='tab:red', marker='o', ms=3,
                label='mu_table (scaled)')
        ax.axhline(1.0 / K, color='k', ls='--', lw=0.8, label='uniform 1/K')
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('prob / weight')
    ax = axes[2]
    ax.plot(snap['grid_logsnr'], snap['grid_prob'], lw=1.8, color='tab:blue')
    ax.set_xlabel('log SNR')
    ax.set_ylabel('p(t)')
    ax.set_title('训练网格上的采样概率 p(t)', fontsize=10)
    ax.grid(alpha=0.3)
    fig.suptitle(f"{snap['tag']} | entropy={snap['entropy']:.3f} | "
                 f"max={w.max():.3f}@bin{int(w.argmax())}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_evolution(snaps, title='', save_path=None):
    """跨 ckpt：t_weight 热力图（行=ckpt，列=桶）+ 熵 / 最大桶权重曲线。"""
    plt = _plt()
    labels = [s['tag'] for s in snaps]
    W = np.stack([s['t_weight'] for s in snaps])
    ents = [s['entropy'] for s in snaps]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.2, 0.55 * len(snaps) + 2)),
                             gridspec_kw={'width_ratios': [2.2, 1]})
    im = axes[0].imshow(W, aspect='auto', cmap='viridis')
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xticks(range(W.shape[1]))
    axes[0].set_xticklabels([str(t) for t in snaps[0]['rep_t']],
                            rotation=45, fontsize=7)
    axes[0].set_xlabel('bin rep t')
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    ax = axes[1]
    ax.plot(ents, range(len(labels)), 'o-', color='tab:orange', label='entropy')
    ax.plot([s['t_weight'].max() for s in snaps], range(len(labels)), 's-',
            color='tab:red', label='max weight')
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.invert_yaxis()
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('value')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(f't_weight evolution | {title}')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_pt_curves(snaps, title='', save_path=None, ax=None):
    """跨 ckpt 的 p(t) 叠画（颜色 = 训练阶段；x = logSNR）。"""
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
    else:
        fig = ax.figure
    colors = _stage_colors(len(snaps))
    for s, c in zip(snaps, colors):
        ax.plot(s['grid_logsnr'], s['grid_prob'], lw=1.6, color=c,
                label=s['tag'])
    K = len(snaps[0]['t_weight'])
    ax.axhline(1.0 / len(snaps[0]['grid_prob']), color='k', ls='--', lw=0.8,
               label='uniform')
    ax.set_xlabel('log SNR')
    ax.set_ylabel('p(t)')
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_montage(task_snaps, kind='pt', save_path=None):
    """四任务拼图（行=任务）：kind='pt' p(t) 叠画 | 'heat' t_weight 热力图。"""
    plt = _plt()
    keys = [k for k in TASKS if k in task_snaps and task_snaps[k]]
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 3.2 * n), squeeze=False)
    for r, key in enumerate(keys):
        ax = axes[r, 0]
        snaps = task_snaps[key]
        if kind == 'pt':
            plot_pt_curves(snaps, title=TASKS[key]['name'], ax=ax)
        else:
            W = np.stack([s['t_weight'] for s in snaps])
            im = ax.imshow(W, aspect='auto', cmap='viridis')
            ax.set_yticks(range(len(snaps)))
            ax.set_yticklabels([s['tag'] for s in snaps], fontsize=7)
            ax.set_xticks(range(W.shape[1]))
            ax.set_xticklabels([str(t) for t in snaps[0]['rep_t']],
                               rotation=45, fontsize=6)
            ax.set_xlabel('bin rep t')
            ax.set_title(TASKS[key]['name'], fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_entropy_all(task_snaps, save_path=None):
    """四任务熵演化对比（x = 归一化训练进度，任务间 ckpt 数不同也能叠画）。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for key, snaps in task_snaps.items():
        if not snaps:
            continue
        x = np.linspace(0, 1, len(snaps))
        ax.plot(x, [s['entropy'] for s in snaps], 'o-', ms=3,
                label=TASKS[key]['name'])
    ax.set_xlabel('训练进度（归一化）')
    ax.set_ylabel('t_weight 归一化熵')
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    return fig


# =====================================================================
# 主流程
# =====================================================================
def run_task(key):
    cfg = TASKS[key]
    pattern = cfg['ckpts']
    if isinstance(pattern, str):
        ckpts = sorted(glob.glob(os.path.join(EDEN_ROOT, pattern),
                                 recursive=True), key=_stage_key)
    else:
        ckpts = sorted(pattern, key=_stage_key)
    if not ckpts:
        print(f'[{key}] 未匹配到 ckpt：{pattern}（请检查 TASKS 配置）')
        return []
    out_dir = os.path.join(OUT_ROOT, key)
    os.makedirs(out_dir, exist_ok=True)

    snaps = []
    for p in ckpts:
        try:
            s = read_tweight_from_ckpt(p)
        except Exception as e:      # 单个 ckpt 读失败不拖垮整批
            print(f'[{key}] {os.path.basename(p)} 读取失败：{e}')
            continue
        if s is None:
            print(f'[{key}] {os.path.basename(p)} 无 t_weight buffer'
                  f'（未启用 t 自适应或旧架构 ckpt），跳过')
            continue
        snaps.append(s)
        print(f"[{key}] {s['tag']}: entropy={s['entropy']:.3f} "
              f"max={s['t_weight'].max():.3f}@bin{int(s['t_weight'].argmax())}")
        if SAVE_SNAPSHOTS:
            plot_snapshot(s, save_path=os.path.join(
                out_dir, f"{s['tag']}_snapshot.png"))
    if not snaps:
        return []
    plot_evolution(snaps, title=cfg['name'],
                   save_path=os.path.join(out_dir, 'evolution.png'))
    plot_pt_curves(snaps, title=cfg['name'],
                   save_path=os.path.join(out_dir, 'pt_curves.png'))
    return snaps


def main():
    ap = argparse.ArgumentParser(description='t 自适应采样概率可视化（四任务）')
    ap.add_argument('tasks', nargs='*', default=list(TASKS),
                    choices=list(TASKS), help='要跑的任务（默认全部）')
    args = ap.parse_args()

    task_snaps = {}
    for key in args.tasks:
        task_snaps[key] = run_task(key)

    montage_dir = os.path.join(OUT_ROOT, '_montage')
    os.makedirs(montage_dir, exist_ok=True)
    plot_montage(task_snaps, kind='pt',
                 save_path=os.path.join(montage_dir, 'montage_pt.png'))
    plot_montage(task_snaps, kind='heat',
                 save_path=os.path.join(montage_dir, 'montage_heat.png'))
    plot_entropy_all(task_snaps,
                     save_path=os.path.join(montage_dir, 'entropy_all.png'))
    print(f'done -> {OUT_ROOT}')


if __name__ == '__main__':
    main()
