"""
掩码缓存诊断（纯 numpy，无框架依赖，任意环境可跑）
====================================================

针对 det 单步图的疑点：prob 面板数值范围在数学上不可能是 64 路 softmax
（64 个概率 min 必 <= 1/64 = 0.015625）。本脚本直接读 npz 缓存，逐图验证：

  1. prob 是否为合法分布（sum == 1）；
  2. prob 是否等于 softmax(g_block)（与 measure_mask_viz / MaskSelector
     的当前实现一致）；
  3. 被蒙块的平均 prob 是否高于保留块（采样是否真的被 G 引导）——
     比值 ~1.0 说明引导未生效（softmax 作用在原始幅度上 ≈ 均匀）。

用法（服务器上，在缓存目录或带路径）：
  python diag_mask_cache.py work_dirs/offline_probe/mask_viz/epoch_4_cache.npz
  python diag_mask_cache.py work_dirs/offline_probe/mask_viz/*_cache.npz
"""
import sys

import numpy as np


def diag(path):
    z = np.load(path)
    S, g, k, ratio, alpha = z['meta'].tolist()
    S, g, k = int(S), int(g), int(k)
    n_blocks = g * g
    print(f'== {path}')
    print(f'   S={S} g={g} k={k}/{n_blocks} ratio={ratio:.3f} alpha={alpha}')
    for i in range(z['prob'].shape[0]):
        p = z['prob'][i, 0].astype(np.float64)
        h = z['g_block'][i, 0].astype(np.float64)
        m = z['mask'][i, 0]
        sm = np.exp(h - h.max())
        sm /= sm.sum()
        masked = m < 0.5
        pm = p[masked].mean() if masked.any() else float('nan')
        pu = p[~masked].mean() if (~masked).any() else float('nan')
        print(f'   img{i}: prob min={p.min():.6g} max={p.max():.6g} '
              f'sum={p.sum():.6g} (uniform={1 / n_blocks:.6g}, '
              f'peak={p.max() * n_blocks:.3f}x)')
        print(f'         h min={h.min():.4g} max={h.max():.4g} '
              f'spread={h.max() - h.min():.4g} -> softmax(h) '
              f'max={sm.max():.6g} peak={sm.max() * n_blocks:.3f}x '
              f'|prob-softmax(h)|_max={np.abs(p - sm).max():.3g}')
        print(f'         masked={int(masked.sum())}/{n_blocks} '
              f'E[p|masked]={pm:.6g} E[p|kept]={pu:.6g} '
              f'ratio={pm / pu:.3f} (1.0 = 引导未生效)')
    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for a in sys.argv[1:]:
        diag(a)
