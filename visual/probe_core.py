"""
离线探针核心：所有分析 ipynb 共用的测量与绘图函数。

设计约定：
  - 训练代码（eden/）不含任何监控逻辑；这里全部离线工作：加载 config + checkpoint，
    取 val 前若干张图，测完出图。
  - 每个 ipynb 只做三件事：build_model / build_val_batch -> 调本文件的 measure_* -> 调 plot_*。
  - 在 mmsegmentation 环境下运行（需要 mmseg/mmengine 建模型与 dataloader）。

包含五组能力：
  1. Dt 谱 + meta-gradient 金标准（Dt 价值判据：两者相关性越高 Dt 越可信）
  2. t_weight / d_table 快照与跨 ckpt 演化（t 自适应学到了什么）
  3. 重建可视化（z_sem 复活判据）
  4. 蒙块选址可视化（结构化噪声蒙的位置与剂量）
  5. 固定 t 的 loss 方差分解（大 t 抖动：测量噪声 vs 任务本质不稳定）

注意：旧 checkpoint 的 t 自适应 buffer 键名（顶层 t_weight/d_table/...）在重构后
变为 t_adaptive.probe.*/t_adaptive.scheduler.*，strict=False 加载即可——
Dt 谱、meta-grad、重建不受影响（只依赖网络权重）；旧 ckpt 的 t_weight 快照
会显示为初始均匀值。
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [58.395, 57.12, 57.375]


# =====================================================================
# 建模与取数
# =====================================================================
def build_model(cfg_path, ckpt=None, device=None, cfg_overrides=None):
    """从 config 建模型（可选加载 checkpoint）。strict=False 兼容旧 ckpt 键名。

    cfg_overrides: dict，建模前 merge 进 cfg.model——用于让旧 ckpt 的仪器结构
    与当前代码默认对齐（如 UNet 时代的 run 需 dict(denoiser_type='unet')，
    否则旧 config 里没这个键、会按新默认建出 DiT，键全对不上）。
    """
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmseg.registry import MODELS

    cfg = Config.fromfile(cfg_path)
    if cfg_overrides:
        cfg.merge_from_dict({'model': cfg_overrides})
    init_default_scope('mmseg')
    model = MODELS.build(cfg.model)
    if ckpt is not None:
        from mmengine.runner.checkpoint import load_state_dict
        # diffae_ema_shadow 是嵌套 dict，mmengine 的键匹配不消费它（报 unexpected
        # key 且 shadow 留在随机初始化状态）——先手动灌进 EMAManager 再加载，
        # 否则 apply_shadow 会把随机初始化权重换进 DiT（adaLN-Zero => 输出恒 0）
        state = torch.load(ckpt, map_location='cpu')
        sd = state.get('state_dict', state)
        if hasattr(model, 'ema') and 'diffae_ema_shadow' in sd:
            model.ema.load_state_dict(sd.pop('diffae_ema_shadow'))
        load_state_dict(model, sd, strict=False)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model.to(device).eval(), cfg


def build_val_batch(model, cfg, num_images=16):
    """
    取 val 前 num_images 张，一次取数、全部测量共用（保证跨 ckpt 可比）。
    Returns:
        dict: cond   —— [n, cond_dim] 压缩后语义条件（无梯度，供 Dt 谱/重建）
              x_diffae —— [n,3,S,S] [-1,1] 扩散输入
              xis    —— list of [1,3,H,W] 归一化原图（带梯度测量用）
              dsis   —— list of data_samples（task loss 用）
    """
    from mmengine.runner import Runner

    device = next(model.parameters()).device
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    val_dataloader = Runner.build_dataloader(cfg.val_dataloader)

    conds, xds, xis, dsis = [], [], [], []
    for batch in val_dataloader:
        data = model.data_preprocessor(batch, training=False)
        xi = data['inputs'].to(device)
        with torch.no_grad():
            # 与训练同分布：走模型自己的 encode_diffae（backbone + decode head）
            conds.append(model.diffae_enc_compress(model.encode_diffae(xi)))
        xd = (xi * std + mean) / 255.0 * 2.0 - 1.0
        xd = F.interpolate(xd, size=(model.image_size, model.image_size),
                           mode='bilinear', align_corners=False)
        xds.append(xd)
        xis.append(xi)
        dsis.extend(data['data_samples'])
        if len(xds) >= num_images:
            break
    n = min(num_images, len(xds))
    return {'cond': torch.cat(conds, 0)[:n], 'x_diffae': torch.cat(xds, 0)[:n],
            'xis': xis[:n], 'dsis': dsis[:n]}


def make_task_loss_fn(model, xis, dsis):
    """下游任务 loss callable（带梯度）。当前为 seg 实现：decode head 训练 loss。"""
    n = len(xis)

    def task_loss_fn():
        losses = {}
        for i, xi in enumerate(xis):
            feats = model.backbone(xi)
            li = model._decode_head_forward_train(feats, [dsis[i]])
            for k, v in li.items():
                if isinstance(v, torch.Tensor) and 'loss' in k:
                    losses[k] = losses.get(k, 0) + v
        return sum(losses.values()) / n

    return task_loss_fn


def make_cond_fn(model, xis):
    """带梯度的 cond callable：扩散梯度必须能经 z_sem 流回 encoder
    （detach 的 cond 会让 encoder 梯度恒为 0）。"""
    def cond_fn():
        # 与训练同分布：经 decode head 的分类器输入特征（encode_diffae 不挂 no_grad，梯度可回流）
        zs = [model.encode_diffae(xi) for xi in xis]
        return model.diffae_enc_compress(torch.cat(zs, 0))

    return cond_fn


# =====================================================================
# 统计工具
# =====================================================================
def spearman(x, y):
    """Spearman 秩相关（含并列名次的平均名次处理）。返回 (rho, p)。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return float('nan'), float('nan')

    def rank(a):
        order = a.argsort()
        r = np.empty(n, float)
        r[order] = np.arange(n)
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return sums[inv] / cnt[inv]

    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom < 1e-12:
        return float('nan'), float('nan')
    rho = float((rx * ry).sum() / denom)
    if abs(rho) >= 1.0:
        return rho, 0.0
    from math import erf, sqrt
    t_stat = rho * sqrt((n - 2) / (1 - rho ** 2))
    p = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))   # 正态近似，n>=8 够用
    return rho, p


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return float('nan')
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


# =====================================================================
# 测量 1：Dt 谱（CRN + 多噪声重复版）
# =====================================================================
@torch.no_grad()
def measure_dt_spectrum(model, batch, n_noise=8, seed=0):
    """
    Dt 谱。可靠性设计（目标：估计噪声不再污染曲线形状与跨 ckpt 比较）：
      - 预抽 n_noise 个噪声（专用种子生成器），全部 t 共用（CRN）：
        逐 t 抖动不再含"换了噪声"的成分；跨 ckpt 用同组噪声，严格可比；
      - 每 t 有效样本 = n_noise × B（默认 8×16=128），均值类估计噪声
        较单实现降 √128 ≈ 11×；采样分布（随机 ε × 固定图集）向训练场景对齐，
        图像集固定是为跨 ckpt 可比付出的有意取舍；
      - 噪声流与 meta-grad 轴相互独立（各自专用种子），避免共享噪声伪造相关。

    Dt 候选定义（每 t，同图同噪声同 t 配对；x0 预测，L = ‖pred−x‖²）：
      dabs —— 绝对 loss 差 mean(L0)−mean(Lz)：原始定义，受跨 t 量纲污染
      drel —— 相对消除率 1 − mean(Lz)/mean(L0)：尺度无关，"z_sem 贡献率"语义。
              必须用 ratio-of-means（分子分母分别平均再相除）——逐样本比
              (L0−Lz)/L0 在小 t 处被 L0≈0 的样本炸掉（有偏且方差发散）。
              注：候选③"D/EMA(L0)"与本式逐点相等（(E[L0]−E[Lz])/E[L0]），不单列
      deff —— 效应量 (mean(L0)−mean(Lz))/std(L0)：量纲与可靠性合并为一个量
      dx0  —— gap = median‖p_z−p_c‖²：线上控制器 d_table 在 x0 模式下
              实际使用的量，裁判实验的"现行行为"对照
    Returns: dict —— t/logsnr/上述四键 [T]；per_image_d [T,B]（图级分析用）
    """
    unet, sampler = model.diffae_unet, model.sampler
    x, cond = batch['x_diffae'], batch['cond']
    device = x.device
    B = x.shape[0]
    ts = np.array(sampler.use_timesteps)
    T = len(ts)

    # CRN：固定种子预抽一组噪声，所有 t 共用（CPU 侧生成，跨 ckpt 一致）
    gen = torch.Generator().manual_seed(seed)
    noises = [torch.randn(x.shape, generator=gen).to(device)
              for _ in range(n_noise)]

    L0 = np.zeros((T, n_noise, B))
    LZ = np.zeros((T, n_noise, B))
    GAP = np.zeros((T, n_noise, B))

    for ti, t_val in enumerate(ts):
        t = torch.full((B,), int(t_val), device=device, dtype=torch.long)
        for j, noise in enumerate(noises):
            x_t = sampler.q_sample(x, t, noise)
            p_c = unet(x_t, t, cond)
            p_z = unet(x_t, t, torch.zeros_like(cond))
            L0[ti, j] = (p_z - x).pow(2).flatten(1).mean(1).cpu().numpy()
            LZ[ti, j] = (p_c - x).pow(2).flatten(1).mean(1).cpu().numpy()
            GAP[ti, j] = (p_z - p_c).pow(2).flatten(1).mean(1).cpu().numpy()

    d = L0 - LZ                                    # [T, R, B]
    l0m = L0.mean(axis=(1, 2))
    lzm = LZ.mean(axis=(1, 2))
    l0s = L0.reshape(T, -1).std(axis=1, ddof=1)
    out = {
        'dabs': d.mean(axis=(1, 2)),
        'drel': 1.0 - lzm / np.maximum(l0m, 1e-12),
        'deff': (l0m - lzm) / np.maximum(l0s, 1e-12),
        'dx0': np.median(GAP.reshape(T, -1), axis=1),
    }
    ab = sampler.alphas_cumprod.detach().cpu().numpy()
    logsnr = np.log(ab / (1.0 - ab))[ts]
    return {'t': ts, 'logsnr': logsnr,
            'per_image_d': d.mean(axis=1), **out}


# =====================================================================
# 测量 2：meta-gradient 谱（Dt 价值的金标准）
# =====================================================================
def measure_meta_grad_spectrum(model, batch, n_noise=8, seed=0):
    """
    一阶 meta gradient（foMAML 形式）：把 t 的采样权重 w(t) 视为外层变量，
        ∂L_task(θ - α Σ_t w_t ∇L_diff(t)) / ∂w_t  ∝  -⟨∇L_task, ∇L_diff(t)⟩
    内积越大（越同向）= 在该 t 上去噪训练对下游任务的真实增益越大。
    与 Dt 谱的相关性越高，Dt 作为"下游贡献度代理"的价值越高。

    方差控制（Dt 的差分配对天然免疫噪声实现，红线是绝对量、需人工补）：
      - 预抽 n_noise 个噪声、全部桶共用（common random numbers）：
        桶间比较不再含"换了噪声"的成分，每桶估计仍无偏；
      - 每桶对 n_noise 个梯度取平均再做内积（线性等价于平均内积），
        单桶方差 ÷ n_noise；
      - 噪声来自独立种子的专用生成器，与 Dt 测量的噪声无关——
        两条谱共用噪声会让波动同向、虚增相关性，独立采样是保守方向。

    Returns: dict —— rep_t / rep_logsnr / bin_edges / meta / meta_cos
    """
    sampler, unet = model.sampler, model.diffae_unet
    probe = model.t_adaptive.probe
    x = batch['x_diffae']
    device = x.device
    B = x.shape[0]

    params = [p for n_, p in model.named_parameters()
              if p.requires_grad and ('backbone' in n_ or 'diffae_enc_compress' in n_)]

    # g_task：一次反传
    task_loss = make_task_loss_fn(model, batch['xis'], batch['dsis'])()
    g_task = torch.autograd.grad(task_loss, params, allow_unused=True)
    g_task = [torch.zeros_like(p) if g is None else g.detach()
              for p, g in zip(params, g_task)]
    norm_task = torch.sqrt(sum((g ** 2).sum() for g in g_task)).clamp_min(1e-12)

    # cond 带梯度、只算一次；各桶共享其子图，反传 retain_graph 保住它
    cond = make_cond_fn(model, batch['xis'])()

    # CRN：固定种子预抽一组噪声，所有桶共用（CPU 侧生成，与设备/全局 RNG 解耦）
    gen = torch.Generator().manual_seed(seed)
    noises = [torch.randn(x.shape, generator=gen).to(device)
              for _ in range(n_noise)]

    meta, meta_cos = [], []
    for bi in range(probe.num_bins):
        t_val = int(probe.bin_rep_t[bi])
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        g_acc = None
        for noise in noises:
            x_t = sampler.q_sample(x, t, noise)
            pred = unet(x_t, t, cond)
            target = x if getattr(sampler, 'predict_x0', False) else noise
            l_diff = F.mse_loss(pred, target)
            g = torch.autograd.grad(l_diff, params, allow_unused=True,
                                    retain_graph=True)
            g = [torch.zeros_like(p) if gg is None else gg.detach()
                 for p, gg in zip(params, g)]
            g_acc = g if g_acc is None else [a + b for a, b in zip(g_acc, g)]
        g_diff = [a / n_noise for a in g_acc]

        ip = sum((gt * gd).sum() for gt, gd in zip(g_task, g_diff))
        norm_diff = torch.sqrt(sum((g ** 2).sum() for g in g_diff)).clamp_min(1e-12)
        meta.append(ip.item())
        meta_cos.append((ip / (norm_task * norm_diff)).item())

    rep_t = probe.bin_rep_t.detach().cpu().numpy().astype(int)
    ab = sampler.alphas_cumprod.detach().cpu().numpy()
    rep_logsnr = np.log(ab / (1.0 - ab))[rep_t]
    return {'rep_t': rep_t, 'rep_logsnr': rep_logsnr,
            'bin_edges': probe.bin_edges.detach().cpu().numpy(),
            'meta': np.array(meta), 'meta_cos': np.array(meta_cos)}


# 参与可视化的 Dt 变体：loss 差直接 + 三个候选/对照形式（x0 预测唯一参数化）
DT_KEYS = ('dabs', 'drel', 'deff', 'dx0')
DT_LABELS = {'dabs': 'D_abs = mean(L0)−mean(Lz)',
             'drel': 'D_rel = 1−mean(Lz)/mean(L0)',
             'deff': 'D_eff = D/σ(L0)',
             'dx0': 'D_x0 gap = median‖p_z−p_c‖²（线上口径）'}


def dt_meta_correlation(dt, mg, keys=DT_KEYS, n_boot=1000, seed=0):
    """Dt 各变体与 meta-grad 的 Spearman / Pearson。
    Dt 侧按桶内平均：用在线探针注册的 bin_edges（logSNR 分位桶）把全网格
    Dt 归入 16 桶取均值——与在线 d_table 的口径一致，且消掉单点取值噪声。
    n_boot>0 时按桶重采样（bootstrap）给 Spearman 的 95% CI——相关本身
    的置信区间，两轴的联合噪声都被捕捉。
    Returns: {key: (rho_s, p_s, rho_p, (ci_lo, ci_hi))}"""
    edges = mg['bin_edges']                 # [num_bins+1]，logSNR 升序
    ls = dt['logsnr']
    K = len(edges) - 1
    rng = np.random.default_rng(seed)
    out = {}
    for key in keys:
        bin_v = []
        for i in range(K):
            m = (ls >= edges[i]) & (ls <= edges[i + 1]) if i == K - 1 \
                else (ls >= edges[i]) & (ls < edges[i + 1])
            if m.any():
                bin_v.append(dt[key][m].mean())
            else:                            # 桶内无网格点（不应发生），退回最近点
                bin_v.append(dt[key][int(np.argmin(np.abs(dt['t'] - mg['rep_t'][i])))])
        bin_v = np.array(bin_v)
        rho_s, p_s = spearman(bin_v, mg['meta'])
        rho_p = pearson(bin_v, mg['meta'])
        ci = (float('nan'), float('nan'))
        if n_boot > 0 and K >= 3:
            boots = []
            for _ in range(n_boot):
                idx = rng.integers(0, K, K)
                r, _ = spearman(bin_v[idx], mg['meta'][idx])
                if np.isfinite(r):
                    boots.append(r)
            if len(boots) >= 50:
                ci = tuple(float(v) for v in np.percentile(boots, [2.5, 97.5]))
        out[key] = (rho_s, p_s, rho_p, ci)
    return out


# =====================================================================
# 测量 3：t_weight 快照（从 ckpt buffer 读，无需任何前向）
# =====================================================================
def read_tweight_snapshot(model):
    """从当前模型读 t 自适应状态。Returns: dict —— t_weight / d_table / rep_t / logsnr / entropy"""
    w = model.t_weight.detach().float().cpu().numpy()
    d = model.d_table.detach().float().cpu().numpy()
    rep_t = model.bin_rep_t.detach().cpu().numpy().astype(int)
    ab = model.sampler.alphas_cumprod.detach().float().cpu().numpy()
    logsnr = np.log(ab / (1.0 - ab))[rep_t]
    ent = float(-(w * np.log(np.clip(w, 1e-12, None))).sum() / np.log(len(w)))
    return {'t_weight': w, 'd_table': d, 'rep_t': rep_t, 'logsnr': logsnr,
            'entropy': ent}


# =====================================================================
# 测量 4：蒙块选址（用模型自己的 mask_selector 现采，与训练同一套逻辑）
# =====================================================================
def measure_mask_selection(model, batch, num_images=4):
    """
    seg 输入梯度热力图 + mask_selector 现采蒙块。
    Returns: dict —— img [n,3,S,S] (0~1) / heat [n,1,S,S]（归一化）/ keep [n,1,S,S]（有效保留率）
    """
    xis = batch['xis'][:num_images]
    dsis = batch['dsis'][:num_images]
    device = xis[0].device
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    heats, imgs = [], []
    for xi, dsi in zip(xis, dsis):
        xg = xi.detach().requires_grad_(True)
        with torch.enable_grad():
            feats = model.backbone(xg)
            losses = model._decode_head_forward_train(feats, [dsi])
            task = sum(v for k, v in losses.items()
                       if isinstance(v, torch.Tensor) and 'loss' in k)
            gx = torch.autograd.grad(task, xg)[0]
        h = gx.abs().mean(dim=1, keepdim=True)
        h = F.adaptive_avg_pool2d(h, model.image_size)
        heats.append(h.detach())
        with torch.no_grad():
            img01 = ((xi.detach() * std + mean) / 255.0).clamp(0, 1)
            imgs.append(F.interpolate(img01, size=(model.image_size, model.image_size),
                                      mode='bilinear', align_corners=False))

    heat = torch.cat(heats, 0)
    img = torch.cat(imgs, 0)
    heat_n = heat / heat.flatten(1).max(dim=1).values.view(-1, 1, 1, 1).clamp_min(1e-8)
    with torch.no_grad():
        keep = model.mask_selector.select(heat)          # 0/1 keep-mask
    if model.mask_applier.soft:                          # 软蒙：蒙块处保留 1-alpha
        keep = 1.0 - model.mask_applier.alpha * (1.0 - keep)
    return {'img': img, 'heat': heat_n, 'keep': keep}


# =====================================================================
# 测量 5：固定 t 的 loss 方差分解（测量噪声 vs 任务本质不稳定）
# =====================================================================
def _msd(v):
    """mean / std(ddof=1) / CV / SNR。CV、SNR 在均值近零时给 nan（D 的均值可跨 0，
    此时 CV 无意义，应以 SNR 为判据）。"""
    v = np.asarray(v, float)
    m = float(v.mean())
    s = float(v.std(ddof=1)) if v.size > 1 else 0.0
    cv = s / abs(m) if abs(m) > 1e-12 else float('nan')
    snr = m / s if s > 1e-12 else float('nan')
    return m, s, cv, snr


@torch.no_grad()
def measure_loss_variance(model, batch, n_t=32, n_eps_a=32, n_noise_b=8,
                          n_rounds_c=8, arm_a_images=16, chunk=16,
                          use_ema=True, seed=0):
    """
    固定 t、重复采样 (x₀, ε) 的 loss 分布测量——回答"大 t 处 Dt 曲线抖动是
    测量噪声还是去噪任务本质不稳定"。

    关键陷阱：loss 的绝对量级随 t 单调增长（可差一个数量级以上），因此
    "原始方差随 t 增大"数学上必然成立、什么都证明不了。必须以相对量为判据：
      CV  = std / |mean|   （变异系数，消除量纲；L₀/Lz 用）
      SNR = mean / std     （D 的均值可跨 0，CV 失效，D 以 SNR 为判据）

    三个变量臂（方差分解）：
      A —— 固定同一批 x₀（前 arm_a_images 张），仅噪声 ε 随机：噪声采样贡献
      B —— 固定同一个 ε（广播到整批），仅数据 x₀ 随机：数据多样性贡献
      C —— x₀ 与 ε 都随机：训练时实际面对的总方差（主结果）
    两分支配对：每次测量用同一个 x_t（同 x₀、同 ε）跑 cond / zero-cond 两次
    前向。Var(D) = Var(L₀)+Var(Lz)−2Cov，配对保留的协方差正是 D 方差远小于
    独立假设的主因——不可拆成两次独立采样。

    模型冻结：全程 eval +（可选）EMA shadow，消除训练动态混淆；
    cond 由当前 ckpt 的 encoder 现算（与训练同分布；不能用建数时随机
    初始化 backbone 算出的 cond）。
    噪声 / 分组由独立种子的专用生成器产生，跨 ckpt 完全相同（CRN），
    保证早/中/晚三条曲线可比。

    Args:
        n_t:         t 网格取点数（在训练网格 use_timesteps 上等距取，20~50 为宜）
        n_eps_a:     臂 A 每图的噪声重复数（≥32 组内方差才稳）
        n_noise_b:   臂 B 的固定 ε 个数（每个 ε 扫全部 N 张图）
        n_rounds_c:  臂 C 轮数（每轮扫全部 N 张图，每块新噪声；总样本 = 轮×N）
        arm_a_images: 臂 A 固定的图像数 Ba
        chunk:       前向分块大小（显存保护）

    Returns: dict ——
      t / logsnr: [n_t]
      arm_{a,b,c}: {l0_mean,l0_std,l0_cv, lz_mean,lz_std,lz_cv,
                    d_mean,d_std,d_cv,d_snr,
                    gap_mean,gap_std,gap_cv,gap_snr} 各 [n_t]
        d  = L(zero) − L(z_sem)（loss 差形式，原生参数化空间，无 eps 增益）；
        gap = ‖p_z − p_c‖²（预测差形式，x0 模式下在线 d_table 实际喂控制器的量）。
        A/B 的 std 为组内均值的平方根（纯噪声 / 纯数据成分）；C 为全样本池化。
      arm_c 另有 corr_l0_lz [n_t]（两分支配对相关，>0.9 说明共享 x_t 主导）
      decomp: {var_a, var_b, var_c, interaction} [n_t]
        在与臂 A 相同的图像集上：var_c ≈ var_a + var_b + interaction
      raw: {Da [n_t,n_eps_a,Ba], Db [n_t,n_noise_b,N], Dc [n_t,R],
            dc_img [R], L0c [n_t,R], Lzc [n_t,R], Gc [n_t,R]}（供复核/再分析）
    """
    unet, sampler = model.diffae_unet, model.sampler
    device = batch['x_diffae'].device
    x_all = batch['x_diffae']
    N = x_all.shape[0]
    Ba = min(arm_a_images, N)
    predict_x0 = getattr(sampler, 'predict_x0', False)

    # t 网格：训练网格上等距取 n_t 个点（值去重，保序）
    grid = np.array(sampler.use_timesteps, dtype=np.int64)
    idx = np.unique(np.linspace(0, len(grid) - 1, n_t).round().astype(int))
    t_list = grid[idx]

    # 全部 N 张图的 cond 逐张现算（当前 ckpt 的 encoder，与训练同分布）。
    # val 图尺寸不一（如 683/703），xis 只能逐图过 backbone，不能 cat 成批。
    cond_all = []
    for i in range(N):
        xi = batch['xis'][i].to(device)
        cond_all.append(model.diffae_enc_compress(model.encode_diffae(xi)))
    cond_all = torch.cat(cond_all, 0)

    # 专用 RNG（CPU 侧生成再上传，与全局/设备 RNG 解耦；跨 ckpt 完全一致）
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to(device)

    def paired_losses(x, cond, t_val, noise):
        """同一 x_t 上的配对前向。Returns: (l_c, l_z, gap) 逐样本 [B]。
        l_c/l_z —— 网络原生参数化空间的 loss（predict_x0 时目标是 x，否则是 ε），
            即训练 loss 实际所在的空间；D = l_z − l_c 是"loss 差"形式的 Dt。
        gap —— ‖p_z − p_c‖²，"预测差"形式的 Dt：x0 模式下在线控制器的
            d_table 实际喂的就是它（t_adaptive.py DtProbe.measure），
            与 loss 差形式并存测量，两种口径的 CV/SNR 都能看。"""
        t = torch.full((x.shape[0],), int(t_val), device=device, dtype=torch.long)
        x_t = sampler.q_sample(x, t, noise)
        p_c = unet(x_t, t, cond)
        p_z = unet(x_t, t, torch.zeros_like(cond))
        target = x if predict_x0 else noise
        l_c = (p_c - target).pow(2).flatten(1).mean(1)
        l_z = (p_z - target).pow(2).flatten(1).mean(1)
        gap = (p_z - p_c).pow(2).flatten(1).mean(1)
        return l_c, l_z, gap

    if use_ema and getattr(model, '_ema_enabled', False):
        model.ema.apply_shadow()
    try:
        Da = np.zeros((len(t_list), n_eps_a, Ba))
        L0a = np.zeros_like(Da)
        Lza = np.zeros_like(Da)
        Ga = np.zeros_like(Da)
        Db = np.zeros((len(t_list), n_noise_b, N))
        L0b = np.zeros_like(Db)
        Lzb = np.zeros_like(Db)
        Gb = np.zeros_like(Db)
        R = n_rounds_c * N
        Dc = np.zeros((len(t_list), R))
        L0c = np.zeros_like(Dc)
        Lzc = np.zeros_like(Dc)
        Gc = np.zeros_like(Dc)
        dc_img = np.zeros(R, dtype=np.int64)

        for ti, t_val in enumerate(t_list):
            # ---- 臂 A：固定 x₀ 批，仅 ε 随机 ----
            xa, ca = x_all[:Ba], cond_all[:Ba]
            for j in range(n_eps_a):
                l_c, l_z, g = paired_losses(xa, ca, t_val, randn(xa.shape))
                L0a[ti, j] = l_z.cpu().numpy()   # 记法统一：L0 = 零条件分支
                Lza[ti, j] = l_c.cpu().numpy()
                Da[ti, j] = (l_z - l_c).cpu().numpy()
                Ga[ti, j] = g.cpu().numpy()

            # ---- 臂 B：固定 ε（广播），仅 x₀ 随机 ----
            for j in range(n_noise_b):
                noise1 = randn(1, *x_all.shape[1:])
                lzs, lcs, gs = [], [], []
                for i0 in range(0, N, chunk):
                    x = x_all[i0:i0 + chunk]
                    l_c, l_z, g = paired_losses(
                        x, cond_all[i0:i0 + chunk], t_val,
                        noise1.expand(x.shape[0], -1, -1, -1))
                    lcs.append(l_c.cpu().numpy())
                    lzs.append(l_z.cpu().numpy())
                    gs.append(g.cpu().numpy())
                L0b[ti, j] = np.concatenate(lzs)
                Lzb[ti, j] = np.concatenate(lcs)
                Db[ti, j] = L0b[ti, j] - Lzb[ti, j]
                Gb[ti, j] = np.concatenate(gs)

            # ---- 臂 C：x₀、ε 都随机（每块每轮新噪声）----
            r = 0
            for _ in range(n_rounds_c):
                for i0 in range(0, N, chunk):
                    x = x_all[i0:i0 + chunk]
                    l_c, l_z, g = paired_losses(
                        x, cond_all[i0:i0 + chunk], t_val, randn(x.shape))
                    k = x.shape[0]
                    L0c[ti, r:r + k] = l_z.cpu().numpy()
                    Lzc[ti, r:r + k] = l_c.cpu().numpy()
                    Dc[ti, r:r + k] = L0c[ti, r:r + k] - Lzc[ti, r:r + k]
                    Gc[ti, r:r + k] = g.cpu().numpy()
                    dc_img[r:r + k] = np.arange(i0, i0 + k)
                    r += k
    finally:
        if use_ema and getattr(model, '_ema_enabled', False):
            model.ema.restore()

    # ---- 统计 ----
    def arm_stats(mat0, matz, group_axis, matg):
        """mat: [n_t, n_rep, n_group]。A/B 取组内 std 的均方根（纯单变量成分），
        C 传 group_axis=None 表示全池化。matg = gap（预测差）矩阵，同口径统计。"""
        d = mat0 - matz
        out = {k: np.zeros(len(t_list)) for k in
               ('l0_mean', 'l0_std', 'l0_cv', 'lz_mean', 'lz_std', 'lz_cv',
                'd_mean', 'd_std', 'd_cv', 'd_snr',
                'gap_mean', 'gap_std', 'gap_cv', 'gap_snr')}
        for ti in range(len(t_list)):
            for key, m in (('l0', mat0[ti]), ('lz', matz[ti]),
                           ('d', d[ti]), ('gap', matg[ti])):
                mean = float(m.mean())
                if group_axis is None:
                    std = float(m.std(ddof=1)) if m.size > 1 else 0.0
                else:  # 组内方差对组取平均再开根：纯单变量成分
                    std = float(np.sqrt(m.std(axis=group_axis, ddof=1)
                                        .mean())) if m.shape[group_axis] > 1 else 0.0
                out[f'{key}_mean'][ti] = mean
                out[f'{key}_std'][ti] = std
                out[f'{key}_cv'][ti] = std / abs(mean) if abs(mean) > 1e-12 \
                    else float('nan')
                if key in ('d', 'gap'):
                    out[f'{key}_snr'][ti] = mean / std if std > 1e-12 else float('nan')
        return out

    arm_a = arm_stats(L0a, Lza, 0, Ga)          # 噪声为重复轴 -> 纯 ε 成分
    arm_b = arm_stats(L0b, Lzb, 1, Gb)          # 图像为重复轴 -> 纯 x₀ 成分
    arm_c = arm_stats(L0c, Lzc, None, Gc)
    # 配对相关（臂 C）：两分支共享 x_t，loss 强正相关
    corr = np.array([pearson(L0c[ti], Lzc[ti]) for ti in range(len(t_list))])
    arm_c['corr_l0_lz'] = corr

    # 方差分解（与臂 A 同一图像集：前 Ba 张）
    same = dc_img < Ba
    decomp = {k: np.zeros(len(t_list)) for k in
              ('var_a', 'var_b', 'var_c', 'interaction')}
    for ti in range(len(t_list)):
        var_a = float(Da[ti].std(axis=0, ddof=1).mean() ** 2)
        var_b = float(Db[ti, :, :Ba].std(axis=1, ddof=1).mean() ** 2)
        var_c = float(Dc[ti, same].std(ddof=1) ** 2)
        decomp['var_a'][ti] = var_a
        decomp['var_b'][ti] = var_b
        decomp['var_c'][ti] = var_c
        decomp['interaction'][ti] = var_c - var_a - var_b

    ab = sampler.alphas_cumprod.detach().cpu().numpy()
    logsnr = np.log(ab / (1.0 - ab))[t_list]
    return {'t': t_list, 'logsnr': logsnr,
            'arm_a': arm_a, 'arm_b': arm_b, 'arm_c': arm_c,
            'decomp': decomp,
            'raw': {'Da': Da, 'Db': Db, 'Dc': Dc, 'dc_img': dc_img,
                    'L0c': L0c, 'Lzc': Lzc, 'Gc': Gc}}


def measure_grad_cos_variance(model, batch, n_t=16, n_noise=16, n_images=8,
                              seed=0):
    """
    可选第四臂：固定 t 下 cos<g_task, g_diff(t)> 的方差——与 meta-grad 红线
    的抖动直接对应。g_task 一次反传固定；g_diff(t) 对每个噪声实现独立反传，
    逐个实现算 cos 再统计分布（不是平均梯度后再 cos——那会把方差平均掉）。

    代价 = n_t × n_noise 次 backbone 反传，建议小网格（16×16）起步。
    Returns: dict —— t / logsnr / cos_mean / cos_std / cos_cv / cos_all [n_t,n_noise]
    """
    sampler, unet = model.sampler, model.diffae_unet
    device = batch['x_diffae'].device
    x = batch['x_diffae'][:n_images]
    xis, dsis = batch['xis'][:n_images], batch['dsis'][:n_images]
    predict_x0 = getattr(sampler, 'predict_x0', False)

    grid = np.array(sampler.use_timesteps, dtype=np.int64)
    idx = np.unique(np.linspace(0, len(grid) - 1, n_t).round().astype(int))
    t_list = grid[idx]

    params = [p for n_, p in model.named_parameters()
              if p.requires_grad and ('backbone' in n_ or 'diffae_enc_compress' in n_)]

    # g_task：一次反传，固定
    task_loss = make_task_loss_fn(model, xis, dsis)()
    g_task = torch.autograd.grad(task_loss, params, allow_unused=True)
    g_task = [torch.zeros_like(p) if g is None else g.detach()
              for p, g in zip(params, g_task)]
    norm_task = torch.sqrt(sum((g ** 2).sum() for g in g_task)).clamp_min(1e-12)

    gen = torch.Generator().manual_seed(seed)
    cos_all = np.zeros((len(t_list), n_noise))
    for ti, t_val in enumerate(t_list):
        # cond 带梯度、每 t 算一次；同 t 的 n_noise 次反传共享其子图
        cond = make_cond_fn(model, xis)()
        t = torch.full((x.shape[0],), int(t_val), device=device, dtype=torch.long)
        for j in range(n_noise):
            noise = torch.randn(x.shape, generator=gen).to(device)
            x_t = sampler.q_sample(x, t, noise)
            pred = unet(x_t, t, cond)
            target = x if predict_x0 else noise
            l_diff = F.mse_loss(pred, target)
            g = torch.autograd.grad(l_diff, params, allow_unused=True,
                                    retain_graph=True)
            g = [torch.zeros_like(p) if gg is None else gg.detach()
                 for p, gg in zip(params, g)]
            ip = sum((gt * gd).sum() for gt, gd in zip(g_task, g))
            norm_d = torch.sqrt(sum((gg ** 2).sum() for gg in g)).clamp_min(1e-12)
            cos_all[ti, j] = (ip / (norm_task * norm_d)).item()
        del cond
        torch.cuda.empty_cache()

    cos_mean = cos_all.mean(1)
    cos_std = cos_all.std(1, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_cv = np.where(np.abs(cos_mean) > 1e-12,
                          cos_std / np.abs(cos_mean), np.nan)
    ab = sampler.alphas_cumprod.detach().cpu().numpy()
    logsnr = np.log(ab / (1.0 - ab))[t_list]
    return {'t': t_list, 'logsnr': logsnr, 'cos_mean': cos_mean,
            'cos_std': cos_std, 'cos_cv': cos_cv, 'cos_all': cos_all}


def lossvar_verdict(res):
    """按判定标准汇总读数（臂 C 为主）：
      rho_cv_* : CV 与 t 的 Spearman 相关——不显著 ≠ 平，但显著 >0 才是"随 t 上升"
      min_snr_high_t: 大 t 半区 |SNR(D)| 最小值——< 1 视为该处信号被噪声淹没
    判定映射：
      CV 随 t 基本平          -> 抖动是量纲效应，控制器不用改，平滑即可
      CV 升但 SNR 不崩        -> 测量端加 EMA/多采样即可
      CV 升且大 t SNR 崩溃    -> "粒度过大"假说成立，SNR(t) 可直接标定 κ·σ_t
    """
    t = res['t']
    c = res['arm_c']
    hi = t >= np.median(t)
    snr = np.abs(c['d_snr'])
    rho_d, p_d = spearman(t, c['d_cv'])
    rho_lz, p_lz = spearman(t, c['lz_cv'])
    rho_l0, p_l0 = spearman(t, c['l0_cv'])
    return {'rho_cv_d': (rho_d, p_d), 'rho_cv_lz': (rho_lz, p_lz),
            'rho_cv_l0': (rho_l0, p_l0),
            'min_snr_high_t': float(np.nanmin(snr[hi])),
            'min_gap_snr_high_t': float(np.nanmin(np.abs(c['gap_snr'])[hi])),
            'mean_corr_l0_lz': float(np.nanmean(c['corr_l0_lz']))}


# =====================================================================
# 绘图
# =====================================================================
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_dt_vs_meta(dt, mg=None, title='', save_path=None, keys=DT_KEYS):
    """Dt 谱（deps/dx0/cos 三变体 × t/logSNR 两视图）+ meta-grad 谱（红，桶代表 t）
    双 y 轴合图，每行左图标注该变体与 meta-grad 的相关系数。
    Returns: (fig, {key: (rho_s, p_s, rho_p)})"""
    plt = _plt()
    corrs = dt_meta_correlation(dt, mg, keys) if mg is not None else {}

    fig, axes = plt.subplots(len(keys), 2, figsize=(13, 3.6 * len(keys)),
                             squeeze=False)
    for row, key in enumerate(keys):
        for col, (x_dt, x_mg, xlab) in enumerate([
                (dt['t'], mg['rep_t'] if mg else None, 't'),
                (dt['logsnr'], mg['rep_logsnr'] if mg else None, 'log SNR')]):
            ax = axes[row, col]
            ax.plot(x_dt, dt[key], lw=2, color='tab:blue',
                    label=f'{DT_LABELS[key]} (Dt)')
            ax.axhline(0, color='k', ls='--', lw=0.8)
            ax.set_xlabel(xlab)
            ax.set_ylabel(DT_LABELS[key], color='tab:blue')
            ax.grid(alpha=0.3)
            if mg is not None:
                ax2 = ax.twinx()
                ax2.plot(x_mg, mg['meta'], lw=2, marker='o', ms=4,
                         color='tab:red', label='meta-grad <g_task, g_diff>')
                ax2.set_ylabel('meta gradient', color='tab:red')
                lines = ax.get_lines() + ax2.get_lines()
                ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
            else:
                ax.legend(fontsize=8)
            if mg is not None and col == 0:
                rho_s, p_s, rho_p, ci = corrs[key]
                ci_s = f' CI95[{ci[0]:.2f},{ci[1]:.2f}]' if np.isfinite(ci[0]) else ''
                ax.set_title(f'{DT_LABELS[key]} ~ meta-grad: '
                             f'Spearman ρ={rho_s:.3f}{ci_s} (p={p_s:.3f}) | '
                             f'Pearson r={rho_p:.3f}', fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, corrs


def plot_tweight(snap, title='', ts_freq=None, save_path=None):
    """t_weight 柱状 + D 表折线（+ 可选实际采样频率），双面板：t / logSNR。"""
    plt = _plt()
    w, d, rep_t, logsnr = snap['t_weight'], snap['d_table'], snap['rep_t'], snap['logsnr']
    K = len(w)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    d_n = d / np.abs(d).max() * w.max() if np.abs(d).max() > 0 else d
    for ax, x, xlab in [(axes[0], rep_t, 't (bin rep)'),
                        (axes[1], logsnr, 'log SNR (bin rep)')]:
        ax.bar(x, w, width=(x[1] - x[0]) * 0.7 if K > 1 else 10,
               alpha=0.7, label='t_weight')
        ax.plot(x, d_n, color='tab:red', marker='o', ms=3, label='D table (scaled)')
        if ts_freq is not None:
            ax.plot(x, ts_freq, color='tab:green', marker='s', ms=3, lw=1.8,
                    label='realized t freq')
        ax.axhline(1.0 / K, color='k', ls='--', lw=0.8, label='uniform')
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('prob / weight')
    fig.suptitle(f'{title} | entropy={snap["entropy"]:.3f} | '
                 f'max={w.max():.3f}@bin{int(w.argmax())}')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_tweight_evolution(snaps, labels, save_path=None):
    """跨 ckpt 的 t_weight 演化：热力图（行=ckpt，列=桶）+ 熵曲线。"""
    plt = _plt()
    W = np.stack([s['t_weight'] for s in snaps])
    ents = [s['entropy'] for s in snaps]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4),
                             gridspec_kw={'width_ratios': [2.2, 1]})
    im = axes[0].imshow(W, aspect='auto', cmap='viridis')
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xticks(range(W.shape[1]))
    axes[0].set_xticklabels([str(t) for t in snaps[0]['rep_t']], rotation=45, fontsize=7)
    axes[0].set_xlabel('bin rep t')
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].plot(range(len(labels)), ents, 'o-', color='tab:orange')
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('entropy')
    axes[1].grid(alpha=0.3)
    fig.suptitle('t_weight evolution')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_mask_selection(sel, title='', save_path=None):
    """蒙块选址三联图：原图 | seg 梯度热力 | 蒙版叠加（蒙区染红）。"""
    plt = _plt()
    img, heat, keep = sel['img'], sel['heat'], sel['keep']
    n = img.shape[0]
    red = torch.tensor([1.0, 0.1, 0.1], device=img.device).view(1, 3, 1, 1)
    overlay = img * keep + (red.expand_as(img) * 0.55 + img * 0.45) * (1.0 - keep)

    fig, axes = plt.subplots(n, 3, figsize=(10, 3.4 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        axes[i, 0].imshow(img[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title('image')
        im = axes[i, 1].imshow(heat[i, 0].cpu().numpy(), cmap='turbo', vmin=0, vmax=1)
        axes[i, 1].set_title('seg heat')
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046)
        axes[i, 2].imshow(overlay[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 2].set_title(f'mask (kept={keep[i].mean().item():.2f})')
        for j in range(3):
            axes[i, j].axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def save_recon(model, batch, save_path, max_samples=4):
    """重建四列对比（原图 | 反演 | 重建 | 仅 z_sem 生成），走 mixin 接口。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_diffae_visualization(
        batch['x_diffae'], save_path,
        max_samples=max_samples, cond=batch['cond'])
    return save_path


ARM_LABELS = {'arm_a': 'A: 仅 ε 随机', 'arm_b': 'B: 仅 x₀ 随机',
              'arm_c': 'C: 皆随机(总)'}
ARM_COLORS = {'arm_a': 'tab:orange', 'arm_b': 'tab:green', 'arm_c': 'tab:blue'}


def plot_loss_variance(res, title='', save_path=None):
    """loss 方差分解六联图（x 轴统一为 t，顶行小字标 logSNR）：
      (0,0) loss 量级陷阱：mean L0/Lz 对数轴——原始方差随 t 必增，不能作判据
      (0,1) CV(L0)、CV(Lz)，臂 C 实线 + A/B 虚线（单变量成分对照）
      (0,2) CV(D) 三臂（D 均值跨 0 时 CV 发散，仅作参考，判据看 SNR）
      (1,0) SNR(D) 三臂 + |SNR|=1 参考线（主判据）
      (1,1) Var(D) 方差分解：var_a(ε) + var_b(x₀) + 交互 ≈ var_c
      (1,2) 臂 C 两分支配对相关 corr(L0,Lz)——共享 x_t 使 D 方差远小于独立假设
    """
    plt = _plt()
    t = res['t']
    logsnr = res['logsnr']
    a, b, c, dec = res['arm_a'], res['arm_b'], res['arm_c'], res['decomp']

    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5))

    def _logsnr_top(ax):
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ticks = np.linspace(0, len(t) - 1, 6).round().astype(int)
        ax2.set_xticks(t[ticks])
        ax2.set_xticklabels([f'{logsnr[i]:.1f}' for i in ticks], fontsize=8)
        ax2.set_xlabel('log SNR', fontsize=8)

    # (0,0) 量级陷阱
    ax = axes[0, 0]
    ax.plot(t, c['l0_mean'], color='tab:gray', lw=2, label='mean L(zero)')
    ax.plot(t, c['lz_mean'], color='tab:blue', lw=2, label='mean L(z_sem)')
    ax.set_yscale('log')
    ax.set_title('loss 量级随 t 增长 ⇒ 原始方差必增（无效判据）', fontsize=10)
    ax.set_xlabel('t'); ax.set_ylabel('loss (log)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (0,1) CV(L)
    ax = axes[0, 1]
    ax.plot(t, c['l0_cv'], color='tab:gray', lw=2, label='CV L(zero) [C]')
    ax.plot(t, c['lz_cv'], color='tab:blue', lw=2, label='CV L(z_sem) [C]')
    ax.plot(t, a['lz_cv'], color='tab:orange', lw=1.2, ls='--', label='CV Lz [A:仅ε]')
    ax.plot(t, b['lz_cv'], color='tab:green', lw=1.2, ls='--', label='CV Lz [B:仅x₀]')
    ax.set_title('CV(loss) = std/|mean|（消除量纲后的真判据）', fontsize=10)
    ax.set_xlabel('t'); ax.set_ylabel('CV')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (0,2) CV(D) 三臂
    ax = axes[0, 2]
    for arm in ('arm_a', 'arm_b', 'arm_c'):
        ax.plot(t, res[arm]['d_cv'], color=ARM_COLORS[arm], lw=2,
                label=ARM_LABELS[arm])
    ax.set_title('CV(D)（D 均值跨 0 处发散，仅参考）', fontsize=10)
    ax.set_xlabel('t'); ax.set_ylabel('CV(D)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (1,0) SNR(D) 三臂——主判据
    ax = axes[1, 0]
    for arm in ('arm_a', 'arm_b', 'arm_c'):
        ax.plot(t, np.abs(res[arm]['d_snr']), color=ARM_COLORS[arm], lw=2,
                label=ARM_LABELS[arm])
    ax.plot(t, np.abs(c['gap_snr']), color='tab:red', lw=1.5, ls=':',
            label='SNR(gap=‖p_z−p_c‖²) [C]（在线 d_table 口径）')
    ax.axhline(1.0, color='k', ls='--', lw=1, label='|SNR|=1（信号=噪声）')
    ax.set_title('|SNR(D)| = |mean(D)|/std(D)——控制器实际吃到的信噪比', fontsize=10)
    ax.set_xlabel('t'); ax.set_ylabel('|SNR(D)|')
    ax.set_yscale('log')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (1,1) 方差分解（与臂 A 同一图像集）
    ax = axes[1, 1]
    ax.plot(t, dec['var_c'], color='tab:blue', lw=2, label='Var(D) 总 [C]')
    ax.plot(t, dec['var_a'], color='tab:orange', lw=2, label='Var_ε [A]')
    ax.plot(t, dec['var_b'], color='tab:green', lw=2, label='Var_x0 [B]')
    ax.plot(t, dec['interaction'], color='tab:red', lw=1.5, ls='--',
            label='交互 (C−A−B)')
    ax.set_title('Var(D) 分解：var_c ≈ var_a + var_b + 交互', fontsize=10)
    ax.set_xlabel('t'); ax.set_ylabel('Var(D)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (1,2) 配对相关
    ax = axes[1, 2]
    ax.plot(t, c['corr_l0_lz'], color='tab:purple', lw=2)
    ax.set_ylim(0, 1.02)
    ax.set_title('corr(L0, Lz)：共享 x_t 的配对相关（Cov 是 D 方差小的主因）',
                 fontsize=10)
    ax.set_xlabel('t'); ax.set_ylabel('Pearson r')
    ax.grid(alpha=0.3)

    for ax in axes.flat:
        _logsnr_top(ax)
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_loss_variance_evolution(results, labels, save_path=None):
    """跨 ckpt 演化：臂 C 的 CV(D) / |SNR(D)| 曲线叠加（早/中/晚）。
    方差结构是否随训练演化——早期若 SNR 普遍 <1，预热期不收信号的设计即被证实。"""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for res, lab in zip(results, labels):
        t, c = res['t'], res['arm_c']
        axes[0].plot(t, c['d_cv'], lw=2, marker='o', ms=3, label=lab)
        axes[1].plot(t, np.abs(c['d_snr']), lw=2, marker='o', ms=3, label=lab)
    axes[0].set_title('CV(D) across checkpoints [arm C]')
    axes[0].set_xlabel('t'); axes[0].set_ylabel('CV(D)'); axes[0].grid(alpha=0.3)
    axes[1].axhline(1.0, color='k', ls='--', lw=1)
    axes[1].set_yscale('log')
    axes[1].set_title('|SNR(D)| across checkpoints [arm C]')
    axes[1].set_xlabel('t'); axes[1].set_ylabel('|SNR(D)|'); axes[1].grid(alpha=0.3)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_grad_cos_variance(gres, title='', save_path=None):
    """可选第四臂：cos<g_task, g_diff(t)> 的逐实现分布（mean ± std 带 + 散点）。"""
    plt = _plt()
    t = gres['t']
    m, s = gres['cos_mean'], gres['cos_std']
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(t, m, lw=2, color='tab:red', label='mean cos')
    ax.fill_between(t, m - s, m + s, color='tab:red', alpha=0.25,
                    label='±1 std')
    rng = np.random.default_rng(0)
    for ti in range(len(t)):
        ax.scatter(np.full(gres['cos_all'].shape[1], t[ti]) + rng.uniform(-6, 6, gres['cos_all'].shape[1]),
                   gres['cos_all'][ti], s=6, color='tab:gray', alpha=0.5)
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('t'); ax.set_ylabel('cos<g_task, g_diff>')
    ax.set_title(title or '固定 t 下梯度方向一致性 cos 的分布')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
