# -*- coding: utf-8 -*-
"""
路线A · 步骤2：LMM 合成混合像元模拟数据集

核心公式（逐波段）
    P_mix = m · E_dom + (1 - m) · E_sub
    m = 优势类占比，真值标签 = 优势类

关键设计（避免循环论证）
    端元组  = 每类均质性最好的 10 个样本点   —— 只用于合成
    训练组  = 其余全部样本点                 —— 只用于训练分类器
    两组严格不重叠，杜绝"测试样本来自训练样本"的质疑。

输出 E:\\混合像元实验\\路线A\\simulated.npz
    spec / label / m / pair / dom_idx / sub_idx  模拟像元
    train_spec / train_label                     训练组（严格分离版）
    all_spec / all_label                         训练组（全样本版，模拟系统实际用法）
"""
import os
import numpy as np
from itertools import combinations

OUT_DIR = r'E:/混合像元实验/路线A'
EM_NPZ = os.path.join(OUT_DIR, 'endmembers.npz')
OUT_NPZ = os.path.join(OUT_DIR, 'simulated.npz')

CLASSES = ['人工表面', '水体', '河漫滩', '耕地']
M_LEVELS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
NOISE_SNR = None        # None = 无噪声（主实验）；设为数值则按该 SNR(dB) 加高斯噪声

d = np.load(EM_NPZ, allow_pickle=True)

# ---------- 端元与训练组 ----------
em = {c: d[f'em_{c}'] for c in CLASSES}
em_rc = {c: d[f'em_rc_{c}'] for c in CLASSES}
all_spec, all_lab = [], []
train_spec, train_lab = [], []
print('=' * 78)
print(f'{"类别":<8}{"样本点":>8}{"端元(合成)":>12}{"训练(严格)":>12}{"训练(全样本)":>14}')
print('=' * 78)
for ci, c in enumerate(CLASSES):
    S = d[f'sample_{c}']          # (n, 200)
    RC = d[f'sample_rc_{c}']      # (n, 2)
    rc_em = em_rc[c]
    key_em = set(map(tuple, rc_em))
    mask_train = np.array([tuple(rc) not in key_em for rc in RC])

    all_spec.append(S); all_lab.append(np.full(len(S), ci, np.int8))
    train_spec.append(S[mask_train]); train_lab.append(np.full(mask_train.sum(), ci, np.int8))
    print(f'{c:<8}{len(S):>8}{len(em[c]):>12}{mask_train.sum():>12}{len(S):>14}')
all_spec = np.vstack(all_spec).astype(np.float32)
all_lab = np.concatenate(all_lab)
train_spec = np.vstack(train_spec).astype(np.float32)
train_lab = np.concatenate(train_lab)
print(f'{"合计":<8}{len(all_spec):>8}{sum(len(em[c]) for c in CLASSES):>12}'
      f'{len(train_spec):>12}{len(all_spec):>14}')

# ---------- 合成 ----------
specs, labels, ms, pairs, didx, sidx = [], [], [], [], [], []
for pi, (a, b) in enumerate(combinations(range(len(CLASSES)), 2)):
    for dom, sub in [(a, b), (b, a)]:          # 两个方向都做，保证每类都当优势类
        E_dom, E_sub = em[CLASSES[dom]], em[CLASSES[sub]]
        for m in M_LEVELS:
            mix = m * E_dom[:, None, :] + (1 - m) * E_sub[None, :, :]   # (10, 10, 200)
            n_d, n_s = len(E_dom), len(E_sub)
            specs.append(mix.reshape(-1, mix.shape[-1]))
            labels.append(np.full(n_d * n_s, dom, np.int8))
            ms.append(np.full(n_d * n_s, m, np.float32))
            pairs.append(np.full(n_d * n_s, pi, np.int8))
            didx.append(np.repeat(np.arange(n_d), n_s).astype(np.int8))
            sidx.append(np.tile(np.arange(n_s), n_d).astype(np.int8))

spec = np.vstack(specs).astype(np.float32)
label = np.concatenate(labels)
m_arr = np.concatenate(ms)
pair = np.concatenate(pairs)
dom_idx = np.concatenate(didx)
sub_idx = np.concatenate(sidx)

# 可选噪声
if NOISE_SNR is not None:
    sigma = spec.std(axis=0, keepdims=True) / (10 ** (NOISE_SNR / 20))
    rng = np.random.default_rng(42)
    spec = spec + rng.normal(0, 1, spec.shape).astype(np.float32) * sigma
    print(f'\n已加入高斯噪声 SNR={NOISE_SNR} dB')

# ---------- 纯像元对照（m = 1.0，无混合） ----------
pure_list, pure_lab = [], []
for ci, c in enumerate(CLASSES):
    pure_list.append(em[c]); pure_lab.append(np.full(len(em[c]), ci, np.int8))
pure_spec = np.vstack(pure_list).astype(np.float32)
pure_lab = np.concatenate(pure_lab)

print('\n' + '=' * 78)
print(f'模拟混合像元总数 : {len(spec)}')
print(f'  构成           : 6 类别对 × 2 方向 × {len(M_LEVELS)} 档 × 10×10 端元组合')
print(f'  真值分布       : ' + '  '.join(
    f'{CLASSES[i]}={int((label==i).sum())}' for i in range(len(CLASSES))))
print(f'  纯像元对照     : {len(pure_spec)} 个（每类 10 条端元，m=1.0）')
print('=' * 78)

np.savez_compressed(
    OUT_NPZ,
    spec=spec, label=label, m=m_arr, pair=pair, dom_idx=dom_idx, sub_idx=sub_idx,
    pure_spec=pure_spec, pure_label=pure_lab,
    train_spec=train_spec, train_label=train_lab,
    all_spec=all_spec, all_label=all_lab,
    classes=np.array(CLASSES), m_levels=np.array(M_LEVELS, np.float32),
)
print(f'\n已保存: {OUT_NPZ}  ({os.path.getsize(OUT_NPZ)/1024**2:.1f} MB)')

# ---------- 合理性自查 ----------
print('\n各档模拟像元的"类间可分性"自查（用 200 维欧氏距离衡量）：')
from itertools import combinations as _cb
for m in [0.55, 0.75, 0.95]:
    sel = np.isclose(m_arr, m)
    dom_cls = label[sel]
    # 同一档内，不同真值类别之间的平均距离 / 同类内平均距离
    intra, inter = [], []
    idx_by_cls = {i: np.where(sel & (label == i))[0] for i in range(len(CLASSES))}
    for i in range(len(CLASSES)):
        s = spec[idx_by_cls[i]]
        if len(s) > 1:
            dd = np.linalg.norm(s[:80, None, :] - s[None, :80, :], axis=2)
            intra.append(dd[np.triu_indices(len(s[:80]), 1)].mean())
    for i, j in _cb(range(len(CLASSES)), 2):
        a, b = spec[idx_by_cls[i]][:80], spec[idx_by_cls[j]][:80]
        inter.append(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).mean())
    print(f'  m={m:.2f}: 类内平均距离={np.mean(intra):>9.1f}  类间平均距离={np.mean(inter):>9.1f}  '
          f'可分性比={np.mean(inter)/np.mean(intra):.3f}')
