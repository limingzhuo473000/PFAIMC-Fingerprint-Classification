# -*- coding: utf-8 -*-
"""
路线A · 步骤9：匹配决策规则改进实验（仍是"指纹+汉明"框架内）

背景：主实验显示最近邻单模板匹配在混合像元上落后判别式方法 5~7pp。
测试三个框架内改进：
  V1 kNN 聚合     —— 每类取 k 近邻距离均值，平滑"模板枢纽"效应
  V2 类中心匹配   —— 与类 token 均值（原型）比汉明距离
  V3 kNN+ margin  —— V1 基础上加"前两类距离接近则弃权"（可靠性机制）
输出 variants_results.npz
"""
import os
import numpy as np

OUT_DIR = r'E:/混合像元实验/路线A'
MR = np.load(os.path.join(OUT_DIR, 'main_results.npz'), allow_pickle=True)
EM = np.load(os.path.join(OUT_DIR, 'endmembers.npz'), allow_pickle=True)

CLASSES = list(MR['classes']); NC = len(CLASSES)
M_LEVELS = MR['m_levels']
te_lab = MR['label']; m_arr = MR['m']
train_spec = MR['train_spec']; train_lab = MR['train_lab']
thrT = EM['cube_thr'].T.astype(np.float32)


def discretize(x):
    x = np.asarray(x, np.float32)
    t = np.ones(x.shape, np.int8)
    for k in range(thrT.shape[0]):
        t += (x > thrT[k][None, :]).astype(np.int8)
    return t


def oa(p, y):
    return (p == y).mean() * 100


tr_tok = discretize(train_spec.astype(np.float32))
te_tok = discretize(None) if False else None
# 重新离散化测试集（main_results 未存 spec）
SIM_OLD = np.load(os.path.join(OUT_DIR, 'simulated.npz'), allow_pickle=True)
te_tok = discretize(MR['train_spec'].astype(np.float32))  # 占位防误用
del te_tok

# 从 npz 恢复测试集：main_results 没存 spec，改由端元重新合成（与步骤8完全一致）
from itertools import combinations
seed_spec, seed_lab = [], []
for ci, c in enumerate(CLASSES):
    em = EM[f'em_{c}']
    seed_spec.append(em[:5]); seed_lab.append(np.full(5, ci, np.int8))
seed_spec = np.vstack(seed_spec).astype(np.float32); seed_lab = np.concatenate(seed_lab)
specs, labels, ms = [], [], []
M_LEVELS_L = list(M_LEVELS)
for pi, (a, b) in enumerate(combinations(range(NC), 2)):
    for dom, sub in [(a, b), (b, a)]:
        E_d, E_s = seed_spec[seed_lab == dom], seed_spec[seed_lab == sub]
        for m in M_LEVELS_L:
            mix = m * E_d[:, None, :] + (1 - m) * E_s[None, :, :]
            specs.append(mix.reshape(-1, mix.shape[-1]))
            labels.append(np.full(mix.shape[0] * mix.shape[1], dom, np.int8))
            ms.append(np.full(mix.shape[0] * mix.shape[1], m, np.float32))
spec = np.vstack(specs).astype(np.float32)
label2 = np.concatenate(labels)
assert (label2 == te_lab).all(), '合成顺序与主实验不一致'
te_tok = discretize(spec)

D = (te_tok[:, None, :] != tr_tok[None, :, :]).sum(axis=2).astype(np.float32)  # (2700,150)
print(f'距离矩阵 {D.shape}')


def tier_row(pred):
    return [oa(pred[np.isclose(m_arr, m)], te_lab[np.isclose(m_arr, m)]) for m in M_LEVELS]


results = {}

# V1: kNN 聚合
for K in [3, 5, 7]:
    pred = np.empty(len(te_tok), np.int8)
    idx_sorted = np.argsort(D, axis=1)
    for i in range(len(te_tok)):
        scores = []
        for c in range(NC):
            cand = [j for j in idx_sorted[i] if train_lab[j] == c][:K]
            scores.append(D[i, cand].mean())
        pred[i] = int(np.argmin(scores))
    results[f'V1-kNN(k={K})'] = pred

# V2: 类中心匹配
proto = np.stack([np.rint(tr_tok[train_lab == c].mean(0)).astype(np.int8) for c in range(NC)])
Dp = (te_tok[:, None, :] != proto[None, :, :]).sum(axis=2)
results['V2-类中心'] = Dp.argmin(1).astype(np.int8)

# V3: kNN(5) + margin 弃权
K = 5
idx_sorted = np.argsort(D, axis=1)
pred3 = np.empty(len(te_tok), np.int8)
n_rej = 0
for i in range(len(te_tok)):
    scores = np.empty(NC)
    for c in range(NC):
        cand = [j for j in idx_sorted[i] if train_lab[j] == c][:K]
        scores[c] = D[i, cand].mean()
    o = np.argsort(scores)
    if scores[o[1]] - scores[o[0]] < 1.0:      # margin 阈值（汉明均值差 <1 位差）
        pred3[i] = -1; n_rej += 1
    else:
        pred3[i] = o[0]
results['V3-kNN+margin'] = pred3

print('\n' + '=' * 106)
print('匹配规则变体 × 混合档 OA (%)   [对照: 原始 Hamming-NN / RF]')
print('=' * 106)
ref = {'Hamming-NN(原始)': MR['pred_Hamming-NN'], 'RF': MR['pred_RF']}
show = {**ref, **results}
hdr = f'{"规则":<18}' + ''.join(f'{m:>8.2f}' for m in M_LEVELS) + f'{"总体":>9}'
print(hdr); print('-' * 106)
for a, p in show.items():
    row = tier_row(p)
    print(f'{a:<18}' + ''.join(f'{v:>8.2f}' for v in row) + f'{oa(p, te_lab):>9.2f}'
          + (f'   弃权{int((p==-1).sum())}' if (p == -1).any() else ''))

np.savez_compressed(os.path.join(OUT_DIR, 'variants_results.npz'),
                    **{f'pred_{k}': v for k, v in results.items()},
                    rejec_v3=n_rej)
print('\n已保存: variants_results.npz')
