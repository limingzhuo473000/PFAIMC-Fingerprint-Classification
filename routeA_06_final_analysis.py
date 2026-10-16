# -*- coding: utf-8 -*-
"""
路线A · 步骤6：最终分析 —— 分档精度 + 弃权分解 + McNemar 显著性检验

问题定位：真实影像上 Hamming-NN 92.66%（零弃权），模拟集上 77.91%。
本步骤查明模拟集差距来源，并产出论文级分档精度表。
"""
import os
import numpy as np
from itertools import combinations
from scipy.stats import binomtest

OUT_DIR = r'E:/混合像元实验/路线A'
EM = np.load(os.path.join(OUT_DIR, 'endmembers.npz'), allow_pickle=True)
SIM = np.load(os.path.join(OUT_DIR, 'simulated.npz'), allow_pickle=True)
PR = np.load(os.path.join(OUT_DIR, 'predictions.npz'), allow_pickle=True)

CLASSES = list(SIM['classes']); NC = len(CLASSES)
ALGOS = list(PR['algorithm_names'])
M_LEVELS = SIM['m_levels']
te_lab = SIM['label']; m_arr = SIM['m']
thrT = EM['cube_thr'].T.astype(np.float32)


def discretize(x):
    x = np.asarray(x, np.float32)
    t = np.ones(x.shape, np.int8)
    for k in range(thrT.shape[0]):
        t += (x > thrT[k][None, :]).astype(np.int8)
    return t


def oa(p, y):
    return (p == y).mean() * 100


tr_tok = discretize(SIM['train_spec'].astype(np.float32))
tr_lab = SIM['train_label']
te_tok = discretize(SIM['spec'].astype(np.float32))


def hamming_variants(te_t, tr_t, tr_l, chunk=500):
    """返回: pred_system(并列弃权), pred_break(破平局), abstain_mask"""
    n = len(te_t)
    p_sys = np.full(n, -1, np.int8)
    p_brk = np.empty(n, np.int8)
    abstain = np.zeros(n, bool)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = (te_t[s:e, None, :] != tr_t[None, :, :]).sum(axis=2)
        md = d.min(axis=1)
        for i in range(e - s):
            tied = d[i] == md[i]
            u = np.unique(tr_l[tied])
            if u.size == 1:
                p_sys[s + i] = p_brk[s + i] = u[0]
            else:
                abstain[s + i] = True
                scores = [d[i][tied & (tr_l == c)].mean() for c in u]
                p_brk[s + i] = u[int(np.argmin(scores))]
    return p_sys, p_brk, abstain


print('计算模拟集 Hamming 变体 ...')
p_sys, p_brk, abstain = hamming_variants(te_tok, tr_tok, tr_lab)
print(f'  系统规则弃权: {int(abstain.sum())} ({abstain.mean()*100:.2f}%)')
print(f'  系统规则 OA = {oa(p_sys, te_lab):.2f}%   破平局 OA = {oa(p_brk, te_lab):.2f}%')
dec = ~abstain
print(f'  已判定像元精度 = {oa(p_sys[dec], te_lab[dec]):.2f}%  (n={int(dec.sum())})')

# 纯像元对照上的变体
pu_tok = discretize(SIM['pure_spec'].astype(np.float32))
pu_lab = SIM['pure_label']
ps_sys, ps_brk, ps_ab = hamming_variants(pu_tok, tr_tok, tr_lab)
print(f'  纯像元: 系统规则 OA = {oa(ps_sys, pu_lab):.2f}% (弃权 {int(ps_ab.sum())})  '
      f'破平局 OA = {oa(ps_brk, pu_lab):.2f}%')

# ============ 1. 分档 OA 表 ============
print('\n' + '=' * 108)
print('1. 各算法 × 混合档 OA (%)   [m = 优势类占比]')
print('=' * 108)
preds = {a: PR[f'pred_{a}'] for a in ALGOS}
preds['Hamming-系统规则'] = p_sys
preds['Hamming-破平局'] = p_brk
order = ['Hamming-系统规则', 'Hamming-破平局', 'SVM', 'RF', 'MLC', 'SAM', 'SVM-token']
hdr = f'{"算法":<18}' + ''.join(f'{m:>8.2f}' for m in M_LEVELS) + f'{"总体":>9}{"衰减":>8}'
print(hdr); print('-' * 108)
table = {}
for a in order:
    p = preds[a]
    row = [oa(p[np.isclose(m_arr, m)], te_lab[np.isclose(m_arr, m)]) for m in M_LEVELS]
    table[a] = row
    print(f'{a:<18}' + ''.join(f'{v:>8.2f}' for v in row)
          + f'{oa(p, te_lab):>9.2f}{row[0] - row[-1]:>8.2f}')

# 弃权率分档（系统规则）
ab_row = [abstain[np.isclose(m_arr, m)].mean() * 100 for m in M_LEVELS]
print(f'{"└ 弃权率%":<18}' + ''.join(f'{v:>8.1f}' for v in ab_row))

# ============ 2. McNemar 检验（破平局版 vs 各对比算法，全模拟集） ============
print('\n' + '=' * 108)
print('2. McNemar 检验  Hamming-破平局 vs 对比算法（模拟集 n=%d，双侧二项检验）' % len(te_lab))
print('=' * 108)
p_base = p_brk
mcn = {}
for a in ['SVM', 'RF', 'MLC', 'SAM', 'SVM-token']:
    p_b = preds[a]
    b01 = int(((p_base == te_lab) & (p_b != te_lab)).sum())   # 本方法对 对比错
    b10 = int(((p_base != te_lab) & (p_b == te_lab)).sum())   # 本方法错 对比对
    n_disc = b01 + b10
    if n_disc == 0:
        pv = 1.0
    else:
        pv = binomtest(min(b01, b10), n_disc, 0.5).pvalue * 1.0
    mcn[a] = (b01, b10, pv)
    win = '本方法显著更优' if (b01 > b10 and pv < 0.05) else (
        '对比算法显著更优' if (b10 > b01 and pv < 0.05) else '无显著差异')
    print(f'  vs {a:<11}: 本方法对/对比对 = {b01:>5}/{b10:<5}  p = {pv:.4f}  → {win}')

# ============ 3. 每类 PA × 混合档（本方法破平局 vs 最强对比） ============
print('\n' + '=' * 108)
print('3. 分档 × 类别 PA (%)：Hamming-破平局 vs RF')
print('=' * 108)
for a in ['Hamming-破平局', 'RF']:
    p = preds[a]
    print(f'  [{a}]')
    for ci, c in enumerate(CLASSES):
        row = []
        for m in M_LEVELS:
            sel = np.isclose(m_arr, m) & (te_lab == ci)
            row.append(oa(p[sel], te_lab[sel]) if sel.sum() else np.nan)
        print(f'    {c:<6}' + ''.join(f'{v:>8.1f}' for v in row))

# ============ 保存 ============
np.savez_compressed(os.path.join(OUT_DIR, 'final_results.npz'),
                    pred_system=p_sys, pred_break=p_brk, abstain=abstain,
                    tier_table=np.array([table[a] for a in order]),
                    algo_order=np.array(order), m_levels=M_LEVELS,
                    ab_rate=np.array(ab_row), mcnemar=np.array(
                        [(a, mcn[a][0], mcn[a][1], mcn[a][2]) for a in mcn],
                        dtype=[('algo', 'U16'), ('b01', 'i4'), ('b10', 'i4'), ('p', 'f8')]))
print('\n已保存: final_results.npz')
