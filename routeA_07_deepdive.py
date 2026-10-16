# -*- coding: utf-8 -*-
"""
路线A · 步骤7：病因深挖
  A. 全样本(170)训练版 —— 贴近系统实际用法（端元点也在模板库里）
  B. 纯像元分对/分错 —— 本方法的类别底子
  C. 类别对 × 混合档 分解 —— 本方法到底在哪些混合组合上失守
"""
import os
import numpy as np
from itertools import combinations

OUT_DIR = r'E:/混合像元实验/路线A'
EM = np.load(os.path.join(OUT_DIR, 'endmembers.npz'), allow_pickle=True)
SIM = np.load(os.path.join(OUT_DIR, 'simulated.npz'), allow_pickle=True)
PR = np.load(os.path.join(OUT_DIR, 'predictions.npz'), allow_pickle=True)

CLASSES = list(SIM['classes']); NC = len(CLASSES)
M_LEVELS = SIM['m_levels']
te_lab = SIM['label']; m_arr = SIM['m']; pair = SIM['pair']
all_spec = SIM['all_spec'].astype(np.float32); all_lab = SIM['all_label']
thrT = EM['cube_thr'].T.astype(np.float32)


def discretize(x):
    x = np.asarray(x, np.float32)
    t = np.ones(x.shape, np.int8)
    for k in range(thrT.shape[0]):
        t += (x > thrT[k][None, :]).astype(np.int8)
    return t


def oa(p, y):
    return (p == y).mean() * 100


def hamming(te_t, tr_t, tr_l, chunk=500):
    n = len(te_t)
    pred = np.full(n, -1, np.int8)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = (te_t[s:e, None, :] != tr_t[None, :, :]).sum(axis=2)
        md = d.min(axis=1)
        for i in range(e - s):
            u = np.unique(tr_l[d[i] == md[i]])
            if u.size == 1:
                pred[s + i] = u[0]
    return pred


all_tok = discretize(all_spec)
te_tok = discretize(SIM['spec'].astype(np.float32))
pu_tok = discretize(SIM['pure_spec'].astype(np.float32))
pu_lab = SIM['pure_label']

# ============ A. 全样本训练 ============
print('=' * 100)
print('A. 全样本(170)训练 —— 系统实际用法')
print('=' * 100)
p_all = hamming(te_tok, all_tok, all_lab)
print(f'  Hamming-NN(全样本)  模拟集 OA = {oa(p_all, te_lab):.2f}%   纯像元 OA = {oa(hamming(pu_tok, all_tok, all_lab), pu_lab):.2f}%')
row = [oa(p_all[np.isclose(m_arr, m)], te_lab[np.isclose(m_arr, m)]) for m in M_LEVELS]
print('  分档: ' + ''.join(f'{v:>8.2f}' for v in row))

# 各算法全样本版（SVM/RF）
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
sc = StandardScaler().fit(all_spec)
Ax, Bx = sc.transform(all_spec), sc.transform(SIM['spec'].astype(np.float32))
svm_all = SVC(kernel='rbf', C=10, gamma='scale').fit(Ax, all_lab).predict(Bx)
rf_all = RandomForestClassifier(500, random_state=42, n_jobs=-1).fit(Ax, all_lab).predict(Bx)
print(f'  SVM(全样本)        模拟集 OA = {oa(svm_all, te_lab):.2f}%')
print(f'  RF(全样本)         模拟集 OA = {oa(rf_all, te_lab):.2f}%')

# ============ B. 纯像元分类别 ============
print('\n' + '=' * 100)
print('B. 纯像元(m=1.0)分对/分错 —— 严格分离训练组(130)')
print('=' * 100)
p_pure = PR['pure_Hamming-NN']
for a in ['Hamming-NN', 'RF', 'SVM']:
    pp = PR[f'pure_{a}']
    pas = [oa(pp[pu_lab == i], pu_lab[pu_lab == i]) for i in range(NC)]
    print(f'  {a:<12} OA={oa(pp, pu_lab):>6.2f}%   ' + ''.join(f'{CLASSES[i]}:{pas[i]:>6.1f}  ' for i in range(NC)))

# ============ C. 类别对 × 档 分解 ============
print('\n' + '=' * 100)
print('C. 类别对 × m 分解 OA (%)  —— 行=类别对(优势/次类)，仅列 Hamming-破平局 与 RF')
print('=' * 100)
PAIR_NAMES = [f'{CLASSES[a]}+{CLASSES[b]}' for a, b in combinations(range(NC), 2)]

# 破平局版（与步骤6一致，就地重算）
tr_tok130 = discretize(SIM['train_spec'].astype(np.float32))
tr_lab130 = SIM['train_label']


def hamming_break(te_t, tr_t, tr_l, chunk=500):
    n = len(te_t)
    pred = np.empty(n, np.int8)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = (te_t[s:e, None, :] != tr_t[None, :, :]).sum(axis=2).astype(np.float32)
        md = d.min(axis=1)
        for i in range(e - s):
            tied = d[i] == md[i]
            u = np.unique(tr_l[tied])
            if u.size == 1:
                pred[s + i] = u[0]
            else:
                scores = [d[i][tied & (tr_l == c)].mean() for c in u]
                pred[s + i] = u[int(np.argmin(scores))]
    return pred


p_brk = hamming_break(te_tok, tr_tok130, tr_lab130)
print(f'  (破平局版总体 OA = {oa(p_brk, te_lab):.2f}%)')

rf_pred = PR['pred_RF']
print(f'{"类别对":<22}{"m":>6}{"Hamming":>10}{"RF":>10}{"SVM":>10}{"SVM-token":>11}{"n":>7}')
print('-' * 100)
worst = []
for pi, pn in enumerate(PAIR_NAMES):
    for m in M_LEVELS:
        sel = np.isclose(m_arr, m) & (pair == pi)
        if not sel.sum():
            continue
        h = oa(p_brk[sel], te_lab[sel]); r = oa(rf_pred[sel], te_lab[sel])
        s1 = oa(PR['pred_SVM'][sel], te_lab[sel]); s2 = oa(PR['pred_SVM-token'][sel], te_lab[sel])
        worst.append((h - max(r, s1, s2), pn, m, h, r, s1, s2, int(sel.sum())))
arr = np.array([w[0] for w in worst])
order_idx = np.argsort(arr)
print(' Hamming 相对最吃亏的 12 个组合:')
for i in order_idx[:12]:
    d, pn, m, h, r, s1, s2, n = worst[i]
    print(f'  {pn:<22}{m:>6.2f}{h:>10.1f}{r:>10.1f}{s1:>10.1f}{s2:>11.1f}{n:>7}   差 {d:+.1f}pp')
print(' Hamming 相对最占优的 8 个组合:')
for i in order_idx[-8:][::-1]:
    d, pn, m, h, r, s1, s2, n = worst[i]
    print(f'  {pn:<22}{m:>6.2f}{h:>10.1f}{r:>10.1f}{s1:>10.1f}{s2:>11.1f}{n:>7}   差 {d:+.1f}pp')
