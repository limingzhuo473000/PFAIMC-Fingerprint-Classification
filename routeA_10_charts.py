# -*- coding: utf-8 -*-
"""
路线A · 步骤10：论文级图表与数据表产出
  图1 纯度-精度衰减曲线（主实验 6 算法）
  图2 汉明距离响应机理曲线（优势类 vs 次类距离随混合比例变化）
  表1 分档 OA CSV      表2 McNemar CSV      表3 真实影像对照 CSV
  路线A实验报告.md
"""
import os
import sys
import numpy as np
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else r'E:/混合像元实验/路线A'
MR = np.load(os.path.join(OUT_DIR, 'main_results.npz'), allow_pickle=True)
RC = np.load(os.path.join(OUT_DIR, 'real_check.npz'), allow_pickle=True)
EM = np.load(os.path.join(OUT_DIR, 'endmembers.npz'), allow_pickle=True)

CLASSES = list(MR['classes']); NC = len(CLASSES)
M_LEVELS = MR['m_levels']
ALGOS = list(MR['algo_order'])
te_lab = MR['label']; m_arr = MR['m']
train_spec = MR['train_spec'].astype(np.float32); train_lab = MR['train_lab']
thrT = EM['cube_thr'].T.astype(np.float32)


def discretize(x):
    x = np.asarray(x, np.float32)
    t = np.ones(x.shape, np.int8)
    for k in range(thrT.shape[0]):
        t += (x > thrT[k][None, :]).astype(np.int8)
    return t


def oa(p, y):
    return (p == y).mean() * 100


STYLE = {
    'Hamming-NN': ('#d62728', 'o', '-', '本方法（指纹+汉明）'),
    'SVM': ('#1f77b4', 's', '--', 'SVM'),
    'SID': ('#2ca02c', '^', '--', 'SID（光谱信息散度）'),
    'MLC': ('#9467bd', 'v', '--', 'MLC'),
    'SAM': ('#ff7f0e', 'D', '--', 'SAM'),
    'SVM-token': ('#8c564b', 'P', ':', 'SVM(token)'),
}

# ---------- 图1 衰减曲线 ----------
fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=200)
for a in ALGOS:
    c, mk, ls, lab = STYLE[a]
    tier = MR['tier'][list(ALGOS).index(a)]
    ax.plot(M_LEVELS, tier, color=c, marker=mk, ls=ls, lw=2 if a == 'Hamming-NN' else 1.4,
            ms=6, label=lab, zorder=3 if a == 'Hamming-NN' else 2)
ax.set_xlabel('优势类占比 m（混合度：m 越小混合越严重）', fontsize=11)
ax.set_ylabel('总体精度 OA (%)', fontsize=11)
ax.set_title('模拟混合像元实验：各算法纯度-精度衰减曲线\n（LMM 端元模拟，20 期 × 10 波段时序，2700 个模拟像元）', fontsize=12)
ax.set_xticks(M_LEVELS)
ax.grid(alpha=0.3, ls=':')
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim(45, 103)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '图1_纯度精度衰减曲线.png'))
plt.close(fig)
print('图1 已保存')

# ---------- 图2 距离响应机理 ----------
tr_tok = discretize(train_spec)
te_tok = discretize(None) if False else None
# 重建测试集与主实验一致
from itertools import combinations
seed_spec, seed_lab = [], []
for ci, c in enumerate(CLASSES):
    em = EM[f'em_{c}']
    seed_spec.append(em[:5]); seed_lab.append(np.full(5, ci, np.int8))
seed_spec = np.vstack(seed_spec).astype(np.float32); seed_lab = np.concatenate(seed_lab)
specs, labels = [], []
for a, b in combinations(range(NC), 2):
    for dom, sub in [(a, b), (b, a)]:
        E_d, E_s = seed_spec[seed_lab == dom], seed_spec[seed_lab == sub]
        for m in M_LEVELS:
            mix = m * E_d[:, None, :] + (1 - m) * E_s[None, :, :]
            specs.append(mix.reshape(-1, mix.shape[-1]))
            labels.append(np.full(mix.shape[0] * mix.shape[1], dom, np.int8))
spec = np.vstack(specs).astype(np.float32)
label2 = np.concatenate(labels)
assert (label2 == te_lab).all()
te_tok = discretize(spec)

D = (te_tok[:, None, :] != tr_tok[None, :, :]).sum(axis=2).astype(np.float32)
d_dom = np.empty(len(spec)); d_sub = np.empty(len(spec))
for i in range(len(spec)):
    dom = te_lab[i]
    sub = seed_lab[pair_of(i)] if False else None
# 优势类模板距离 / 次类模板距离（利用合成顺序恢复次类）
subs = []
k = 0
for a, b in combinations(range(NC), 2):
    for dom, sub in [(a, b), (b, a)]:
        for m in M_LEVELS:
            subs.append(np.full(25, sub, np.int8))
sub_lab = np.concatenate(subs)

fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=200)
mm = M_LEVELS
# 平均距离：逐像元到两类全部模板的平均
d_dom2 = [np.mean([D[i][train_lab == te_lab[i]].mean()
                   for i in np.where(np.isclose(m_arr, m))[0]]) for m in mm]
d_sub2 = [np.mean([D[i][train_lab == sub_lab[i]].mean()
                   for i in np.where(np.isclose(m_arr, m))[0]]) for m in mm]
# 最小距离：逐像元到两类模板的最近距离（NN 实际决策量）
d_dom_min = [np.mean([D[i][train_lab == te_lab[i]].min()
                      for i in np.where(np.isclose(m_arr, m))[0]]) for m in mm]
d_sub_min = [np.mean([D[i][train_lab == sub_lab[i]].min()
                      for i in np.where(np.isclose(m_arr, m))[0]]) for m in mm]
ax.plot(mm, d_dom2, 'o-', color='#d62728', lw=2, label='到优势类模板 平均汉明距离')
ax.plot(mm, d_sub2, 's-', color='#1f77b4', lw=2, label='到次类模板 平均汉明距离')
ax.plot(mm, d_dom_min, 'o--', color='#d62728', lw=1.4, alpha=0.65, label='到优势类模板 最近汉明距离（NN 决策量）')
ax.plot(mm, d_sub_min, 's--', color='#1f77b4', lw=1.4, alpha=0.65, label='到次类模板 最近汉明距离（NN 决策量）')
ax.set_xlabel('优势类占比 m', fontsize=11)
ax.set_ylabel('汉明距离（200 维 token 不一致位数）', fontsize=11)
ax.set_title('机理：混合像元指纹与优势/次类模板的汉明距离响应\n（虚线为最近邻实际决策量，其交点即类别翻转临界点）', fontsize=12)
ax.set_xticks(mm)
ax.grid(alpha=0.3, ls=':')
ax.legend(fontsize=8.5, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '图2_距离响应机理曲线.png'))
plt.close(fig)
print('图2 已保存')

# ---------- 表1 分档 OA ----------
with open(os.path.join(OUT_DIR, '表1_主实验分档OA.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['算法'] + [f'm={m:.2f}' for m in M_LEVELS] + ['总体OA%', '衰减(pp)'])
    for a in ALGOS:
        tier = MR['tier'][list(ALGOS).index(a)]
        p = MR[f'pred_{a}']
        w.writerow([a] + [f'{v:.2f}' for v in tier] + [f'{oa(p, te_lab):.2f}', f'{tier[0]-tier[-1]:.2f}'])
print('表1 已保存')

# ---------- 表2 McNemar ----------
with open(os.path.join(OUT_DIR, '表2_McNemar检验.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['对比算法', '本方法对/对方错(b01)', '本方法错/对方对(b10)', 'p值(双侧二项)', '结论(α=0.05)'])
    for row in MR['mcnemar']:
        a, b01, b10, pv = row['algo'], int(row['b01']), int(row['b10']), float(row['p'])
        concl = '本方法显著更优' if (b01 > b10 and pv < 0.05) else ('对方显著更优' if (b10 > b01 and pv < 0.05) else '无显著差异')
        w.writerow([a, b01, b10, f'{pv:.4g}', concl])
print('表2 已保存')

# ---------- 表3 真实影像对照 ----------
te_lab_r = RC['te_lab']
with open(os.path.join(OUT_DIR, '表3_真实影像对照.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['算法', 'OA%', 'Kappa', '人工表面PA%', '水体PA%', '河漫滩PA%', '耕地PA%'])
    for a in RC['algos']:
        p = RC[f'pred_{a}']
        cm = np.zeros((NC, NC), int)
        for i in range(len(p)):
            if p[i] >= 0:
                cm[te_lab_r[i], p[i]] += 1
        po = np.trace(cm) / len(p)
        pe = (cm.sum(0) * cm.sum(1)).sum() / len(p) ** 2
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0
        pas = [cm[i, i] / max(1, cm[i].sum()) * 100 for i in range(NC)]
        w.writerow([a, f'{oa(p, te_lab_r):.2f}', f'{kappa:.4f}'] + [f'{v:.1f}' for v in pas])
print('表3 已保存')
print('\n全部产出完成')
