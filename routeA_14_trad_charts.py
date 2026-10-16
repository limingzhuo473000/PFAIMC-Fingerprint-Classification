# -*- coding: utf-8 -*-
"""
路线A · 步骤14：传统分类器对比论文级图表

图1  真实评价集 OA/Kappa 对比（本方法 vs 6 传统分类器 + SVM）
图2  时序信息增益对比（单时相 → 多时相，突出本方法利用率最高）
图3  模拟混合集分档衰减曲线（本方法 vs 传统分类器）
图4  模拟集总体 OA 排序条形图
"""
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else r'E:/混合像元实验/路线A_7月版'
CHART_DIR = r'E:/混合像元实验/传统分类器对比'
os.makedirs(CHART_DIR, exist_ok=True)

T = np.load(os.path.join(OUT_DIR, 'traditional_results.npz'), allow_pickle=True)
R = np.load(os.path.join(OUT_DIR, 'real_traditional.npz'), allow_pickle=True)

CLASSES = ['人工表面', '水体', '河漫滩', '耕地']
M_LEVELS = T['m_levels']
TRAD = [str(x) for x in T['trad_order']]
order_mt = [str(x) for x in T['order_mt']]
label = T['label']; m_arr = T['m']


def oa(p, y):
    return (p == y).mean() * 100


# 配色：本方法红金突出，传统蓝灰，现代绿
C_OUR = '#c0392b'
C_TRAD = '#4a6fa5'
C_MODERN = '#2e8b57'

# ============ 图1 真实评价集对比 ============
te_lab = R['te_lab']
order_real = ['MLC', 'SVM', 'SID', 'Hamming-NN', 'MahaD', 'SAM', 'MDC', 'PP']
vals, kaps, cols = [], [], []
for a in order_real:
    p = R[f'mt_{a}']
    vals.append(oa(p, te_lab))
    cm = np.zeros((4, 4), int)
    for i in range(len(p)):
        if p[i] >= 0:
            cm[te_lab[i], p[i]] += 1
    po = np.trace(cm) / len(p); pe = (cm.sum(0) * cm.sum(1)).sum() / len(p) ** 2
    kaps.append((po - pe) / (1 - pe))
    cols.append(C_OUR if a == 'Hamming-NN' else (C_MODERN if a == 'SVM' else C_TRAD))

fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=200)
bars = ax.bar(range(len(order_real)), vals, color=cols, width=0.62,
              edgecolor='white', linewidth=1.2)
for i, (b, v, k) in enumerate(zip(bars, vals, kaps)):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f'{v:.2f}%', ha='center',
            fontsize=10.5, fontweight='bold' if i == 3 else 'normal',
            color=C_OUR if i == 3 else '#333')
    ax.text(b.get_x() + b.get_width() / 2, 6, f'κ={k:.3f}', ha='center', fontsize=8.2,
            color='white', rotation=90)
ax.axhline(vals[3], color=C_OUR, ls='--', lw=1.2, alpha=0.6)
ax.set_xticks(range(len(order_real)))
ax.set_xticklabels([a + '\n(本方法)' if a == 'Hamming-NN' else a for a in order_real], fontsize=10)
ax.set_ylabel('总体精度 OA (%)', fontsize=11)
ax.set_ylim(0, 106)
ax.set_title('真实影像评价集：本方法 vs 传统分类器（163 模板点 → 109 独立评价点）',
             fontsize=12.5, pad=12)
ax.grid(axis='y', alpha=0.25, ls=':')
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, '图1_真实评价集对比.png'))
plt.close(fig)
print('图1 已保存')

# ============ 图2 时序增益 ============
order_g = ['MahaD', 'SVM', 'MLC', 'MDC', 'PP', 'SID', 'SAM', 'Hamming-NN']
sm, sb = R['single_mean'], R['single_best']
order_s = [str(x) for x in R['order_single']]
mt_map = {a: oa(R[f'mt_{a}'], te_lab) for a in order_g}
single_mean_map = {a: float(sm[order_s.index(a)]) for a in order_g if a in order_s}
single_mean_map['Hamming-NN'] = float(np.mean(R['single_hamming']))

fig, ax = plt.subplots(figsize=(10.4, 5.0), dpi=200)
x = np.arange(len(order_g)); w = 0.36
v1 = [single_mean_map[a] for a in order_g]
v2 = [mt_map[a] for a in order_g]
c1 = ['#a8c4e0' if a != 'Hamming-NN' else '#e8a0a0' for a in order_g]
c2 = [C_TRAD if a not in ('Hamming-NN', 'SVM') else (C_OUR if a == 'Hamming-NN' else C_MODERN)
      for a in order_g]
b1 = ax.bar(x - w / 2, v1, w, color=c1, edgecolor='white', label='单时相（10 维，20 期均值）')
b2 = ax.bar(x + w / 2, v2, w, color=c2, edgecolor='white', label='多时相（200 维，20 期时序）')
for i, (a, y1, y2) in enumerate(zip(order_g, v1, v2)):
    ax.annotate('', xy=(i + w / 2, y2), xytext=(i - w / 2, y1),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.0, alpha=0.7))
    ax.text(i, max(y1, y2) + 2.2, f'+{y2 - y1:.1f}pp\n(+{(y2 - y1) / y1 * 100:.0f}%)',
            ha='center', fontsize=8.6,
            fontweight='bold' if a == 'Hamming-NN' else 'normal',
            color=C_OUR if a == 'Hamming-NN' else '#555')
ax.set_xticks(x)
ax.set_xticklabels([a + '\n★本方法' if a == 'Hamming-NN' else a for a in order_g], fontsize=10)
ax.set_ylabel('真实评价集 OA (%)', fontsize=11)
ax.set_ylim(0, 112)
ax.set_title('时序信息增益：单时相 → 多时相（本方法对时序的利用率是传统分类器的 2~7 倍）',
             fontsize=12.5, pad=12)
ax.legend(fontsize=9.5, loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.25, ls=':')
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, '图2_时序增益对比.png'))
plt.close(fig)
print('图2 已保存')

# ============ 图3 模拟集分档曲线 ============
fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=200)
style = {
    'Hamming-NN': dict(color=C_OUR, lw=3.0, marker='o', ms=8, zorder=10, label='本方法(指纹+汉明)'),
    'MLC': dict(color='#7f8c8d', lw=1.6, marker='^', ms=5, label='MLC 最大似然'),
    'MDC': dict(color='#8e44ad', lw=1.6, marker='v', ms=5, label='MDC 最小距离'),
    'MahaD': dict(color='#2980b9', lw=1.8, marker='D', ms=5, label='MahaD 马氏距离'),
    'PP': dict(color='#16a085', lw=1.6, marker='P', ms=5, label='PP 平行六面体'),
    'SAM': dict(color='#e67e22', lw=1.8, marker='s', ms=5, label='SAM 光谱角'),
    'SID': dict(color='#95a5a6', lw=1.6, marker='x', ms=6, label='SID 信息散度'),
    'SVM': dict(color=C_MODERN, lw=1.6, marker='*', ms=8, ls='--', label='SVM（现代基准）'),
}
for a in ['SVM', 'SID', 'SAM', 'PP', 'MahaD', 'MDC', 'MLC', 'Hamming-NN']:
    p = T[f'mt_{a}']
    row = [oa(p[np.isclose(m_arr, m)], label[np.isclose(m_arr, m)]) for m in M_LEVELS]
    ax.plot(M_LEVELS, row, **style[a])
ax.set_xlabel('优势类占比 m（越大越接近纯像元）', fontsize=11)
ax.set_ylabel('总体精度 OA (%)', fontsize=11)
ax.set_title('模拟混合像元集：分档精度衰减（2700 像元，6 类别对 × 9 档）', fontsize=12.5, pad=12)
ax.set_xticks(M_LEVELS)
ax.grid(alpha=0.3, ls=':')
ax.legend(fontsize=9, ncol=2, loc='lower right', framealpha=0.92)
ax.axvspan(0.55, 0.70, alpha=0.07, color='red')
ax.text(0.625, 32, '重度混合区\n（本方法短板）', ha='center', fontsize=8.6, color='#a33')
ax.axvspan(0.85, 0.95, alpha=0.07, color='green')
ax.text(0.90, 32, '高纯度区\n（本方法追平/反超）', ha='center', fontsize=8.6, color='#276')
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, '图3_模拟集分档衰减.png'))
plt.close(fig)
print('图3 已保存')

# ============ 图4 模拟集总体排序 ============
fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=200)
tot = {a: oa(T[f'mt_{a}'], label) for a in order_mt}
od = sorted(tot, key=lambda a: -tot[a])
cols4 = [C_OUR if a == 'Hamming-NN' else (C_MODERN if a == 'SVM' else C_TRAD) for a in od]
bs = ax.barh(range(len(od))[::-1], [tot[a] for a in od], color=cols4, height=0.62,
             edgecolor='white')
for i, a in enumerate(od):
    ax.text(tot[a] + 0.8, len(od) - 1 - i, f'{tot[a]:.2f}%', va='center', fontsize=10,
            fontweight='bold' if a == 'Hamming-NN' else 'normal',
            color=C_OUR if a == 'Hamming-NN' else '#333')
ax.set_yticks(range(len(od))[::-1])
ax.set_yticklabels([a + ' ★' if a == 'Hamming-NN' else a for a in od], fontsize=10)
ax.set_xlabel('模拟混合像元集总体精度 OA (%)', fontsize=11)
ax.set_xlim(0, 103)
ax.set_title('模拟混合像元集总体精度排序（2700 像元，m=0.55~0.95）', fontsize=12.5, pad=12)
ax.grid(axis='x', alpha=0.25, ls=':')
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, '图4_模拟集总体排序.png'))
plt.close(fig)
print('图4 已保存')
print('\n全部图表输出至:', CHART_DIR)
