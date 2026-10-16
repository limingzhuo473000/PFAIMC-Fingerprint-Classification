# -*- coding: utf-8 -*-
"""
路线A · 步骤11：离散化级数对比（5 级 vs 2 级）

对照两组实验（同一批样本、同一批模拟像元、同一批对比算法），
只改变分位数离散化的级数，量化级数对"指纹匹配"方法的影响。

  5 级：E:/混合像元实验/路线A_7月版      （4 个分位阈值/波段，log2(5)=2.32 bit）
  2 级：E:/混合像元实验/路线A_7月版_L2   （1 个分位阈值/波段，log2(2)=1.00 bit）

输出到 E:/混合像元实验/离散化级数对比/
  图A_分档OA对比.png   图B_关键指标对比.png
  表1_分档OA对比.csv   表2_关键指标对比.csv
"""
import os
import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

DIR_5 = r'E:/混合像元实验/路线A_7月版'
DIR_2 = r'E:/混合像元实验/路线A_7月版_L2'
OUT = r'E:/混合像元实验/离散化级数对比'
os.makedirs(OUT, exist_ok=True)


def load_cfg(d):
    MR = np.load(os.path.join(d, 'main_results.npz'), allow_pickle=True)
    RC = np.load(os.path.join(d, 'real_check.npz'), allow_pickle=True)
    EM = np.load(os.path.join(d, 'endmembers.npz'), allow_pickle=True)
    n_cls = int(EM['n_classes'][0]) if 'n_classes' in EM.files else 5
    return MR, RC, EM, n_cls


def discretize(x, thrT):
    x = np.asarray(x, np.float32)
    t = np.ones(x.shape, np.int8)
    for k in range(thrT.shape[0]):
        t += (x > thrT[k][None, :]).astype(np.int8)
    return t


def oa(p, y):
    return (p == y).mean() * 100


def kappa_of(p, y, nc):
    cm = np.zeros((nc, nc), int)
    for i in range(len(p)):
        if p[i] >= 0:
            cm[y[i], p[i]] += 1
    po = np.trace(cm) / len(p)
    pe = (cm.sum(0) * cm.sum(1)).sum() / len(p) ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def mean_entropy(tok):
    """每波段 token 分布的经验熵（bit），再对波段取平均 = 单个 token 的实际信息量"""
    ents = []
    for j in range(tok.shape[1]):
        _, cnt = np.unique(tok[:, j], return_counts=True)
        p = cnt / cnt.sum()
        ents.append(float(-(p * np.log2(p)).sum()))
    return float(np.mean(ents))


CFG = {}
for tag, d in [('5', DIR_5), ('2', DIR_2)]:
    MR, RC, EM, n_cls = load_cfg(d)
    thrT = EM['cube_thr'].T.astype(np.float32)
    ALGOS = list(MR['algo_order'])
    tr_tok = discretize(MR['train_spec'], thrT)
    i_h = ALGOS.index('Hamming-NN')
    CFG[tag] = dict(
        MR=MR, RC=RC, EM=EM, n_cls=n_cls, thrT=thrT, ALGOS=ALGOS, tr_tok=tr_tok,
        tier=MR['tier'][i_h],
        oa_all=oa(MR['pred_Hamming-NN'], MR['label']),
        pure=oa(MR['pred_pure_hamming'], MR['seed_lab']),
        amb=int(MR['abstain_hamming']),
        real_oa=oa(RC['pred_Hamming-NN'], RC['te_lab']),
        real_kappa=kappa_of(RC['pred_Hamming-NN'], RC['te_lab'], len(MR['classes'])),
        ent=mean_entropy(tr_tok),
        tier_by_algo={a: MR['tier'][ALGOS.index(a)] for a in ALGOS},
        real_by_algo={a: (oa(RC[f'pred_{a}'], RC['te_lab']),
                          kappa_of(RC[f'pred_{a}'], RC['te_lab'], len(MR['classes'])))
                      for a in RC['algos']},
    )

M = CFG['5']['MR']['m_levels']
CLASSES = list(CFG['5']['MR']['classes'])
print('=' * 74)
print('离散化级数对比（Hamming-NN）')
print('=' * 74)
hdr = f'{"级数":<6}{"阈值/波段":>10}{"信息量bit":>10}{"模拟总体":>10}{"纯像元":>9}{"真实OA":>9}{"弃权率":>9}'
print(hdr)
print('-' * 74)
for tag in ['5', '2']:
    c = CFG[tag]
    print(f'{tag+" 级":<6}{c["thrT"].shape[0]:>10}{c["ent"]:>10.3f}{c["oa_all"]:>10.2f}'
          f'{c["pure"]:>9.1f}{c["real_oa"]:>9.2f}{c["amb"] / 2700 * 100:>9.2f}')

# ---------- 图A 分档曲线对比 ----------
fig, ax = plt.subplots(figsize=(9, 5.6), dpi=200)
STY = {
    'Hamming-NN': ('#d62728', 'o', '-', 2.6),
    'SAM': ('#ff7f0e', 'D', '--', 1.4),
    'SVM': ('#1f77b4', 's', '--', 1.4),
}
for tag, ls_mul, alpha, suf in [('5', 1.0, 1.0, ''), ('2', 1.0, 0.85, '(2 级)')]:
    c = CFG[tag]
    t = c['tier_by_algo']['Hamming-NN']
    ax.plot(M, t, color='#d62728', marker='o', ls='-' if tag == '5' else '--',
            lw=2.6 if tag == '5' else 2.0, ms=7, alpha=alpha,
            label=f'本方法 {tag} 级' if tag == '5' else '本方法 2 级')
for a in ['SAM', 'SVM']:
    c, mk, ls, lw = STY[a]
    ax.plot(M, CFG['5']['tier_by_algo'][a], color=c, marker=mk, ls=ls, lw=lw, ms=6, label=a)
ax.set_xlabel('优势类占比 m（m 越小混合越严重）', fontsize=11)
ax.set_ylabel('总体精度 OA (%)', fontsize=11)
ax.set_title('离散化级数对精度-混合度响应的影响\n（同一批 2700 个模拟像元、同一批样本与模板）', fontsize=12)
ax.set_xticks(M)
ax.grid(alpha=0.3, ls=':')
ax.legend(fontsize=9.5, loc='lower right')
ax.set_ylim(45, 103)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '图A_分档OA对比.png'))
plt.close(fig)
print('\n图A 已保存')

# ---------- 图B 关键指标柱状对比 ----------
fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.3), dpi=200)
items = [
    ('模拟集总体 OA (%)', [CFG['5']['oa_all'], CFG['2']['oa_all']], 70),
    ('纯像元测试 OA (%)', [CFG['5']['pure'], CFG['2']['pure']], 80),
    ('真实影像 OA (%)', [CFG['5']['real_oa'], CFG['2']['real_oa']], 88),
    ('模拟集弃权率 (%)', [CFG['5']['amb'] / 27, CFG['2']['amb'] / 27], 0),
]
for ax, (title, vals, bottom) in zip(axes, items):
    bars = ax.bar([0, 1], vals, width=0.5, color=['#4C72B0', '#DD8452'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['5 级', '2 级'], fontsize=11)
    ax.set_title(title, fontsize=11)
    for x, v in zip([0, 1], vals):
        ax.text(x, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10.5)
    ax.grid(alpha=0.3, ls=':', axis='y')
    lo, hi = min(vals), max(vals)
    ax.set_ylim(bottom, hi + (hi - lo) * 0.35 + 0.8)
fig.suptitle('离散化级数：5 级 vs 2 级 关键指标对比', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT, '图B_关键指标对比.png'))
plt.close(fig)
print('图B 已保存')

# ---------- 表1 分档 OA 对比 ----------
with open(os.path.join(OUT, '表1_分档OA对比.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['算法'] + [f'm={m:.2f}' for m in M] + ['总体OA%'])
    for tag in ['5', '2']:
        c = CFG[tag]
        w.writerow([f'本方法 {tag} 级'] + [f'{v:.2f}' for v in c['tier_by_algo']['Hamming-NN']]
                   + [f'{c["oa_all"]:.2f}'])
    for a in ['SVM', 'SID', 'MLC', 'SAM', 'SVM-token']:
        w.writerow([f'{a}（与级数无关）'] + [f'{v:.2f}' for v in CFG['5']['tier_by_algo'][a]]
                   + [f'{oa(CFG["5"]["MR"][f"pred_{a}"], CFG["5"]["MR"]["label"]):.2f}'])
print('表1 已保存')

# ---------- 表2 关键指标对比 ----------
with open(os.path.join(OUT, '表2_关键指标对比.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['指标', '5 级', '2 级', '差值(2级−5级)'])
    rows = [
        ('离散级数', 5, 2, -3),
        ('分位阈值数/波段', CFG['5']['thrT'].shape[0], CFG['2']['thrT'].shape[0],
         CFG['2']['thrT'].shape[0] - CFG['5']['thrT'].shape[0]),
        ('单token经验熵(bit)', CFG['5']['ent'], CFG['2']['ent'], CFG['2']['ent'] - CFG['5']['ent']),
        ('模拟集总体OA%', CFG['5']['oa_all'], CFG['2']['oa_all'], CFG['2']['oa_all'] - CFG['5']['oa_all']),
        ('纯像元测试OA%', CFG['5']['pure'], CFG['2']['pure'], CFG['2']['pure'] - CFG['5']['pure']),
        ('模拟集弃权率%', CFG['5']['amb'] / 27, CFG['2']['amb'] / 27,
         (CFG['2']['amb'] - CFG['5']['amb']) / 27),
        ('真实影像OA%', CFG['5']['real_oa'], CFG['2']['real_oa'],
         CFG['2']['real_oa'] - CFG['5']['real_oa']),
        ('真实影像Kappa', CFG['5']['real_kappa'], CFG['2']['real_kappa'],
         CFG['2']['real_kappa'] - CFG['5']['real_kappa']),
    ]
    for name, a, b, d in rows:
        w.writerow([name, f'{a:.3f}' if isinstance(a, float) else a,
                    f'{b:.3f}' if isinstance(b, float) else b,
                    f'{d:+.3f}' if isinstance(d, float) else f'{d:+d}'])
    w.writerow([])
    w.writerow(['—— 各对比算法（输入为原始光谱，理论上与级数无关）——'])
    w.writerow(['算法', '5级真实OA%', '2级真实OA%', '差值'])
    for a in ['SVM', 'SID', 'MLC', 'SAM']:
        v5 = CFG['5']['real_by_algo'][a][0]
        v2 = CFG['2']['real_by_algo'][a][0]
        w.writerow([a, f'{v5:.2f}', f'{v2:.2f}', f'{v2 - v5:+.2f}'])
print('表2 已保存')

print('\n' + '=' * 74)
print('分档 OA 逐档差异（2 级 − 5 级）')
print('=' * 74)
d5 = CFG['5']['tier_by_algo']['Hamming-NN']
d2 = CFG['2']['tier_by_algo']['Hamming-NN']
for m, a, b in zip(M, d5, d2):
    print(f'  m={m:.2f}:  5级 {a:>6.2f}   2级 {b:>6.2f}   差 {b - a:>+6.2f}')
print(f'\n全部产出: {OUT}')
