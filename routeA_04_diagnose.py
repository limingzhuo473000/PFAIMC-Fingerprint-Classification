# -*- coding: utf-8 -*-
"""
路线A · 步骤4：分档精度统计 + 病因诊断

诊断三个疑点
  A. 分档 OA 曲线：本方法在哪一档掉队？是持续劣势还是低混合度特有？
  B. 真实纯像元 5 折交叉验证：本方法在"真实数据"上是否也垫底？
     —— 若真实数据上不垫底、只在模拟集垫底，说明问题出在"线性混合 + 硬离散化"的组合。
  C. 全样本训练(170)版对照：排除"端元挖走最纯样本"的影响。
"""
import os
import numpy as np

OUT_DIR = r'E:/混合像元实验/路线A'
EM = np.load(os.path.join(OUT_DIR, 'endmembers.npz'), allow_pickle=True)
SIM = np.load(os.path.join(OUT_DIR, 'simulated.npz'), allow_pickle=True)
PR = np.load(os.path.join(OUT_DIR, 'predictions.npz'), allow_pickle=True)

CLASSES = list(SIM['classes']); NC = len(CLASSES)
ALGOS = list(PR['algorithm_names'])
M_LEVELS = SIM['m_levels']
te_lab = PR['te_lab']; m_arr = SIM['m']; pair = SIM['pair']
thrT = EM['cube_thr'].T.astype(np.float32)


def discretize(x):
    x = np.asarray(x, np.float32)
    t = np.ones(x.shape, np.int8)
    for k in range(thrT.shape[0]):
        t += (x > thrT[k][None, :]).astype(np.int8)
    return t


def oa(pred, y):
    return (pred == y).mean() * 100


# ============ A. 分档 OA ============
print('=' * 100)
print('A. 各算法 × 混合档 总体精度 OA (%)   [m = 优势类占比，越低越难]')
print('=' * 100)
hdr = '算法'.ljust(13) + ''.join(f'{m:>9.2f}' for m in M_LEVELS) + f'{"总体":>10}{"衰减":>9}'
print(hdr)
print('-' * 100)
curve = {}
for a in ALGOS:
    p = PR[f'pred_{a}']
    row = [oa(p[np.isclose(m_arr, m)], te_lab[np.isclose(m_arr, m)]) for m in M_LEVELS]
    curve[a] = row
    print(a.ljust(13) + ''.join(f'{v:>9.2f}' for v in row) + f'{oa(p, te_lab):>10.2f}{row[0]-row[-1]:>9.2f}')

# ============ B. 真实纯像元 5 折 CV ============
print('\n' + '=' * 100)
print('B. 真实样本点上的 5 折交叉验证 OA (%)  —— 检验"本方法在真实数据上是否也垫底"')
print('=' * 100)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

all_spec = SIM['all_spec'].astype(np.float32); all_lab = SIM['all_label']
all_tok = discretize(all_spec)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_res = {a: [] for a in ALGOS}

for tr_i, te_i in skf.split(all_spec, all_lab):
    A, B = all_spec[tr_i], all_spec[te_i]
    AT, BT = all_tok[tr_i], all_tok[te_i]
    la, lb = all_lab[tr_i], all_lab[te_i]
    # 本方法
    d = (BT[:, None, :] != AT[None, :, :]).sum(axis=2)
    pd_ = np.full(len(B), -1, np.int8)
    for i in range(len(B)):
        u = np.unique(la[d[i] == d[i].min()])
        if u.size == 1:
            pd_[i] = u[0]
    cv_res['Hamming-NN'].append(oa(pd_, lb))
    # 其他
    sc = StandardScaler().fit(A)
    Ax, Bx = sc.transform(A), sc.transform(B)
    cv_res['SVM'].append(oa(SVC(kernel='rbf', C=10, gamma='scale').fit(Ax, la).predict(Bx), lb))
    cv_res['RF'].append(oa(RandomForestClassifier(300, random_state=42, n_jobs=-1).fit(Ax, la).predict(Bx), lb))
    pc = PCA(n_components=min(20, len(tr_i) - 1), random_state=42).fit(Ax)
    cv_res['MLC'].append(oa(QuadraticDiscriminantAnalysis(reg_param=0.3).fit(pc.transform(Ax), la)
                            .predict(pc.transform(Bx)), lb))
    mus = np.stack([A[la == c].mean(0) for c in range(NC)])
    tn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    mn = mus / (np.linalg.norm(mus, axis=1, keepdims=True) + 1e-9)
    cv_res['SAM'].append(oa((tn @ mn.T).argmax(1), lb))
    cv_res['SVM-token'].append(oa(SVC(kernel='rbf', C=10, gamma='scale')
                                  .fit(AT.astype(np.float32), la).predict(BT.astype(np.float32)), lb))

for a in ALGOS:
    v = np.array(cv_res[a])
    print(f'{a:<14} 5折 OA = {v.mean():>6.2f}%   (各折: ' + ' '.join(f'{x:.1f}' for x in v) + ')')

# ============ C. 全样本训练版 ============
print('\n' + '=' * 100)
print('C. 用全部 170 个样本点训练（含端元，贴近系统实际用法）在模拟集上的 OA (%)')
print('=' * 100)
tr_tok = discretize(all_spec)
d = (PR['te_lab'].shape and None)  # noop
te_tok = discretize(SIM['spec'].astype(np.float32))
pd_ = np.full(len(te_tok), -1, np.int8)
CH = 400
for s in range(0, len(te_tok), CH):
    e = min(s + CH, len(te_tok))
    dd = (te_tok[s:e, None, :] != tr_tok[None, :, :]).sum(axis=2)
    md = dd.min(axis=1)
    for i in range(e - s):
        u = np.unique(all_lab[dd[i] == md[i]])
        if u.size == 1:
            pd_[s + i] = u[0]
print(f'{"Hamming-NN(全样本)":<22} {oa(pd_, te_lab):>6.2f}%   '
      f'(未分类 {int((pd_==-1).sum())})')

sc = StandardScaler().fit(all_spec)
Ax, Bx = sc.transform(all_spec), sc.transform(SIM['spec'].astype(np.float32))
print(f'{"SVM(全样本)":<22} {oa(SVC(kernel="rbf", C=10, gamma="scale").fit(Ax, all_lab).predict(Bx), te_lab):>6.2f}%')
print(f'{"RF(全样本)":<22} {oa(RandomForestClassifier(500, random_state=42, n_jobs=-1).fit(Ax, all_lab).predict(Bx), te_lab):>6.2f}%')

# ============ 混淆矩阵（本方法 vs 最好的对比算法） ============
print('\n' + '=' * 100)
print('D. 混淆矩阵（整体模拟集，行=真值，列=预测，最后一行/列=未分类）')
print('=' * 100)
for a in ['Hamming-NN', 'SAM']:
    p = PR[f'pred_{a}']
    cm = np.zeros((NC, NC + 1), int)
    for i in range(len(p)):
        cm[te_lab[i], p[i] if p[i] >= 0 else NC] += 1
    print(f'\n[{a}]')
    print('        ' + ''.join(f'{c:>10}' for c in CLASSES) + f'{"未分类":>10}{"PA%":>8}')
    for i, c in enumerate(CLASSES):
        pa = cm[i, i] / cm[i].sum() * 100
        print(f'{c:<8}' + ''.join(f'{v:>10}' for v in cm[i]) + f'{pa:>8.1f}')
    ua = [cm[i, i] / max(1, cm[:, i].sum()) * 100 for i in range(NC)]
    print('UA%     ' + ''.join(f'{v:>10.1f}' for v in ua))

np.savez_compressed(os.path.join(OUT_DIR, 'diagnosis.npz'),
                    curve=np.array([curve[a] for a in ALGOS]),
                    cv=np.array([cv_res[a] for a in ALGOS]),
                    algos=np.array(ALGOS), m_levels=M_LEVELS)
print('\n已保存诊断结果: diagnosis.npz')
