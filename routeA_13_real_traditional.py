# -*- coding: utf-8 -*-
"""
路线A · 步骤13：真实影像评价集上的「传统分类器」全面对比

用户重点：与传统分类器对比才是核心。

训练 = 模板样本点（163，7月版）  测试 = 独立评价样本点（109，系统 89.91% 同口径）
算法：
  本方法   指纹离散化 token + 汉明距离最近邻（系统判决规则，含阈值/并列弃权）
  传统分类器 MLC / MDC / MahaD / PP(平行六面体) / SAM / SID
  现代基准  SVM
两种输入：
  A. 多时相 200 维（20 期 × 10 波段）
  B. 单时相 10 维（逐期跑 20 次，报均值/最优）—— 量化时序信息增益

输出：真实集对比表 + 单时相增益 + Kappa/PA + npz
"""
import os
import sys
import numpy as np
import rasterio
import pyogrio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

IMG_DIR = r'E:/指纹/124036中/图像'
TRAIN_SH = sys.argv[2] if len(sys.argv) > 2 else r'E:/指纹/124036中/样本/实验样本_7月版'
TEST_SH = r'E:/指纹/124036中/样本/评价样本'
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else r'E:/混合像元实验/路线A_7月版'
EM = np.load(os.path.join(OUT_DIR, 'endmembers.npz'), allow_pickle=True)

CLASSES = ['人工表面', '水体', '河漫滩', '耕地']
NC = len(CLASSES)
thrT = EM['cube_thr'].T.astype(np.float32)
tifs = [str(b) for b in EM['band_names']]
N_BAND = len(tifs)
N_PER_DATE = N_BAND // 20
N_PCA = 20


def discretize(x, cols=None):
    x = np.asarray(x, np.float32)
    T = thrT if cols is None else thrT[:, cols]
    t = np.ones(x.shape, np.int8)
    for k in range(T.shape[0]):
        t += (x > T[k][None, :]).astype(np.int8)
    return t


# ---------- 1. 读入立方体 ----------
print('读入全影像立方体 ...')
with rasterio.open(os.path.join(IMG_DIR, tifs[0])) as s:
    H, W, transform = s.height, s.width, s.transform
cube = np.empty((N_BAND, H, W), np.float32)
for i, f in enumerate(tifs):
    with rasterio.open(os.path.join(IMG_DIR, f)) as s:
        cube[i] = s.read(1)
print(f'  立方体 {cube.shape}')


def read_spec(shp_dir, label):
    xs_all, ys_all, lb_all = [], [], []
    for ci, c in enumerate(CLASSES):
        p = os.path.join(shp_dir, c + '.shp')
        if not os.path.exists(p):
            print(f'  !! 缺 {p}'); continue
        gdf = pyogrio.read_dataframe(p)
        xs = np.array([g.x for g in gdf.geometry]); ys = np.array([g.y for g in gdf.geometry])
        cols = np.floor((xs - transform.c) / transform.a).astype(int)
        rows = np.floor((transform.f - ys) / (-transform.e)).astype(int)
        ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        xs_all.append(xs[ok]); ys_all.append(ys[ok]); lb_all.append(np.full(ok.sum(), ci, np.int8))
    xs_all = np.concatenate(xs_all); ys_all = np.concatenate(ys_all); lb_all = np.concatenate(lb_all)
    cols = np.floor((xs_all - transform.c) / transform.a).astype(int)
    rows = np.floor((transform.f - ys_all) / (-transform.e)).astype(int)
    sp = cube[:, rows, cols].T.astype(np.float32)
    print(f'  {label}: {sp.shape}  ' + '  '.join(f'{CLASSES[i]}={int((lb_all==i).sum())}' for i in range(NC)))
    return sp, lb_all


print('\n读取样本光谱 ...')
tr_spec, tr_lab = read_spec(TRAIN_SH, '训练(模板样本)')
te_spec, te_lab = read_spec(TEST_SH, '测试(评价样本)')


def oa(p, y):
    return (p == y).mean() * 100


def kappa_of(p, y, nc=NC):
    cm = np.zeros((nc, nc), int)
    for i in range(len(p)):
        if p[i] >= 0:
            cm[y[i], p[i]] += 1
    po = np.trace(cm) / len(p)
    pe = (cm.sum(0) * cm.sum(1)).sum() / len(p) ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def hamming_nn(te_t, tr_t, tr_l):
    d = (te_t[:, None, :] != tr_t[None, :, :]).sum(axis=2)
    pred = np.full(len(te_t), -1, np.int8); amb = 0
    for i in range(len(te_t)):
        u = np.unique(tr_l[d[i] == d[i].min()])
        if u.size == 1:
            pred[i] = u[0]
        else:
            amb += 1
    return pred, amb


# ---------- 2. 传统分类器（与 routeA_12 同一套实现） ----------
def _pca(Xtr, Xte, n=N_PCA):
    p = PCA(n_components=min(n, Xtr.shape[0] - 1, Xtr.shape[1]), random_state=42).fit(Xtr)
    return p.transform(Xtr), p.transform(Xte)


def mlc(Xtr, Xte, ytr):
    sc = StandardScaler().fit(Xtr)
    A, B = _pca(sc.transform(Xtr), sc.transform(Xte))
    return QuadraticDiscriminantAnalysis(reg_param=0.3).fit(A, ytr).predict(B).astype(np.int8)


def mdc(Xtr, Xte, ytr):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    mus = np.stack([A[ytr == c].mean(0) for c in range(NC)])
    return ((B[:, None, :] - mus[None, :, :]) ** 2).sum(-1).argmin(1).astype(np.int8)


def mahad(Xtr, Xte, ytr):
    sc = StandardScaler().fit(Xtr)
    A, B = _pca(sc.transform(Xtr), sc.transform(Xte))
    mus = np.stack([A[ytr == c].mean(0) for c in range(NC)])
    Ci = np.linalg.pinv(np.cov(A.T) + np.eye(A.shape[1]) * 1e-3)
    d = np.empty((len(B), NC))
    for c in range(NC):
        e = B - mus[c]
        d[:, c] = np.einsum('ij,jk,ik->i', e, Ci, e)
    return d.argmin(1).astype(np.int8)


def parallelepiped(Xtr, Xte, ytr, k=1.5):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    lo = np.stack([A[ytr == c].mean(0) - k * A[ytr == c].std(0) for c in range(NC)])
    hi = np.stack([A[ytr == c].mean(0) + k * A[ytr == c].std(0) for c in range(NC)])
    hit = ((B[None, :, :] >= lo[:, None, :]) & (B[None, :, :] <= hi[:, None, :])).all(-1)
    n_hit = hit.sum(0)
    pred = np.full(len(B), -1, np.int8)
    uq = np.where(n_hit == 1)[0]
    if uq.size:
        pred[uq] = hit[:, uq].argmax(0).astype(np.int8)
    amb = np.where(n_hit != 1)[0]
    if amb.size:
        pred[amb] = mdc(Xtr, Xte, ytr)[amb]
    return pred, int((n_hit == 0).sum()), int((n_hit > 1).sum())


def sam(Xtr, Xte, ytr):
    mus = np.stack([Xtr[ytr == c].mean(0) for c in range(NC)])
    tn = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
    mn = mus / (np.linalg.norm(mus, axis=1, keepdims=True) + 1e-9)
    return (tn @ mn.T).argmax(1).astype(np.int8)


def sid_matrix(te, tr):
    def f(X):
        X = np.asarray(X, np.float64)
        Xs = X - X.min(axis=1, keepdims=True) + 1e-6
        P = Xs / Xs.sum(axis=1, keepdims=True)
        return P, np.log(P)
    P, lP = f(te); Q, lQ = f(tr)
    Hp = (P * lP).sum(1); Hq = (Q * lQ).sum(1)
    return Hp[:, None] + Hq[None, :] - P @ lQ.T - (Q @ lP.T).T


def sid(Xtr, Xte, ytr):
    return ytr[sid_matrix(Xte, Xtr).argmin(1)].astype(np.int8)


def svm(Xtr, Xte, ytr):
    sc = StandardScaler().fit(Xtr)
    return SVC(kernel='rbf', C=10, gamma='scale').fit(sc.transform(Xtr), ytr).predict(
        sc.transform(Xte)).astype(np.int8)


TRAD = ['MLC', 'MDC', 'MahaD', 'PP', 'SAM', 'SID']
FUNCS = {'MLC': mlc, 'MDC': mdc, 'MahaD': mahad, 'SAM': sam, 'SID': sid, 'SVM': svm}


def run_trad(Xtr, Xte, ytr):
    out = {}
    for a in TRAD:
        if a == 'PP':
            out[a] = parallelepiped(Xtr, Xte, ytr)[0]
        else:
            out[a] = FUNCS[a](Xtr, Xte, ytr)
    return out


# ---------- 3. 多时相 ----------
print('\n' + '=' * 100)
print('真实影像评价集 · 多时相（200 维）—— 训练 163 模板点 → 测试 109 评价点')
print('=' * 100)
mt = run_trad(tr_spec, te_spec, tr_lab)
mt['SVM'] = svm(tr_spec, te_spec, tr_lab)
mt['Hamming-NN'], amb_real = hamming_nn(discretize(te_spec), discretize(tr_spec), tr_lab)
print(f'{"算法":<12}{"OA%":>9}{"Kappa":>9}{"弃权":>7}')
print('-' * 100)
order = ['Hamming-NN'] + TRAD + ['SVM']
for a in order:
    p = mt[a]
    amb = amb_real if a == 'Hamming-NN' else 0
    tag = '  ← 本方法' if a == 'Hamming-NN' else ('  （传统）' if a in TRAD else '')
    print(f'{a:<12}{oa(p, te_lab):>9.2f}{kappa_of(p, te_lab):>9.4f}{amb:>7}{tag}')
n_correct = int((mt['Hamming-NN'] == te_lab).sum())
print(f'\n本方法错 {len(te_lab) - n_correct}/{len(te_lab)} 个，正确 {n_correct} 个')

# ---------- 4. 单时相（逐期 20 次） ----------
print('\n' + '=' * 100)
print('真实影像评价集 · 单时相（10 维/期，逐期跑 20 次）')
print('=' * 100)
r_single = {a: [] for a in TRAD + ['SVM']}
r_single_h = []
for d in range(N_BAND // N_PER_DATE):
    cols = np.arange(d * N_PER_DATE, (d + 1) * N_PER_DATE)
    o = run_trad(tr_spec[:, cols], te_spec[:, cols], tr_lab)
    o['SVM'] = svm(tr_spec[:, cols], te_spec[:, cols], tr_lab)
    for a in TRAD + ['SVM']:
        r_single[a].append(oa(o[a], te_lab))
    r_single_h.append(oa(hamming_nn(discretize(te_spec[:, cols], cols),
                                    discretize(tr_spec[:, cols], cols), tr_lab)[0], te_lab))

print(f'{"算法":<12}{"单时相均值":>12}{"单时相最优":>12}{"多时相":>10}{"净增益":>10}{"相对增益":>12}')
print('-' * 100)
for a in TRAD + ['SVM']:
    ms, bs, mtv = float(np.mean(r_single[a])), float(np.max(r_single[a])), oa(mt[a], te_lab)
    tag = '  （传统）' if a in TRAD else ''
    print(f'{a:<12}{ms:>12.2f}{bs:>12.2f}{mtv:>10.2f}{mtv - ms:>+10.2f}{(mtv - ms) / ms * 100:>+11.1f}%{tag}')
msh, bsh = float(np.mean(r_single_h)), float(np.max(r_single_h))
mth = oa(mt['Hamming-NN'], te_lab)
print(f'{"Hamming-NN":<12}{msh:>12.2f}{bsh:>12.2f}{mth:>10.2f}{mth - msh:>+10.2f}{(mth - msh) / msh * 100:>+11.1f}%   ← 本方法')

np.savez_compressed(
    os.path.join(OUT_DIR, 'real_traditional.npz'),
    **{f'mt_{k}': v for k, v in mt.items()}, te_lab=te_lab,
    order=np.array(order), trad=np.array(TRAD), classes=np.array(CLASSES),
    single_mean=np.array([np.mean(r_single[a]) for a in TRAD + ['SVM']]),
    single_best=np.array([np.max(r_single[a]) for a in TRAD + ['SVM']]),
    single_all=np.array([r_single[a] for a in TRAD + ['SVM']]),
    order_single=np.array(TRAD + ['SVM']),
    single_hamming=np.array(r_single_h), abstain=amb_real)
print('\n已保存: real_traditional.npz')
