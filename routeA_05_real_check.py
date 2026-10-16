# -*- coding: utf-8 -*-
"""
路线A · 步骤5：真实影像对照实验（快速版）

目的：模拟集上本方法垫底（77.91%），必须判断是"方法弱"还是"模拟/决策规则低估了它"。
做法：训练 = 170 个模板样本点，测试 = 109 个独立评价样本点（与系统 89.91% 同口径）。
     同时输出 Hamming-NN 的"弃权分解"：跨类并列未分类率、已判定像元精度。
输出 E:/混合像元实验/路线A/real_check.npz
"""
import os
import sys
import numpy as np
import rasterio
import pyogrio

IMG_DIR = r'E:/指纹/124036中/图像'
TRAIN_SH = sys.argv[2] if len(sys.argv) > 2 else r'E:/指纹/124036中/样本/实验样本'
TEST_SH = r'E:/指纹/124036中/样本/评价样本'
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else r'E:/混合像元实验/路线A'
EM = np.load(os.path.join(OUT_DIR, 'endmembers.npz'), allow_pickle=True)

CLASSES = ['人工表面', '水体', '河漫滩', '耕地']
NC = len(CLASSES)
thrT = EM['cube_thr'].T.astype(np.float32)     # (4, 200)
tifs = list(EM['band_names'])


def discretize(x):
    x = np.asarray(x, np.float32)
    t = np.ones(x.shape, np.int8)
    for k in range(thrT.shape[0]):
        t += (x > thrT[k][None, :]).astype(np.int8)
    return t


# ---------- 1. 整幅读入立方体（337MB，秒级） ----------
print('读入全影像立方体 ...')
with rasterio.open(os.path.join(IMG_DIR, tifs[0])) as s:
    H, W, transform = s.height, s.width, s.transform
cube = np.empty((len(tifs), H, W), np.float32)
for i, f in enumerate(tifs):
    with rasterio.open(os.path.join(IMG_DIR, f)) as s:
        cube[i] = s.read(1)
print(f'  立方体 {cube.shape}')


def read_spec(shp_dir, label):
    xs_all, ys_all, lb_all = [], [], []
    for ci, c in enumerate(CLASSES):
        p = os.path.join(shp_dir, c + '.shp')
        if not os.path.exists(p):
            print(f'  !! 缺 {p}')
            continue
        gdf = pyogrio.read_dataframe(p)
        xs = np.array([g.x for g in gdf.geometry])
        ys = np.array([g.y for g in gdf.geometry])
        cols = np.floor((xs - transform.c) / transform.a).astype(int)
        rows = np.floor((transform.f - ys) / (-transform.e)).astype(int)
        ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        xs_all.append(xs[ok]); ys_all.append(ys[ok])
        lb_all.append(np.full(ok.sum(), ci, np.int8))
    xs_all = np.concatenate(xs_all); ys_all = np.concatenate(ys_all)
    lb_all = np.concatenate(lb_all)
    cols = np.floor((xs_all - transform.c) / transform.a).astype(int)
    rows = np.floor((transform.f - ys_all) / (-transform.e)).astype(int)
    spec = cube[:, rows, cols].T.astype(np.float32)          # (n, 200)
    print(f'  {label}: {spec.shape}  ' + '  '.join(
        f'{CLASSES[i]}={int((lb_all==i).sum())}' for i in range(NC)))
    return spec, lb_all


print('\n读取样本光谱 ...')
tr_spec, tr_lab = read_spec(TRAIN_SH, '训练(模板样本)')
te_spec, te_lab = read_spec(TEST_SH, '测试(评价样本)')

tr_tok, te_tok = discretize(tr_spec), discretize(te_spec)
n0_tr = int((tr_spec == 0).any(axis=1).sum())
n0_te = int((te_spec == 0).any(axis=1).sum())
print(f'  含 0 值(NoData/真实0)像元点: 训练 {n0_tr}  评价 {n0_te}')


def oa(p, y):
    return (p == y).mean() * 100


def hamming_nn(te_t, tr_t, tr_l):
    """系统决策规则：全局最小距离；跨类并列 -> 弃权(-1)"""
    d = (te_t[:, None, :] != tr_t[None, :, :]).sum(axis=2)
    pred = np.full(len(te_t), -1, np.int8)
    amb = 0
    for i in range(len(te_t)):
        u = np.unique(tr_l[d[i] == d[i].min()])
        if u.size == 1:
            pred[i] = u[0]
        else:
            amb += 1
    return pred, amb


def hamming_nn_break(te_t, tr_t, tr_l):
    """变体：跨类并列时按"并列类模板的平均距离"破平局（不弃权）"""
    d = (te_t[:, None, :] != tr_t[None, :, :]).sum(axis=2).astype(np.float32)
    pred = np.empty(len(te_t), np.int8)
    amb = 0
    for i in range(len(te_t)):
        m = d[i].min()
        tied = d[i] == m
        u = np.unique(tr_l[tied])
        if u.size == 1:
            pred[i] = u[0]
        else:
            amb += 1
            # 破平局：每个并列类的模板平均距离
            scores = [d[i][tied & (tr_l == c)].mean() for c in u]
            pred[i] = u[int(np.argmin(scores))]
    return pred, amb


results = {}
results['Hamming-NN'], amb_sys = hamming_nn(te_tok, tr_tok, tr_lab)
results['Hamming-NN+破平局'], amb_brk = hamming_nn_break(te_tok, tr_tok, tr_lab)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def sid_matrix(te, tr):
    """光谱信息散度 SID(te, tr) -> (n_te, n_tr)。负值波段逐样本平移为正。"""
    def to_logdist(X):
        X = np.asarray(X, np.float64)
        Xs = X - X.min(axis=1, keepdims=True) + 1e-6
        P = Xs / Xs.sum(axis=1, keepdims=True)
        return P, np.log(P)
    P, lP = to_logdist(te)
    Q, lQ = to_logdist(tr)
    Hp = (P * lP).sum(axis=1)
    Hq = (Q * lQ).sum(axis=1)
    return Hp[:, None] + Hq[None, :] - P @ lQ.T - (Q @ lP.T).T


sc = StandardScaler().fit(tr_spec)
A, B = sc.transform(tr_spec), sc.transform(te_spec)
results['SVM'] = SVC(kernel='rbf', C=10, gamma='scale').fit(A, tr_lab).predict(B).astype(np.int8)
results['SID'] = tr_lab[sid_matrix(te_spec, tr_spec).argmin(axis=1)].astype(np.int8)
pc = PCA(n_components=min(20, len(tr_spec) - 1), random_state=42).fit(A)
results['MLC'] = QuadraticDiscriminantAnalysis(reg_param=0.3).fit(
    pc.transform(A), tr_lab).predict(pc.transform(B)).astype(np.int8)
mus = np.stack([tr_spec[tr_lab == c].mean(0) for c in range(NC)])
tn = te_spec / (np.linalg.norm(te_spec, axis=1, keepdims=True) + 1e-9)
mn = mus / (np.linalg.norm(mus, axis=1, keepdims=True) + 1e-9)
results['SAM'] = (tn @ mn.T).argmax(1).astype(np.int8)
results['SVM-token'] = SVC(kernel='rbf', C=10, gamma='scale').fit(
    tr_tok.astype(np.float32), tr_lab).predict(te_tok.astype(np.float32)).astype(np.int8)

print('\n' + '=' * 92)
print('真实影像 · 单轮匹配（训练=模板样本点，测试=独立评价样本点；未分类计为错分）')
print('=' * 92)
print(f'{"算法":<20}{"OA%":>9}{"Kappa":>9}{"弃权数":>8}   ' + ''.join(f'{c + "PA":>10}' for c in CLASSES))
print('-' * 92)
real = {}
for name, p in results.items():
    amb = amb_sys if name == 'Hamming-NN' else (amb_brk if '+' in name else 0)
    cm = np.zeros((NC, NC), int)
    for i in range(len(p)):
        if p[i] >= 0:
            cm[te_lab[i], p[i]] += 1
    po = np.trace(cm) / len(p)
    pe = (cm.sum(0) * cm.sum(1)).sum() / len(p) ** 2
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0
    pas = [cm[i, i] / max(1, cm[i].sum()) * 100 for i in range(NC)]
    real[name] = p
    print(f'{name:<20}{oa(p, te_lab):>9.2f}{kappa:>9.4f}{amb:>8}   '
          + ''.join(f'{v:>10.1f}' for v in pas))

print('\n[Hamming-NN 弃权分解]  弃权(跨类并列)数 =', amb_sys,
      f'({amb_sys / len(te_tok) * 100:.1f}%)')
p_sys = results['Hamming-NN']
decided = p_sys >= 0
print(f'  已判定像元: {int(decided.sum())} 个, 其中正确 {int((p_sys[decided] == te_lab[decided]).sum())}'
      f'  已判定精度 = {oa(p_sys[decided], te_lab[decided]):.2f}%')

np.savez_compressed(os.path.join(OUT_DIR, 'real_check.npz'),
                    **{f'pred_{k}': v for k, v in real.items()}, te_lab=te_lab,
                    algos=np.array(list(real.keys())), classes=np.array(CLASSES))
print('\n已保存: real_check.npz')
