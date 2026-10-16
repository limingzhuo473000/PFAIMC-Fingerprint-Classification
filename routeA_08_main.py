# -*- coding: utf-8 -*-
"""
路线A · 步骤8（最终主实验）：端元分组设计 —— 合成组与训练锚点组严格分离

设计（解决"严格分离挖走全部纯模板惩罚 NN"与"循环论证"的两难）：
  每类 10 条端元按均质性 CV 排序：
    合成组 = CV 最低的 5 条 —— 只用于 LMM 合成混合像元（测试集），绝不在训练集出现
    锚点组 = 其余 5 条 —— 作为纯像元模板留在训练集（贴近系统"样本点全做模板"的实际用法）
  训练集 = 170 样本 - 20 合成组 = 150 点（含每类 5 条纯锚点）
  纯像元测试 = 20 条合成组端元（m=1.0，且不在训练集）
  测试集 = 6 对 × 2 方向 × 9 档 × 5×5 组合 = 2700 混合像元

所有算法用同一训练集，输出分档 OA / PA / McNemar。
输出 E:/混合像元实验/路线A/main_results.npz
"""
import os
import sys
import numpy as np
from itertools import combinations
from scipy.stats import binomtest

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else r'E:/混合像元实验/路线A'
EM = np.load(os.path.join(OUT_DIR, 'endmembers.npz'), allow_pickle=True)

CLASSES = ['人工表面', '水体', '河漫滩', '耕地']
NC = len(CLASSES)
M_LEVELS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
N_SEED, N_ANCHOR = 5, 5
thrT = EM['cube_thr'].T.astype(np.float32)


def discretize(x):
    x = np.asarray(x, np.float32)
    t = np.ones(x.shape, np.int8)
    for k in range(thrT.shape[0]):
        t += (x > thrT[k][None, :]).astype(np.int8)
    return t


def oa(p, y):
    return (p == y).mean() * 100


# ---------- 1. 端元分组 ----------
seed_spec, seed_lab = [], []       # 合成组
anchor_spec, anchor_lab = [], []   # 锚点组（进训练集）
train_spec, train_lab = [], []     # 全部训练点

for ci, c in enumerate(CLASSES):
    em = EM[f'em_{c}']                     # (10, 200) 已按 CV 升序
    rc = EM[f'em_rc_{c}']
    S = EM[f'sample_{c}']                  # 该类全部样本
    RC = EM[f'sample_rc_{c}']
    seed_spec.append(em[:N_SEED]);  seed_lab.append(np.full(N_SEED, ci, np.int8))
    anchor_spec.append(em[N_SEED:]); anchor_lab.append(np.full(N_ANCHOR, ci, np.int8))
    key_seed = set(map(tuple, rc[:N_SEED]))
    mask_tr = np.array([tuple(r) not in key_seed for r in RC])
    train_spec.append(S[mask_tr]); train_lab.append(np.full(mask_tr.sum(), ci, np.int8))
    print(f'{c:<6} 合成组 5 (CV {EM[f"em_cv_{c}"][:5].mean():.4f})  锚点组 5  '
          f'训练点(去合成组) {int(mask_tr.sum())}/{len(S)}')

seed_spec = np.vstack(seed_spec).astype(np.float32); seed_lab = np.concatenate(seed_lab)
anchor_spec = np.vstack(anchor_spec).astype(np.float32); anchor_lab = np.concatenate(anchor_lab)
train_spec = np.vstack(train_spec).astype(np.float32); train_lab = np.concatenate(train_lab)
print(f'训练 {train_spec.shape}  合成组 {seed_spec.shape}（纯像元测试兼合成种子）')

# ---------- 2. LMM 合成 ----------
specs, labels, ms, pairs = [], [], [], []
for pi, (a, b) in enumerate(combinations(range(NC), 2)):
    for dom, sub in [(a, b), (b, a)]:
        E_d, E_s = seed_spec[seed_lab == dom], seed_spec[seed_lab == sub]
        for m in M_LEVELS:
            mix = m * E_d[:, None, :] + (1 - m) * E_s[None, :, :]
            specs.append(mix.reshape(-1, mix.shape[-1]))
            labels.append(np.full(mix.shape[0] * mix.shape[1], dom, np.int8))
            ms.append(np.full(mix.shape[0] * mix.shape[1], m, np.float32))
            pairs.append(np.full(mix.shape[0] * mix.shape[1], pi, np.int8))
spec = np.vstack(specs).astype(np.float32)
label = np.concatenate(labels)
m_arr = np.concatenate(ms); pair = np.concatenate(pairs)
print(f'模拟混合像元 {spec.shape}（6对 × 2方向 × {len(M_LEVELS)}档 × 25组合）')

train_tok = discretize(train_spec)
te_tok = discretize(spec)
pure_tok = discretize(seed_spec)       # 纯像元测试 = 合成组端元（不在训练集）

# ---------- 3. 六种算法 ----------
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def sid_matrix(te, tr):
    """光谱信息散度 SID(te, tr) -> (n_te, n_tr)。
    SID = Σ [p·ln(p/q) + q·ln(q/p)]，p/q 为光谱归一化概率分布。
    波段含负值(NDVI/NDBI/NDWI)，逐样本平移为正后再归一化（惯用处理）。"""
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


preds = {}


def hamming_system(te_t, tr_t, tr_l, chunk=500):
    n = len(te_t)
    pred = np.full(n, -1, np.int8)
    amb = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = (te_t[s:e, None, :] != tr_t[None, :, :]).sum(axis=2)
        md = d.min(axis=1)
        for i in range(e - s):
            u = np.unique(tr_l[d[i] == md[i]])
            if u.size == 1:
                pred[s + i] = u[0]
            else:
                amb += 1
    return pred, amb


print('\n训练与预测 ...')
preds['Hamming-NN'], amb = hamming_system(te_tok, train_tok, train_lab)
print(f'  [1/6] Hamming-NN  弃权 {amb} ({amb/len(te_tok)*100:.2f}%)')
p_pure_h, _ = hamming_system(pure_tok, train_tok, train_lab)

sc = StandardScaler().fit(train_spec)
Xtr, Xte, Xpu = sc.transform(train_spec), sc.transform(spec), sc.transform(seed_spec)
svm = SVC(kernel='rbf', C=10, gamma='scale').fit(Xtr, train_lab)
preds['SVM'] = svm.predict(Xte).astype(np.int8)
print('  [2/6] SVM')
preds['SID'] = train_lab[sid_matrix(spec, train_spec).argmin(axis=1)].astype(np.int8)
print('  [3/6] SID (光谱信息散度, 逐模板最小SID)')
pc = PCA(n_components=min(20, len(train_spec) - 1), random_state=42).fit(Xtr)
preds['MLC'] = QuadraticDiscriminantAnalysis(reg_param=0.3).fit(
    pc.transform(Xtr), train_lab).predict(pc.transform(Xte)).astype(np.int8)
print('  [4/6] MLC')
mus = np.stack([train_spec[train_lab == c].mean(0) for c in range(NC)])
tn = spec / (np.linalg.norm(spec, axis=1, keepdims=True) + 1e-9)
mn = mus / (np.linalg.norm(mus, axis=1, keepdims=True) + 1e-9)
preds['SAM'] = (tn @ mn.T).argmax(1).astype(np.int8)
print('  [5/6] SAM')
preds['SVM-token'] = SVC(kernel='rbf', C=10, gamma='scale').fit(
    train_tok.astype(np.float32), train_lab).predict(te_tok.astype(np.float32)).astype(np.int8)
print('  [6/6] SVM-token')
p_pure_svm = svm.predict(Xpu).astype(np.int8)

# ---------- 4. 分档 OA ----------
print('\n' + '=' * 106)
print('主实验（锚点组设计）各算法 × 混合档 OA (%)')
print('=' * 106)
hdr = f'{"算法":<12}' + ''.join(f'{m:>8.2f}' for m in M_LEVELS) + f'{"总体":>9}{"衰减":>8}'
print(hdr); print('-' * 106)
tier = {}
for a in preds:
    p = preds[a]
    row = [oa(p[np.isclose(m_arr, m)], label[np.isclose(m_arr, m)]) for m in M_LEVELS]
    tier[a] = row
    print(f'{a:<12}' + ''.join(f'{v:>8.2f}' for v in row) + f'{oa(p, label):>9.2f}{row[0]-row[-1]:>8.2f}')
print(f'\n纯像元测试(20条,不在训练集): Hamming-NN {oa(p_pure_h, seed_lab):.1f}%  SVM {oa(p_pure_svm, seed_lab):.1f}%')

# ---------- 5. McNemar ----------
print('\n' + '=' * 106)
print('McNemar 检验（Hamming-NN vs 各算法，n=%d）' % len(label))
print('=' * 106)
mcn = {}
for a in ['SVM', 'SID', 'MLC', 'SAM', 'SVM-token']:
    b01 = int(((preds['Hamming-NN'] == label) & (preds[a] != label)).sum())
    b10 = int(((preds['Hamming-NN'] != label) & (preds[a] == label)).sum())
    pv = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    mcn[a] = (b01, b10, float(pv))
    win = '本方法显著更优' if (b01 > b10 and pv < 0.05) else ('对方显著更优' if (b10 > b01 and pv < 0.05) else '无显著差异')
    print(f'  vs {a:<11}: {b01:>5}/{b10:<5}  p={pv:.4f}  → {win}')

np.savez_compressed(
    os.path.join(OUT_DIR, 'main_results.npz'),
    **{f'pred_{k}': v for k, v in preds.items()},
    label=label, m=m_arr, pair=pair, seed_lab=seed_lab, pred_pure_hamming=p_pure_h,
    pred_pure_svm=p_pure_svm, abstain_hamming=amb,
    tier=np.array([tier[a] for a in preds]), algo_order=np.array(list(preds.keys())),
    m_levels=np.array(M_LEVELS, np.float32), classes=np.array(CLASSES),
    train_spec=train_spec, train_lab=train_lab,
    mcnemar=np.array([(a, mcn[a][0], mcn[a][1], mcn[a][2]) for a in mcn],
                     dtype=[('algo', 'U16'), ('b01', 'i4'), ('b10', 'i4'), ('p', 'f8')]))
print('\n已保存: main_results.npz')
