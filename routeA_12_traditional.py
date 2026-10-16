# -*- coding: utf-8 -*-
"""
路线A · 步骤12：与传统分类器全面对比（重点实验）

用户要求：重点是与**传统分类器**对比，而非只跟 SVM/集成学习比。

传统遥感监督分类器（教科书级基线）：
  1. MLC   最大似然分类（每类高斯假设，PCA 降维后 QDA 实现）
  2. MDC   最小距离分类（到类均值的欧氏距离最近）
  3. MahaD 马氏距离分类（考虑类内协方差结构的距离）
  4. PP    平行六面体分类（各类在各维上的取值区间盒，多命中/零命中用 MDC 兜底）
  5. SAM   光谱角制图（只看光谱方向/形状，忽略幅度）
  6. SID   光谱信息散度（信息论角度的分布差异）
对照：本方法（多时相离散化指纹 + 汉明距离最近邻）、SVM（现代基准）

两个维度：
  A. 输入 = 多时相 200 维（本方法用 token；传统分类器用原始光谱）
  B. 输入 = 单时相 10 维（同样的传统分类器，逐期跑 20 次，报 mean / best）
     —— 用于量化"多时相时序信息"带给传统分类器的增益 vs 本方法的增益

协议与 routeA_08 主实验完全一致（同训练集、同 2700 个模拟混合像元、同档位）。
输出：分档 OA 表、单时相列、真实评价集、McNemar、图表
"""
import os
import sys
import numpy as np
from itertools import combinations
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else r'E:/混合像元实验/路线A_7月版'
EM = np.load(os.path.join(OUT_DIR, 'endmembers.npz'), allow_pickle=True)

CLASSES = ['人工表面', '水体', '河漫滩', '耕地']
NC = len(CLASSES)
M_LEVELS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
N_SEED, N_ANCHOR = 5, 5
thrT = EM['cube_thr'].T.astype(np.float32)
band_names = [str(b) for b in EM['band_names']]
N_BAND = len(band_names)
N_PER_DATE = N_BAND // 20          # 每期波段数（10）
N_PCA = 20


def discretize(x, cols=None):
    """按分位数阈值把连续光谱离散化为 token（1~L）。cols 支持只取部分波段（单时相消融）。"""
    x = np.asarray(x, np.float32)
    T = thrT if cols is None else thrT[:, cols]
    t = np.ones(x.shape, np.int8)
    for k in range(T.shape[0]):
        t += (x > T[k][None, :]).astype(np.int8)
    return t


def oa(p, y):
    return (p == y).mean() * 100


# ================= 1. 与主实验一致的训练/测试构造 =================
seed_spec, seed_lab, train_spec, train_lab = [], [], [], []
for ci, c in enumerate(CLASSES):
    em = EM[f'em_{c}']
    rc = EM[f'em_rc_{c}']
    S = EM[f'sample_{c}']
    RC = EM[f'sample_rc_{c}']
    seed_spec.append(em[:N_SEED]); seed_lab.append(np.full(N_SEED, ci, np.int8))
    key_seed = set(map(tuple, rc[:N_SEED]))
    mask_tr = np.array([tuple(r) not in key_seed for r in RC])
    train_spec.append(S[mask_tr]); train_lab.append(np.full(mask_tr.sum(), ci, np.int8))

seed_spec = np.vstack(seed_spec).astype(np.float32); seed_lab = np.concatenate(seed_lab)
train_spec = np.vstack(train_spec).astype(np.float32); train_lab = np.concatenate(train_lab)

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
print(f'训练 {train_spec.shape} | 模拟混合测试 {spec.shape} | 纯像元测试 {seed_spec.shape}')


# ================= 2. 传统分类器实现 =================
def mdc(Xtr, ytr, Xte, dd=None):
    """最小距离分类：到各类均值（标准化空间）欧氏距离最近。"""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    if dd:                                  # 可选 PCA 降维
        A, B = pca_(Xtr, Xte, dd)
    mus = np.stack([A[ytr == c].mean(0) for c in range(NC)])
    d = ((B[:, None, :] - mus[None, :, :]) ** 2).sum(-1)
    return d.argmin(1).astype(np.int8)


def pca_(Xtr, Xte, n=N_PCA):
    p = PCA(n_components=min(n, Xtr.shape[0] - 1, Xtr.shape[1]), random_state=42).fit(Xtr)
    return p.transform(Xtr), p.transform(Xte)


def mahad(Xtr, ytr, Xte, dd=N_PCA):
    """马氏距离分类：用池化协方差，PCA 降维保证协方差可逆。"""
    A, B = pca_(StandardScaler().fit(Xtr).transform(Xtr),
                StandardScaler().fit(Xtr).transform(Xte), dd)
    mus = np.stack([A[ytr == c].mean(0) for c in range(NC)])
    C = np.cov(A.T) + np.eye(A.shape[1]) * 1e-3
    Ci = np.linalg.pinv(C)
    d = np.empty((len(B), NC))
    for c in range(NC):
        e = B - mus[c]
        d[:, c] = np.einsum('ij,jk,ik->i', e, Ci, e)
    return d.argmin(1).astype(np.int8)


def mlc(Xtr, ytr, Xte, dd=N_PCA):
    """最大似然分类：每类多元高斯，PCA 降维后判别（等价于 QDA，含正则化）。"""
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    sc = StandardScaler().fit(Xtr)
    A, B = pca_(sc.transform(Xtr), sc.transform(Xte), dd)
    return QuadraticDiscriminantAnalysis(reg_param=0.3).fit(A, ytr).predict(B).astype(np.int8)


def parallelepiped(Xtr, ytr, Xte, k=1.5):
    """平行六面体分类：各类在各维上取 [μ-kσ, μ+kσ] 超盒。
    命中 1 类→判该类；命中 ≥2 类→MDC 兜底；命中 0 类→MDC 兜底。"""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    lo = np.stack([A[ytr == c].mean(0) - k * A[ytr == c].std(0) for c in range(NC)])
    hi = np.stack([A[ytr == c].mean(0) + k * A[ytr == c].std(0) for c in range(NC)])
    hit = ((B[None, :, :] >= lo[:, None, :]) & (B[None, :, :] <= hi[:, None, :])).all(-1)
    n_hit = hit.sum(0)
    pred = np.full(len(B), -1, np.int8)
    uniq = np.where(n_hit == 1)[0]
    if uniq.size:
        pred[uniq] = hit[:, uniq].argmax(0).astype(np.int8)
    amb = np.where(n_hit != 1)[0]
    if amb.size:
        fallback = mdc(Xtr, ytr, Xte)      # MDC 兜底（只取落在 amb 位置的结果）
        pred[amb] = fallback[amb]
    return pred, int((n_hit == 0).sum()), int((n_hit > 1).sum())


def sam(Xtr, ytr, Xte):
    """光谱角制图：与各类均值向量夹角最小。"""
    mus = np.stack([Xtr[ytr == c].mean(0) for c in range(NC)])
    tn = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
    mn = mus / (np.linalg.norm(mus, axis=1, keepdims=True) + 1e-9)
    return (tn @ mn.T).argmax(1).astype(np.int8)


def sid_matrix(te, tr):
    """SID = Σ[p·ln(p/q) + q·ln(q/p)]，逐样本平移为正后归一化为概率分布。"""
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


def sid(Xtr, ytr, Xte):
    return ytr[sid_matrix(Xte, Xtr).argmin(axis=1)].astype(np.int8)


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


TRAD = ['MLC', 'MDC', 'MahaD', 'PP', 'SAM', 'SID']


def run_all(Xtr, ytr, Xte):
    """在给定输入空间上跑全套传统分类器，返回 {算法: 预测}。"""
    out = {}
    out['MLC'] = mlc(Xtr, ytr, Xte)
    out['MDC'] = mdc(Xtr, ytr, Xte)
    out['MahaD'] = mahad(Xtr, ytr, Xte)
    out['PP'], n0, nm = parallelepiped(Xtr, ytr, Xte)
    out['SAM'] = sam(Xtr, ytr, Xte)
    out['SID'] = sid(Xtr, ytr, Xte)
    return out, (n0, nm)


# ================= 3. 多时相输入（200 维） =================
print('\n' + '=' * 112)
print('A. 多时相输入（200 维：20 期 × 10 波段）')
print('=' * 112)
preds_mt, (pp_n0, pp_nm) = run_all(train_spec, train_lab, spec)
pp_mt = (pp_n0, pp_nm)
preds_mt['Hamming-NN'], amb = hamming_system(discretize(spec), discretize(train_spec), train_lab)
sc = StandardScaler().fit(train_spec)
preds_mt['SVM'] = SVC(kernel='rbf', C=10, gamma='scale').fit(
    sc.transform(train_spec), train_lab).predict(sc.transform(spec)).astype(np.int8)
print(f'平行六面体(多时相): 零命中 {pp_n0} ({pp_n0/len(spec)*100:.2f}%)、多命中 {pp_nm} ({pp_nm/len(spec)*100:.2f}%)，二者均 MDC 兜底')
print(f'本方法弃权 {amb} ({amb/len(spec)*100:.2f}%)')

# ================= 4. 单时相输入（10 维 × 20 期） =================
print('\n' + '=' * 112)
print('B. 单时相输入（10 维/期，逐期跑 20 次）')
print('=' * 112)
single = {a: [] for a in TRAD}
single_pp = []
single_h = []
for d in range(N_BAND // N_PER_DATE):
    cols = np.arange(d * N_PER_DATE, (d + 1) * N_PER_DATE)
    p, (n0, nm) = run_all(train_spec[:, cols], train_lab, spec[:, cols])
    for a in TRAD:
        single[a].append(oa(p[a], label))
    single_pp.append((n0, nm))
    single_h.append(oa(hamming_system(discretize(spec[:, cols], cols),
                                      discretize(train_spec[:, cols], cols),
                                      train_lab)[0], label))
single_mean = {a: float(np.mean(single[a])) for a in TRAD}
single_best = {a: float(np.max(single[a])) for a in TRAD}
print(f'单时相下本方法平均 {np.mean(single_h):.2f}%  最优期 {np.max(single_h):.2f}%')

# ================= 5. 汇总：分档 OA 表 =================
print('\n' + '=' * 112)
print('表A  多时相（200 维）各算法 × 混合档 OA (%)')
print('=' * 112)
hdr = f'{"算法":<12}' + ''.join(f'{m:>8.2f}' for m in M_LEVELS) + f'{"总体":>9}{"衰减":>8}'
print(hdr); print('-' * 112)
tier = {}
order_mt = ['Hamming-NN'] + TRAD + ['SVM']
for a in order_mt:
    p = preds_mt[a]
    row = [oa(p[np.isclose(m_arr, m)], label[np.isclose(m_arr, m)]) for m in M_LEVELS]
    tier[a] = row
    tag = '  ← 本方法' if a == 'Hamming-NN' else ('  （传统）' if a in TRAD else '')
    print(f'{a:<12}' + ''.join(f'{v:>8.2f}' for v in row) + f'{oa(p, label):>9.2f}{row[0]-row[-1]:>8.2f}{tag}')

print('\n' + '=' * 112)
print('表B  单时相 vs 多时相：传统分类器的时序增益（200 维总体 OA）')
print('=' * 112)
print(f'{"算法":<12}{"单时相均值":>12}{"单时相最优":>12}{"多时相":>10}{"净增益":>10}{"相对增益":>12}')
print('-' * 112)
gain_rows = []
for a in TRAD:
    mt = oa(preds_mt[a], label)
    mean_s, best_s = single_mean[a], single_best[a]
    gain = mt - mean_s
    rel = gain / mean_s * 100
    gain_rows.append((a, mean_s, best_s, mt, gain, rel))
    print(f'{a:<12}{mean_s:>12.2f}{best_s:>12.2f}{mt:>10.2f}{gain:>+10.2f}{rel:>+11.1f}%')
mt_h = oa(preds_mt['Hamming-NN'], label)
print(f'{"Hamming-NN":<12}{np.mean(single_h):>12.2f}{np.max(single_h):>12.2f}{mt_h:>10.2f}'
      f'{mt_h - np.mean(single_h):>+10.2f}{(mt_h - np.mean(single_h)) / np.mean(single_h) * 100:>+11.1f}%   ← 本方法')

# ================= 6. McNemar（本方法 vs 各传统分类器） =================
print('\n' + '=' * 112)
print('McNemar 检验（本方法 vs 传统分类器，n=%d）' % len(label))
print('=' * 112)
mcn = {}
for a in TRAD + ['SVM']:
    b01 = int(((preds_mt['Hamming-NN'] == label) & (preds_mt[a] != label)).sum())
    b10 = int(((preds_mt['Hamming-NN'] != label) & (preds_mt[a] == label)).sum())
    pv = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    mcn[a] = (b01, b10, float(pv))
    win = '本方法显著更优' if (b01 > b10 and pv < 0.05) else ('对方显著更优' if (b10 > b01 and pv < 0.05) else '无显著差异')
    print(f'  vs {a:<8}: {b01:>5}/{b10:<5}  p={pv:.4f}  → {win}')

# ================= 7. 真实评价点（109 个） =================
print('\n' + '=' * 112)
print('真实影像评价集对照（沿用 routeA_05 的结果文件，若存在）')
print('=' * 112)
real_path = os.path.join(OUT_DIR, 'real_check.npz')
if os.path.exists(real_path):
    R = np.load(real_path, allow_pickle=True)
    print(f'  已存在 real_check.npz，键: {[k for k in R.files if k.startswith("pred")]}')

# ================= 8. 保存 =================
np.savez_compressed(
    os.path.join(OUT_DIR, 'traditional_results.npz'),
    **{f'mt_{k}': v for k, v in preds_mt.items()},
    label=label, m=m_arr, pair=pair, seed_lab=seed_lab,
    tier_mt=np.array([tier[a] for a in order_mt]), order_mt=np.array(order_mt),
    m_levels=np.array(M_LEVELS, np.float32), classes=np.array(CLASSES),
    single_mean=np.array([single_mean[a] for a in TRAD]), single_best=np.array([single_best[a] for a in TRAD]),
    single_all=np.array([single[a] for a in TRAD]), trad_order=np.array(TRAD),
    single_hamming=np.array(single_h),
    mcnemar=np.array([(a, mcn[a][0], mcn[a][1], mcn[a][2]) for a in mcn],
                     dtype=[('algo', 'U16'), ('b01', 'i4'), ('b10', 'i4'), ('p', 'f8')]))
print('\n已保存: traditional_results.npz')
print(f'全影像离散化阈值来源: {OUT_DIR}/endmembers.npz (cube_thr {EM["cube_thr"].shape})')
