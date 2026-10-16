# -*- coding: utf-8 -*-
"""
路线A · 步骤3：多算法统一分类

5 种算法，同一套训练样本（严格分离版，130 点）、同一套模拟测试集（10800 像元）
  1. 本方法 Hamming-NN : 离散化 token -> 汉明距离最近邻（忠实复现 process03 决策规则）
  2. SVM               : RBF 核，逐波段 z-score 标准化后的原始光谱
  3. RF                : 随机森林 500 棵
  4. MLC               : PCA(20) + 正则化 QDA（每类高斯最大似然）
  5. SAM               : 光谱角匹配（类均值作参考）
  附加对照 SVM-token   ：SVM 直接吃离散化 token，检验"离散化本身"的贡献

输出 E:\\混合像元实验\\路线A\\predictions.npz
"""
import os
import numpy as np

OUT_DIR = r'E:/混合像元实验/路线A'
EM = np.load(os.path.join(OUT_DIR, 'endmembers.npz'), allow_pickle=True)
SIM = np.load(os.path.join(OUT_DIR, 'simulated.npz'), allow_pickle=True)
OUT = os.path.join(OUT_DIR, 'predictions.npz')

CLASSES = list(SIM['classes'])
N_CLASSES = len(CLASSES)
thr = EM['cube_thr']                      # (200, 4)
thrT = thr.T.astype(np.float32)           # (4, 200) 便于广播


def discretize(spec):
    """Step1 规则：x<=t1→1 ... x>t4→5"""
    x = np.asarray(spec, np.float32)
    tok = np.ones(x.shape, np.int8)
    for k in range(thrT.shape[0]):
        tok += (x > thrT[k][None, :]).astype(np.int8)
    return tok


tr_spec = SIM['train_spec'].astype(np.float32); tr_lab = SIM['train_label']
te_spec = SIM['spec'].astype(np.float32);      te_lab = SIM['label']
pu_spec = SIM['pure_spec'].astype(np.float32); pu_lab = SIM['pure_label']

tr_tok = discretize(tr_spec)
te_tok = discretize(te_spec)
pu_tok = discretize(pu_spec)

print('=' * 74)
print(f'训练集(严格分离) {tr_spec.shape}  |  模拟测试集 {te_spec.shape}  |  纯像元 {pu_spec.shape}')
print(f'训练样本分布: ' + '  '.join(f'{CLASSES[i]}={int((tr_lab==i).sum())}' for i in range(N_CLASSES)))
print('=' * 74)

preds = {}


# ---------- 1. 本方法：汉明距离最近邻 ----------
def hamming_predict(te_tok, tr_tok, tr_lab, chunk=400):
    n = len(te_tok)
    pred = np.full(n, -1, np.int8)          # -1 = 未分类（超阈值或跨类并列）
    unclassified_amb = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = (te_tok[s:e, None, :] != tr_tok[None, :, :]).sum(axis=2)   # (chunk, n_tr)
        md = d.min(axis=1)
        for i in range(e - s):
            labs = np.unique(tr_lab[d[i] == md[i]])
            if labs.size == 1:
                pred[s + i] = labs[0]
            else:
                unclassified_amb += 1
    return pred, unclassified_amb


print('\n[1/6] 本方法 Hamming-NN ...')
p, amb = hamming_predict(te_tok, tr_tok, tr_lab)
preds['Hamming-NN'] = p
print(f'      完成，跨类并列未分类 {amb} 个 ({amb/len(p)*100:.2f}%)')

# ---------- 2~4. 原始光谱输入 ----------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(tr_spec)
Xtr, Xte, Xpu = sc.transform(tr_spec), sc.transform(te_spec), sc.transform(pu_spec)

print('[2/6] SVM (RBF) ...')
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=10.0, gamma='scale').fit(Xtr, tr_lab)
preds['SVM'] = svm.predict(Xte).astype(np.int8)

print('[3/6] RF ...')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1).fit(Xtr, tr_lab)
preds['RF'] = rf.predict(Xte).astype(np.int8)

print('[4/6] MLC (PCA20 + 正则化 QDA) ...')
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
npc = min(20, len(tr_spec) - 1)
pca = PCA(n_components=npc, random_state=42).fit(Xtr)
qda = QuadraticDiscriminantAnalysis(reg_param=0.3).fit(pca.transform(Xtr), tr_lab)
preds['MLC'] = qda.predict(pca.transform(Xte)).astype(np.int8)

# ---------- 5. SAM 光谱角 ----------
print('[5/6] SAM (光谱角) ...')


def sam_predict(te, tr, tr_lab):
    means = np.stack([tr[tr_lab == c].mean(axis=0) for c in range(N_CLASSES)])
    tn = te / (np.linalg.norm(te, axis=1, keepdims=True) + 1e-9)
    mn = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-9)
    return (tn @ mn.T).argmax(axis=1).astype(np.int8)


preds['SAM'] = sam_predict(te_spec, tr_spec, tr_lab)

# ---------- 附加：SVM 吃 token ----------
print('[6/6] SVM on token（离散化贡献对照）...')
svm_t = SVC(kernel='rbf', C=10.0, gamma='scale').fit(tr_tok.astype(np.float32), tr_lab)
preds['SVM-token'] = svm_t.predict(te_tok.astype(np.float32)).astype(np.int8)


# ---------- 训练集自评 + 纯像元 ----------
def evaluate(pred, y):
    ok = pred == y
    return ok.mean() * 100


print('\n' + '=' * 74)
print(f'{"算法":<14}{"模拟集OA":>12}{"纯像元OA":>12}{"训练集OA":>12}{"未分类":>10}')
print('=' * 74)
pure_preds, train_preds = {}, {}
for name, pr in preds.items():
    if name == 'Hamming-NN':
        pp, _ = hamming_predict(pu_tok, tr_tok, tr_lab)
        tp, _ = hamming_predict(tr_tok, tr_tok, tr_lab)
    elif name == 'SVM':
        pp, tp = svm.predict(Xpu).astype(np.int8), svm.predict(Xtr).astype(np.int8)
    elif name == 'RF':
        pp, tp = rf.predict(Xpu).astype(np.int8), rf.predict(Xtr).astype(np.int8)
    elif name == 'MLC':
        pp = qda.predict(pca.transform(Xpu)).astype(np.int8)
        tp = qda.predict(pca.transform(Xtr)).astype(np.int8)
    elif name == 'SAM':
        pp, tp = sam_predict(pu_spec, tr_spec, tr_lab), sam_predict(tr_spec, tr_spec, tr_lab)
    else:
        pp = svm_t.predict(pu_tok.astype(np.float32)).astype(np.int8)
        tp = svm_t.predict(tr_tok.astype(np.float32)).astype(np.int8)
    pure_preds[name], train_preds[name] = pp, tp
    n_unc = int((pr == -1).sum())
    print(f'{name:<14}{evaluate(pr, te_lab):>11.2f}%{evaluate(pp, pu_lab):>11.2f}%'
          f'{evaluate(tp, tr_lab):>11.2f}%{n_unc:>10}')

np.savez_compressed(
    OUT,
    **{f'pred_{k}': v for k, v in preds.items()},
    **{f'pure_{k}': v for k, v in pure_preds.items()},
    **{f'train_{k}': v for k, v in train_preds.items()},
    te_lab=te_lab, pu_lab=pu_lab, tr_lab=tr_lab,
    algorithm_names=np.array(list(preds.keys())),
    classes=np.array(CLASSES),
)
print(f'\n已保存: {OUT}')
