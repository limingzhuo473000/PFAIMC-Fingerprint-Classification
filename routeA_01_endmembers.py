# -*- coding: utf-8 -*-
"""
路线A · 步骤1：端元光谱库提取 + 分位数阈值标定

输入
  E:\\指纹\\124036中\\图像            200 个 TIF（20 期 × 10 波段）
  E:\\指纹\\124036中\\样本\\实验样本   4 类模板样本 SHP
输出
  E:\\混合像元实验\\路线A\\endmembers.npz
      cube_thr      (200, 4)  每波段分位数阈值（5 级 → 4 个分位点，0 参与统计）
      band_names    (200,)    波段文件名，与 Step1 字典序一致
      em_<类名>     (n, 200)  该类纯端元光谱
      em_cv_<类名>  (n,)      端元均质性得分（越低越纯）
      sample_<类名> (n, 200)  该类全部样本点光谱
      sample_rc_<类名> (n, 2) 样本点行列号
      sample_xy_<类名> (n, 2) 样本点投影坐标
"""
import os
import sys
import numpy as np
import pyogrio

IMG_DIR = r'E:/指纹/124036中/图像'
SHP_DIR = sys.argv[2] if len(sys.argv) > 2 else r'E:/指纹/124036中/样本/实验样本'
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else r'E:/混合像元实验/路线A'
OUT_NPZ = os.path.join(OUT_DIR, 'endmembers.npz')

N_CLASSES = int(sys.argv[3]) if len(sys.argv) > 3 else 5   # 离散级数（默认 5，与 124036 基线一致）
EXCLUDE_ZERO = False   # 基线配置：0 值参与分位数统计
WIN = 5                # 均质性评估窗口
EM_PER_CLASS = 10      # 每类保留的端元条数
CLASSES = ['人工表面', '水体', '河漫滩', '耕地']

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1. 波段列表（与 Step1 的 os.listdir 字典序一致） ----------
tifs = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith('.tif'))
print(f'波段数: {len(tifs)}')
print(f'前 12 个: {tifs[:12]}')

import rasterio
with rasterio.open(os.path.join(IMG_DIR, tifs[0])) as s:
    H, W = s.height, s.width
    transform, crs = s.transform, s.crs
print(f'影像尺寸: {W} x {H} | CRS: {crs}')

# ---------- 2. 读入全影像立方体 ----------
cube = np.empty((len(tifs), H, W), np.float32)
for i, f in enumerate(tifs):
    with rasterio.open(os.path.join(IMG_DIR, f)) as s:
        cube[i] = s.read(1)
    if (i + 1) % 50 == 0:
        print(f'  已读 {i+1}/{len(tifs)} 波段')
print(f'立方体: {cube.shape}  {cube.nbytes/1024**2:.0f} MB')

# ---------- 3. 每波段分位数阈值（复现 Step1 规则） ----------
qs = [(k + 1) / N_CLASSES for k in range(N_CLASSES - 1)]
cube_thr = np.empty((len(tifs), N_CLASSES - 1), np.float32)
for i in range(len(tifs)):
    v = cube[i].ravel()
    if EXCLUDE_ZERO:
        v = v[v > 0]
    else:
        v = v[~np.isnan(v)]
    cube_thr[i] = np.quantile(v, qs)
print(f'\n离散级数: {N_CLASSES} 级（{N_CLASSES - 1} 个分位阈值/波段）')
print(f'分位数阈值 shape: {cube_thr.shape}')
print(f'示例 [B1 20210101]: {cube_thr[0]}')
print(f'示例 [NDVI 20210101]: {cube_thr[7]}')

# 背景等级占比自查（0 值应落在等级 1）
bg = (cube <= cube_thr[:, 0][:, None, None]).mean()
print(f'全影像落在等级1的像元比例: {bg*100:.2f}%（<5级时背景集中在等级1）')


def discretize(spec, thr):
    """按 Step1 规则离散化。spec: (..., nb) ; thr: (nb, n_thr) -> token 1..N_CLASSES"""
    tok = np.ones(spec.shape, np.int8)
    for k in range(thr.shape[1]):
        tok += (spec > thr[:, k]).astype(np.int8)
    return tok


# ---------- 4. 样本点读取与光谱提取 ----------
def rowcol(xs, ys):
    """投影坐标 -> 行列号（floor 语义，与 process02 的 rowcol 一致）"""
    cols = np.floor((xs - transform.c) / transform.a).astype(int)
    rows = np.floor((transform.f - ys) / (-transform.e)).astype(int)
    return rows, cols


out = {'cube_thr': cube_thr, 'band_names': np.array(tifs),
       'n_classes': np.array([N_CLASSES], np.int32)}

print('\n' + '=' * 74)
print(f'{"类别":<8}{"样本点":>8}{"窗口有效":>10}{"端元":>8}   均质性得分(CV)')
print('=' * 74)
for cls in CLASSES:
    gdf = pyogrio.read_dataframe(os.path.join(SHP_DIR, cls + '.shp'))
    xs = np.array([g.x for g in gdf.geometry])
    ys = np.array([g.y for g in gdf.geometry])
    rows, cols = rowcol(xs, ys)

    # 剔除越界点
    ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    rows, cols, xs, ys = rows[ok], cols[ok], xs[ok], ys[ok]

    spec = cube[:, rows, cols].T                        # (n, 200)
    out[f'sample_{cls}'] = spec.astype(np.float32)
    out[f'sample_rc_{cls}'] = np.stack([rows, cols], axis=1)
    out[f'sample_xy_{cls}'] = np.stack([xs, ys], axis=1)

    # 5x5 窗口均质性：各波段 CV 的中位数
    half = WIN // 2
    cv_list, valid = [], []
    for r, c in zip(rows, cols):
        r0, r1 = max(0, r - half), min(H, r + half + 1)
        c0, c1 = max(0, c - half), min(W, c + half + 1)
        win = cube[:, r0:r1, c0:c1]                      # (200, h, w)
        mu = win.mean(axis=(1, 2))
        sd = win.std(axis=(1, 2))
        cv = sd / (np.abs(mu) + 1e-6)
        cv_list.append(np.median(cv))
        valid.append(win.shape[1] * win.shape[2] == WIN * WIN)
    cv_arr = np.array(cv_list)
    valid = np.array(valid)

    # 只用窗口完整的点，取 CV 最小的 EM_PER_CLASS 个
    idx = np.where(valid)[0]
    order = idx[np.argsort(cv_arr[idx])]
    em_idx = order[:EM_PER_CLASS]

    out[f'em_{cls}'] = spec[em_idx].astype(np.float32)
    out[f'em_cv_{cls}'] = cv_arr[em_idx].astype(np.float32)
    out[f'em_rc_{cls}'] = np.stack([rows[em_idx], cols[em_idx]], axis=1)

    print(f'{cls:<8}{len(xs):>8}{valid.sum():>10}{len(em_idx):>8}   '
          f'{cv_arr[em_idx].min():.4f} ~ {cv_arr[em_idx].max():.4f} (中位 {np.median(cv_arr[em_idx]):.4f})')

np.savez_compressed(OUT_NPZ, **out)
print('=' * 74)
print(f'\n已保存: {OUT_NPZ}  ({os.path.getsize(OUT_NPZ)/1024**2:.1f} MB)')

# ---------- 5. 端元可分性速查 ----------
print('\n端元光谱自查（使用 B1@20210101 / B5@20210101 两列示意）:')
print(f'{"类别":<8}{"B1均值":>12}{"B5均值":>12}')
for cls in CLASSES:
    em = out[f'em_{cls}']
    print(f'{cls:<8}{em[:, 0].mean():>12.1f}{em[:, 4].mean():>12.1f}')
