# -*- coding: utf-8 -*-
"""对比 30m 多光谱 与 15m 全色/子网格，展示混合像元分层的分辨率基础。"""
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

IMG = r'E:/指纹/124036中/图像'
OUT = r'E:/混合像元实验/分辨率对比_30m_vs_15m.png'


def stretch(a, lo=2, hi=98):
    v = a[a != 0]
    if v.size == 0:
        return np.zeros_like(a, np.float32)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((a - p1) / (p2 - p1 + 1e-9), 0, 1).astype(np.float32)


# 30m 假彩色 (NIR-R-G)
stack = []
for b in ['B5', 'B4', 'B3']:
    with rasterio.open(f'{IMG}/20210322_{b}.tif') as s:
        stack.append(s.read(1).astype(np.float32))
rgb30 = np.dstack([stretch(x) for x in stack])

with rasterio.open(r'E:/混合像元实验/PAN_15m.tif') as s:
    pan = s.read(1).astype(np.float32)
pan[rgb30[:, :, 0].repeat(2, 0).repeat(2, 1) == 0] = 0
rgb15 = np.dstack([stretch(pan)] * 3)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第一行：全景
axes[0, 0].imshow(rgb30)
axes[0, 0].set_title('30m 多光谱 假彩色 (NIR-R-G)\n输入数据 · 657×642', fontsize=12)
axes[0, 1].imshow(rgb15)
axes[0, 1].set_title('15m 全色 B8\n参考数据 · 1314×1284', fontsize=12)

# 第三列：30m 网格叠加示意
axes[0, 2].imshow(rgb30)
h, w = rgb30.shape[:2]
for x in np.arange(0, w, 10):
    axes[0, 2].axvline(x, color='w', lw=0.23, alpha=0.65)
for y in np.arange(0, h, 10):
    axes[0, 2].axhline(y, color='w', lw=0.23, alpha=0.65)
axes[0, 2].set_title('30m 像元网格 (每 10 格画线)\n后续每个像元拆成 2×2 子像元', fontsize=12)

# 第二行：局部放大
r, c, n = 300, 300, 45
sub30 = rgb30[r:r + n, c:c + n]
sub15 = rgb15[r * 2:r * 2 + n * 2, c * 2:c * 2 + n * 2]
axes[1, 0].imshow(sub30, interpolation='nearest')
axes[1, 0].set_title(f'局部放大 30m ({n}×{n} 像元)', fontsize=12)
axes[1, 1].imshow(sub15, interpolation='nearest')
axes[1, 1].set_title(f'同范围 15m ({n*2}×{n*2} 子像元)', fontsize=12)

# 2x2 对应关系示意
axes[1, 2].imshow(sub30, interpolation='nearest')
for i in range(n + 1):
    axes[1, 2].axvline(i - .5, color='w', lw=0.7, alpha=0.8)
    axes[1, 2].axhline(i - .5, color='w', lw=0.7, alpha=0.8)
for i in range(n):
    for j in range(n):
        axes[1, 2].plot(j, i, marker='+', color='#00e5ff', ms=3.4, mew=0.85)
axes[1, 2].set_title('1 个 30m 像元 = 4 个 15m 子像元\n(十字为中心，四象限各 1 子像元)', fontsize=12)

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#999')
        sp.set_linewidth(0.8)

fig.suptitle('Landsat 8 全色 B8 (15m) 作为混合像元分层的参考数据 · 2021-03-22 · 云量 0.05%',
             fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print('已保存:', OUT)
