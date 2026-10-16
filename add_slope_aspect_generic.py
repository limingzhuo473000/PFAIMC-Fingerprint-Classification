# -*- coding: utf-8 -*-
"""
通用版：把坡度/坡向作为两个新波段并入任意图像目录

用法:
    python add_slope_aspect_generic.py <图像目录> <坡度坡向源目录>

本次执行:
    图像目录 = E:/裁剪后122          (200 个 TIF, 20期x10, EPSG:32650, 285x285)
    源目录   = E:/高程和坡向/122042坡度   (slope/ aspect/ 两个 ArcInfo Grid, EPSG:4326)

- 参考网格: 图像目录中字典序第一个 *_B1.tif
- 重采样:   SLOPE 双线性(连续场); ASPECT 最近邻(角度不可插值)
- nodata:   输出统一 0.0(与 Landsat 波段一致); 源 nodata(-3.4e38)及越界值置 0
- 截断:     SLOPE 物理范围 0~90; ASPECT -1~360(ArcGIS 平地=-1)
- 输出:     <图像目录>/SLOPE.tif, ASPECT.tif  (float32, 排在所有日期波段之后)
"""
import os
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds

IMG_DIR = sys.argv[1] if len(sys.argv) > 1 else r"E:/裁剪后122"
DEM_DIR = sys.argv[2] if len(sys.argv) > 2 else r"E:/高程和坡向/122042坡度"

# ---------- 1. 参考网格 ----------
tifs = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(".tif"))
ref_name = next((f for f in tifs if f.upper().endswith("_B1.TIF")), None)
if ref_name is None:
    print(f"!! {IMG_DIR} 中未找到 *_B1.tif")
    sys.exit(1)
dup = [f for f in tifs if f.upper() in ("SLOPE.TIF", "ASPECT.TIF")]
if dup:
    print(f"!! 已存在 {dup}，如需重做请先删除")
    sys.exit(1)

REF_TIF = os.path.join(IMG_DIR, ref_name)
with rasterio.open(REF_TIF) as ref:
    dst_crs = ref.crs
    dst_transform = ref.transform
    dst_w, dst_h = ref.width, ref.height
    profile = ref.profile.copy()
    ref_data = ref.read(1)
ll = transform_bounds(ref.crs, "EPSG:4326", *ref.bounds, densify_pts=21)

print(f"图像目录: {IMG_DIR}  (TIF {len(tifs)} 个)")
print(f"参考网格: {ref_name} -> {dst_w}x{dst_h} @ {dst_crs}, dtype={ref.dtypes}, nodata={ref.nodata}")
print(f"场景经纬度: ({ll[0]:.5f}, {ll[1]:.5f}) ~ ({ll[2]:.5f}, {ll[3]:.5f})")
ref_valid = int(np.count_nonzero(ref_data))
print(f"参考 B1 有效像元: {ref_valid} ({ref_valid / ref_data.size * 100:.2f}%)")

profile.update(count=1, dtype="float32", nodata=0.0)

# ---------- 2. 逐波段转换 ----------
tasks = [
    (os.path.join(DEM_DIR, "slope"),  "SLOPE.tif",  Resampling.bilinear),
    (os.path.join(DEM_DIR, "aspect"), "ASPECT.tif", Resampling.nearest),
]

for src_path, out_name, method in tasks:
    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_nodata = src.nodata if src.nodata is not None else -3.4e38
        print(f"\n--- {out_name} ---")
        print(f"源: {src.width}x{src.height} {src.crs} nodata={src_nodata}")

        dst = np.full((dst_h, dst_w), 0.0, dtype="float32")
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=0.0,
            resampling=method,
        )

    # 物理范围截断: 清除重采样边界混入的 nodata 巨值/异常值
    if "SLOPE" in out_name:
        bad = ~np.isfinite(dst) | (dst < 0) | (dst > 90)
    else:
        bad = ~np.isfinite(dst) | (dst < -1) | (dst > 360)
    n_fixed = int(bad.sum())
    dst[bad] = 0.0

    out_path = os.path.join(IMG_DIR, out_name)
    with rasterio.open(out_path, "w", **profile) as dst_f:
        dst_f.write(dst, 1)

    # 体检
    m = dst != 0.0
    n_valid = int(m.sum())
    v = dst[m]
    only_b1 = int(np.logical_and(ref_data != 0, ~m).sum())
    only_dem = int(np.logical_and(m, ref_data == 0).sum())
    print(f"输出: {out_path}")
    print(f"  截断清除像元: {n_fixed} | 有效: {n_valid} ({n_valid / dst.size * 100:.2f}%) [B1: {ref_valid}]")
    print(f"  范围: {v.min():.4f} ~ {v.max():.4f}  均值: {v.mean():.4f}")
    print(f"  NaN: {int(np.isnan(dst).sum())}  巨值: {int((np.abs(dst) > 1e6).sum())}")
    print(f"  与B1有效区差异: B1有而它无={only_b1}, 它有而B1无={only_dem}")

print("\n完成。序列长度 200 -> 202, 需重跑 Step1 生成新 merged.csv。")
