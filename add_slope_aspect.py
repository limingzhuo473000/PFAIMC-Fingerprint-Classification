# -*- coding: utf-8 -*-
"""
将坡度(SLOPE)与坡向(ASPECT)作为新波段加入 E:/指纹/124036中/图像/

- 源数据: E:/高程和坡向/{slope,aspect}  (ArcInfo Binary Grid, EPSG:4326, float32)
- 目标网格: 与 Landsat 单波段 TIF 完全一致 (EPSG:32649, 657x642, 30m), nodata=0
- 重采样: slope 用双线性(连续场); aspect 用最近邻(角度 359/0 不能插值)
- 命名: SLOPE.tif / ASPECT.tif —— 字典序排在所有日期波段之后, 序列长度 +2
"""
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds

REF_TIF = r"E:/指纹/124036中/图像/20210101_B1.tif"
IMG_DIR = r"E:/指纹/124036中/图像"
DEM_DIR = r"E:/高程和坡向"

with rasterio.open(REF_TIF) as ref:
    dst_crs = ref.crs
    dst_transform = ref.transform
    dst_w, dst_h = ref.width, ref.height
    ref_profile = ref.profile.copy()
    ref_bounds_ll = transform_bounds(ref.crs, "EPSG:4326", *ref.bounds, densify_pts=21)
    print(f"参考网格: {dst_w}x{dst_h} @ {dst_crs}")
    print(f"场景经纬度范围: {ref_bounds_ll}")

# 统计 Landsat 参考波段的有效像元(非0), 用于对齐检查
with rasterio.open(REF_TIF) as ref:
    ref_data = ref.read(1)
ref_valid = np.count_nonzero(ref_data)
print(f"参考波段 B1 有效像元: {ref_valid} ({ref_valid / ref_data.size * 100:.2f}%)")

dst_profile = ref_profile
dst_profile.update(count=1, dtype="float32", nodata=0.0)

tasks = [
    (os.path.join(DEM_DIR, "slope"),  "SLOPE.tif",  Resampling.bilinear),
    (os.path.join(DEM_DIR, "aspect"), "ASPECT.tif", Resampling.nearest),
]

for src_path, out_name, method in tasks:
    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_nodata = src.nodata if src.nodata is not None else -3.4e38
        print(f"\n--- {out_name} ---")
        print(f"源: {src.width}x{src.height} {src.crs}, nodata={src_nodata}")

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

    # 数值体检前先做物理范围截断(清除重采样边界混入的 nodata 巨值)
    if "SLOPE" in out_name:
        bad = ~np.isfinite(dst) | (dst < 0) | (dst > 90)   # 坡度物理范围 0~90
    else:
        bad = ~np.isfinite(dst) | (dst < -1) | (dst > 360)  # 坡向范围 -1(平地)~360
    dst[bad] = 0.0

    n_nan = int(np.isnan(dst).sum())
    n_huge = int((np.abs(dst) > 1e6).sum())
    valid_mask = dst != 0.0
    n_valid = int(valid_mask.sum())

    out_path = os.path.join(IMG_DIR, out_name)
    with rasterio.open(out_path, "w", **dst_profile) as dst_file:
        dst_file.write(dst, 1)
        dst_file.update_tags(DESCRIPTION=out_name.replace(".tif", ""))

    vv = dst[valid_mask]
    print(f"输出: {out_path}")
    print(f"  有效像元: {n_valid} ({n_valid / dst.size * 100:.2f}%)  [B1: {ref_valid}]")
    print(f"  范围: {vv.min():.4f} ~ {vv.max():.4f}  均值: {vv.mean():.4f}")
    print(f"  NaN: {n_nan}  巨值残留: {n_huge}")

    # 与 Landsat 有效区一致性
    diff = int(np.logical_xor(valid_mask, ref_data != 0).sum())
    print(f"  与 B1 有效区不一致像元: {diff}")

print("\n完成。")
