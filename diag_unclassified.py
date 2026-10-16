# -*- coding: utf-8 -*-
"""诊断恢复实验中2个未分类验证像素的匹配情况"""
import sys, os
import numpy as np
import rasterio
from rasterio import features as _features
import geopandas as gpd

ROOT = r"E:\指纹\124036中"
MERGED = os.path.join(ROOT, r"30\step1\merged.csv")
STEP3_OUT = os.path.join(ROOT, r"恢复实验\round_17\step3\out1_140.csv")
STEP2 = os.path.join(ROOT, r"恢复实验\step2")
VAL = os.path.join(ROOT, r"样本\评价样本")
IMG = os.path.join(ROOT, r"图像")

# 1. 读取参考影像的仿射变换与尺寸
tif0 = None
for f in os.listdir(IMG):
    if f.endswith(".tif"):
        tif0 = os.path.join(IMG, f); break
with rasterio.open(tif0) as src:
    transform = src.transform
    w, h = src.width, src.height
print(f"参考影像: {os.path.basename(tif0)}  {w}x{h}")

# 2. 读取step3输出（每格: 序列/替换值/-1）
with open(STEP3_OUT, encoding="utf-8") as f:
    lines = [l.rstrip("\n") for l in f]
grid = [l.split(",") for l in lines if l.strip()]
print(f"step3网格: {len(grid)}行 x {len(grid[0])}列")

# 3. 读取各类模板序列
templates = {}  # class -> list of seq(list of int)
for name in ["人工表面", "水体", "河漫滩", "耕地"]:
    p = os.path.join(STEP2, f"{name}.csv")
    seqs = []
    with open(p, encoding="utf-8-sig") as f:
        header = f.readline()
        for line in f:
            parts = line.split(",")
            # 最后一列是提取值
            cell = parts[-1].strip().strip('"')
            vals = cell.split()
            if len(vals) >= 100:
                seqs.append([int(float(v)) for v in vals])
    templates[name] = seqs
    print(f"{name}: {len(seqs)}条模板序列, 长度={len(seqs[0]) if seqs else 0}")

def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

# 4. 遍历评价样本SHP，找未分类像素
for name in ["人工表面", "水体", "河漫滩", "耕地"]:
    shp = os.path.join(VAL, f"{name}.shp")
    gdf = gpd.read_file(shp)
    if gdf.crs is not None and str(gdf.crs) != str(src.crs):
        pass  # 假设同投影
    unclassified = []
    all_pixels = 0
    for geom in gdf.geometry:
        mask = _features.geometry_mask([geom], out_shape=(h, w), transform=transform, invert=True)
        for r, c in zip(*np.where(mask)):
            all_pixels += 1
            cell = grid[r][c].strip()
            # 未匹配的格子仍是长序列（含空格）
            if " " in cell:
                seq = [int(float(v)) for v in cell.split()]
                unclassified.append((r, c, seq))
    print(f"\n{name}: 验证像素共{all_pixels}, 其中未匹配(未分类) {len(unclassified)} 个")
    for (r, c, seq) in unclassified:
        print(f"  像素(row={r}, col={c}):")
        dists = {}
        for cls, seqs in templates.items():
            ds = [hamming(seq, t) for t in seqs]
            dists[cls] = min(ds)
        print(f"    与各类最小汉明距离: {dists}")
        mn = min(dists.values())
        tied = [k for k, v in dists.items() if v == mn]
        print(f"    最小距离={mn}, 最近类={tied}, {'★歧义(等距多类)→未分类' if len(tied)>1 else '唯一最近类→本应分类'}")
