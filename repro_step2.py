# -*- coding: utf-8 -*-
"""复现 step2 SHP 提取失败：直接调用 process02 的提取函数，输出真实诊断"""
import os, sys, traceback

BACKEND = r"E:\预处理\预处理\网页端 - 副本(全多进程+计时)\网页端 - 副本(全多进程+计时)\backend"
sys.path.insert(0, BACKEND)

SHP_DIR = r"E:\指纹\124036中\样本\实验样本_7月版"
TIF_DIR = r"E:\指纹\124036英文测试\五指数"
OUT = r"E:\output\temp_sample_processing\repro_20260918"

os.makedirs(OUT, exist_ok=True)

tif_files = sorted(f for f in os.listdir(TIF_DIR) if f.lower().endswith((".tif", ".tiff")))
print(f"TIF数量: {len(tif_files)}, 首个: {tif_files[0] if tif_files else '无'}", flush=True)
raster = os.path.join(TIF_DIR, tif_files[0])

# 先在主进程里单独测试 raster 与 SHP 读取
try:
    import rasterio
    with rasterio.open(raster) as src:
        print(f"栅格OK: {src.width}x{src.height}, CRS={src.crs}, nodata={src.nodata}", flush=True)
except Exception:
    traceback.print_exc()

try:
    import geopandas as gpd
    shps = [f for f in os.listdir(SHP_DIR) if f.endswith(".shp")]
    print(f"SHP文件: {shps}", flush=True)
    for s in shps:
        g = gpd.read_file(os.path.join(SHP_DIR, s))
        print(f"  {s}: {len(g)}要素, CRS={g.crs}", flush=True)
except Exception:
    traceback.print_exc()

# 再走真实的多进程路径
try:
    from scripts.process02 import extract_points_with_pixel_location_batch
    ok = extract_points_with_pixel_location_batch(SHP_DIR, raster, OUT)
    print(f"批量提取返回: {ok}", flush=True)
    print(f"输出目录内容: {os.listdir(OUT)}", flush=True)
except Exception:
    traceback.print_exc()
