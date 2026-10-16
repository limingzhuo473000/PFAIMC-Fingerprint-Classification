# -*- coding: utf-8 -*-
"""
准备「混合像元分等级精度评价」所需的高分辨率参考数据。

作用：
  把 Landsat 8 B8 全色(15m) 重采样对齐到研究区 30m 网格的 2x2 子网格，
  使每个 30m 像元精确对应 4 个 15m 子像元 -> 可统计亚像元类别组成 -> 得到混合度标签。

用法：
  python prep_pan_reference.py <参考30m影像> <Landsat下载目录> <输出目录> [文件前缀]

示例：
  python prep_pan_reference.py "E:/指纹/124036中/图像/20210322_B1.tif" ^
      "E:/高程和坡向/LC81240362021081LGN00" "E:/混合像元实验"
"""
import os
import sys
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from affine import Affine


def find_band(src_dir, prefix, band):
    """在下载目录里按后缀找波段文件（兼容 .TIF/.tif 命名差异）"""
    for ext in ('.TIF', '.tif'):
        p = os.path.join(src_dir, f'{prefix}{band}{ext}')
        if os.path.exists(p):
            return p
    # 退而求其次：模糊匹配
    for f in os.listdir(src_dir):
        stem = os.path.splitext(f)[0]
        if stem.endswith('_' + band) or stem.endswith(band):
            return os.path.join(src_dir, f)
    raise FileNotFoundError(f'未找到波段 {band}，目录: {src_dir}')


def guess_prefix(src_dir):
    """从目录中自动识别文件名前缀（取 B8 文件）"""
    for f in os.listdir(src_dir):
        stem = os.path.splitext(f)[0]
        if stem.endswith('B8'):
            return stem[:-2]
    raise FileNotFoundError('目录中未找到 B8 全色文件')


def main():
    ref_path = sys.argv[1] if len(sys.argv) > 1 else r'E:/指纹/124036中/图像/20210322_B1.tif'
    src_dir = sys.argv[2] if len(sys.argv) > 2 else r'E:/高程和坡向/LC81240362021081LGN00'
    out_dir = sys.argv[3] if len(sys.argv) > 3 else r'E:/混合像元实验'
    prefix = sys.argv[4] if len(sys.argv) > 4 else guess_prefix(src_dir)

    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(ref_path) as r:
        ref_crs, ref_tf = r.crs, r.transform
        W, H = r.width, r.height
        left, bottom, right, top = r.bounds

    # 目标 15m 网格：与 30m 网格同原点，像元减半 —— 每个 30m 像元恰好覆盖 2x2 个目标像元
    dst_tf = Affine(15.0, 0.0, ref_tf.c, 0.0, -15.0, ref_tf.f)
    dW, dH = W * 2, H * 2

    print('=' * 62)
    print(f'参考网格 : {ref_path}')
    print(f'           {W} x {H} @30m   CRS={ref_crs}')
    print(f'目标网格 : {dW} x {dH} @15m  (2x2 子像元)')
    print(f'源数据   : {src_dir}')
    print(f'前缀     : {prefix}')
    print('=' * 62)

    def read_reproj(band, resampling=Resampling.bilinear):
        """读取研究区窗口 -> 重投影到 15m 目标网格"""
        p = find_band(src_dir, prefix, band)
        with rasterio.open(p) as src:
            win = rasterio.windows.from_bounds(left, bottom, right, top, src.transform)
            win = Window(int(np.floor(win.col_off)) - 2, int(np.floor(win.row_off)) - 2,
                         int(np.ceil(win.width)) + 4, int(np.ceil(win.height)) + 4)
            win = win.intersection(Window(0, 0, src.width, src.height))
            raw = src.read(1, window=win).astype(np.float32)
            stf, scrs = src.window_transform(win), src.crs
        dst = np.zeros((dH, dW), np.float32)
        reproject(raw, dst, src_transform=stf, src_crs=scrs,
                  dst_transform=dst_tf, dst_crs=ref_crs, resampling=resampling)
        return dst

    pan = read_reproj('B8')
    print(f'\n[1/2] B8 全色 -> 15m 网格  数值 {pan.min():.0f} ~ {pan.max():.0f}  '
          f'均值 {pan.mean():.1f}')

    # 保存 15m 全色（L1TP DN 值，保留原始整型便于后续做纹理/边缘）
    prof = dict(driver='GTiff', height=dH, width=dW, count=1,
                dtype='uint16', crs=ref_crs, transform=dst_tf,
                nodata=0, compress='lzw', tiled=True)
    pan_out = os.path.join(out_dir, 'PAN_15m.tif')
    with rasterio.open(pan_out, 'w', **prof) as dst:
        dst.write(np.clip(pan, 0, 65535).astype(np.uint16), 1)
    print(f'      -> {pan_out}')

    # 同时输出 30m 网格中心位置的 15m 子像元索引映射说明
    meta_txt = os.path.join(out_dir, 'READ_ME_网格关系.txt')
    with open(meta_txt, 'w', encoding='utf-8') as f:
        f.write('30m 网格与 15m 子网格对应关系\n')
        f.write('=' * 50 + '\n')
        f.write(f'30m 网格 : origin=({ref_tf.c}, {ref_tf.f})  {W} x {H}  res=30\n')
        f.write(f'15m 网格 : origin=({dst_tf.c}, {dst_tf.f})  {dW} x {dH}  res=15\n\n')
        f.write('对任意 30m 像元 (row, col)，其 4 个 15m 子像元为：\n')
        f.write('  (row*2 + r, col*2 + c)   r, c ∈ {0, 1}\n')
        f.write('  子像元 (r=0,c=0) 对应 30m 像元的左上象限，其余按行列顺延。\n\n')
        f.write('混合度 = 优势类子像元数 / 4\n')
        f.write('  4/4 -> 纯像元   3/4 -> 轻度混合   2/4 -> 中度混合   1/4 -> 重度混合\n')
    print(f'      -> {meta_txt}')
    print('\n完成。下一步：对 15m 数据分类 -> 按 2x2 聚合得到混合度标签。')


if __name__ == '__main__':
    main()
