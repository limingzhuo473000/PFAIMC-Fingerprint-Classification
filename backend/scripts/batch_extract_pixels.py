"""
批量提取多光谱影像像元值
功能：从多个时相的多波段影像中提取样本点的光谱值，输出为 CSV/Excel
"""

import os
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol


def extract_by_category(shp_folder: str, tif_folder: str, output_dir: str):
    """
    按类别批量提取样本点的光谱值，每个类别生成一个 CSV
    
    参数:
        shp_folder: 包含多个类别 SHP 文件的文件夹
        tif_folder: 存放 .tif 影像的文件夹
        output_dir: 输出 CSV 的文件夹
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 扫描所有 SHP 文件
    shp_files = sorted(glob.glob(os.path.join(shp_folder, "*.shp")))
    if not shp_files:
        raise FileNotFoundError(f"未找到 .shp 文件: {shp_folder}")
    
    print(f"发现 {len(shp_files)} 个类别文件")
    
    # 定义类别映射（根据文件名或内部字段）
    category_map = {
        '耕地': ['耕地', 'cultivated', 'farmland'],
        '河漫滩': ['河漫滩', 'floodplain'],
        '人工表面': ['人工表面', 'artificial', 'impervious', 'built'],
        '水体': ['水体', 'water']
    }
    
    for shp_path in shp_files:
        basename = os.path.splitext(os.path.basename(shp_path))[0]
        print(f"\n{'='*60}")
        print(f"处理类别: {basename}")
        print(f"{'='*60}")
        
        try:
            # 读取样本点
            gdf = gpd.read_file(shp_path)
            
            if len(gdf) == 0:
                print(f"   跳过: 空文件")
                continue
            
            if 'Point' not in gdf.geometry.iloc[0].geom_type:
                print(f"   跳过: 非点要素 ({gdf.geometry.iloc[0].geom_type})")
                continue
            
            points = list(zip(gdf.geometry.x, gdf.geometry.y))
            point_ids = gdf.index.tolist()
            
            # 扫描所有 TIFF 文件并按日期分组
            tif_files = sorted(glob.glob(os.path.join(tif_folder, "*.tif")))
            
            # 按日期分组（支持 B1-B7 和指数文件）
            date_features = {}
            for tif in tif_files:
                basename_tif = os.path.basename(tif)
                
                # 匹配波段文件: 20210101_B1.tif
                match_band = re.match(r'(\d{8})_B(\d+)\.tif', basename_tif)
                # 匹配指数文件: 20210101_NDVI.tif
                match_idx = re.match(r'(\d{8})_(NDVI|NDBI|NDWI)\.tif', basename_tif)
                
                if match_band:
                    date_str = match_band.group(1)
                    feature_name = f'B{match_band.group(2)}'
                elif match_idx:
                    date_str = match_idx.group(1)
                    feature_name = match_idx.group(2)
                else:
                    continue
                
                if date_str not in date_features:
                    date_features[date_str] = {}
                date_features[date_str][feature_name] = tif
            
            print(f"   发现 {len(date_features)} 个时相")
            
            # 逐时相提取像元值
            all_data = []
            
            for date_str in sorted(date_features.keys()):
                features_dict = date_features[date_str]
                
                # 读取第一个文件获取 transform 和 shape
                first_feature_path = list(features_dict.values())[0]
                with rasterio.open(first_feature_path) as src:
                    transform = src.transform
                    height = src.height
                    width = src.width
                
                # 对每个点提取所有特征的值
                for idx, (x, y) in enumerate(points):
                    r, c = rowcol(transform, x, y, op=np.round)
                    r, c = int(r), int(c)
                    
                    if 0 <= r < height and 0 <= c < width:
                        row_data = {'point_id': point_ids[idx], 'date': date_str}
                        
                        # 提取该时相所有特征的值
                        for feat_name, feat_path in sorted(features_dict.items()):
                            with rasterio.open(feat_path) as src:
                                data = src.read(1)
                                value = data[r, c]
                                row_data[feat_name] = float(value) if not np.isnan(value) else None
                        
                        all_data.append(row_data)
            
            if not all_data:
                print(f"   警告: 未提取到有效数据")
                continue
            
            # 保存结果
            df = pd.DataFrame(all_data)
            
            # 重塑为宽表格式
            pivot_df = df.pivot(index='point_id', columns='date')
            pivot_df.columns = [f"{date}_{col}" for col, date in pivot_df.columns]
            pivot_df.reset_index(inplace=True)
            
            # 合并原始属性
            result = gdf.merge(pivot_df, left_on=gdf.index, right_on='point_id', how='left')
            result.drop(columns=['point_id', 'key_0'], errors='ignore', inplace=True)
            
            # 导出 CSV
            output_csv = os.path.join(output_dir, f"{basename}.csv")
            result.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"   ✓ 完成! {len(result)} 个点, {len(result.columns)-1} 个特征列")
            print(f"   输出: {output_csv}")
            
        except Exception as e:
            print(f"   ✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    # ========== 配置区域 ==========
    SHP_FOLDER = r"E:\指纹\124036中\样本\实验样本"  # 包含多个类别 SHP 的文件夹
    TIF_FOLDER = r"E:\指纹\研究区"  # 影像文件夹
    OUTPUT_DIR = r"E:\指纹\124036中\tiqu"  # 输出文件夹
    # ================================
    
    try:
        extract_by_category(SHP_FOLDER, TIF_FOLDER, OUTPUT_DIR)
        print("\n" + "="*60)
        print("全部提取完成!")
        print("="*60)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
