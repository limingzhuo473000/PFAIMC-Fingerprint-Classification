"""
步骤4: ENVI进阶 - 核心处理逻辑

从 04envi进阶.py 提取封装
"""

import os
import csv
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd


# ============================================================
# 辅助函数
# ============================================================

def get_image_count_from_folder(folder_path: str):
    """从文件夹中统计影像图数量"""
    supported_extensions = ['.tif', '.tiff', '.img']
    image_files = []
    for ext in supported_extensions:
        image_files.extend(glob(os.path.join(folder_path, f'*{ext}')))
        image_files.extend(glob(os.path.join(folder_path, f'*{ext.upper()}')))
    image_files = sorted(set(image_files))
    return len(image_files), image_files


def process_csv_pandas(input_file: str, output_file: str, target_length: int, max_sequence_length: int = None):
    """Pandas库处理CSV文件"""
    try:
        if not os.path.exists(input_file):
            return False

        df = pd.read_csv(input_file, dtype=str, header=None, keep_default_na=False)

        def replace_target_cell(cell):
            if cell.strip() == "":
                return '0'
            if cell.strip() == "-1":
                return '-1'
            cell_parts = cell.strip().split()
            # 步骤3输出格式：
            #   - 匹配成功：单元格被替换为单个替换值（如 "1" / "5"）
            #   - 未匹配：仍保留原始空格分隔序列（如 "3 5 2 7 1 4 6"）
            # 因此，只要单元格包含多个部分（有空格），即视为未匹配序列 → 赋 0；
            # 单值单元格保留，可能是步骤3的替换值，也可能是单波段原始值。
            # 注意：旧逻辑用 len(parts) == target_length 判断，当 target_length=1 时
            # 会把所有单值替换值也误判为未匹配序列并覆写成 0，导致重分类后全为未分类。
            if len(cell_parts) > 1:
                return '0'
            return cell

        df_processed = df.apply(lambda row: row.apply(replace_target_cell), axis=1)

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df_processed.to_csv(output_file, index=False, header=False, encoding='utf-8')
        return True

    except Exception as e:
        print(f"CSV处理出错: {e}")
        return False


def csv_to_txt(input_csv: str, output_txt: str):
    """将CSV文件转换为TXT格式"""
    try:
        if not os.path.exists(input_csv):
            return False

        try:
            with open(input_csv, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                data = list(reader)
        except UnicodeDecodeError:
            with open(input_csv, 'r', encoding='gbk') as csvfile:
                reader = csv.reader(csvfile)
                data = list(reader)

        output_dir = os.path.dirname(output_txt)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_txt, 'w', encoding='utf-8') as txtfile:
            for row in data:
                cleaned_row = []
                for cell in row:
                    cell = cell.strip()
                    if not cell:
                        cleaned_row.append('0')
                    elif ' ' in cell:
                        first_val = cell.split()[0]
                        cleaned_row.append(first_val)
                    else:
                        cleaned_row.append(cell)
                txtfile.write(' '.join(cleaned_row) + '\n')

        return True

    except Exception as e:
        print(f"CSV转TXT出错: {e}")
        return False


def txt_to_envi_with_metadata(txt_file_path: str, output_file_path: str, ref_tif_path: str = None):
    """将TXT文件转换为ENVI格式"""
    try:
        if not os.path.exists(txt_file_path):
            return False

        import rasterio

        crs = None
        transform = None
        ref_width = 1000
        ref_height = 1000

        if ref_tif_path and os.path.exists(ref_tif_path):
            try:
                with rasterio.open(ref_tif_path) as src_ref:
                    ref_width = src_ref.width
                    ref_height = src_ref.height
                    transform = src_ref.transform
                    crs = src_ref.crs
            except Exception:
                pass

        # 读取TXT文件
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(txt_file_path, 'r', encoding='gbk') as f:
                lines = f.readlines()

        txt_rows = []
        for line in lines:
            line = line.strip()
            if line:
                numeric_vals = []
                for val in line.split():
                    try:
                        numeric_vals.append(float(val))
                    except ValueError:
                        numeric_vals.append(0.0)
                if numeric_vals:
                    txt_rows.append(numeric_vals)

        if not txt_rows:
            return False

        output_height = len(txt_rows)
        output_width = max(len(row) for row in txt_rows) if txt_rows else 0

        emit_geo_anyway = (ref_tif_path and os.path.exists(ref_tif_path) and
                          (output_height != ref_height or output_width != ref_width))

        data = np.full((output_height, output_width), -9999.0, dtype=np.float32)

        for i in range(output_height):
            row_data = txt_rows[i]
            actual_len = len(row_data)
            if actual_len >= output_width:
                data[i, :] = row_data[:output_width]
            else:
                data[i, :actual_len] = row_data
                data[i, actual_len:] = -9999.0

        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 保存ENVI格式
        with rasterio.open(
            output_file_path,
            'w',
            driver='ENVI',
            height=output_height,
            width=output_width,
            count=1,
            dtype=data.dtype,
            crs=crs if not emit_geo_anyway else None,
            transform=transform if not emit_geo_anyway else None,
            nodata=-9999
        ) as dst:
            dst.write(data, 1)

        # 创建HDR文件
        hdr_file = output_file_path + '.hdr'
        with open(hdr_file, 'w', encoding='utf-8') as f:
            f.write('ENVI\n')
            f.write(f'description = {{Remote sensing classification result}}\n')
            f.write(f'samples = {output_width}\n')
            f.write(f'lines = {output_height}\n')
            f.write(f'bands = 1\n')
            f.write('header offset = 0\n')
            f.write('file type = ENVI Standard\n')
            f.write('data type = 4\n')
            f.write('interleave = bsq\n')
            f.write('byte order = 0\n')

            if crs and transform and not emit_geo_anyway:
                x_start = transform[2]
                y_start = transform[5]
                pixel_width = transform[0]
                pixel_height = abs(transform[4])
                f.write(f'x start = {x_start}\n')
                f.write(f'y start = {y_start}\n')
                f.write(f'map info = {{UTM, 1, 1, {x_start}, {y_start}, {pixel_width}, {pixel_height}, North, 0}}\n')
                f.write(f'coordinate system string = {crs}\n')

            f.write('data ignore value = -9999\n')

        return True

    except Exception as e:
        print(f"TXT转ENVI失败: {e}")
        return False


# ============================================================
# 主函数
# ============================================================

def run_step4(
    task_manager,
    task_id: str,
    input_csv: str,
    images_folder: str,
    output_dir: str,
    add_geo: bool = True,
    max_sequence_length: int = None,
) -> dict:
    """
    执行步骤4处理
    """
    def update_progress(progress: float, message: str):
        task_manager.update_task(task_id, progress=progress, message=message)

    try:
        if not os.path.exists(input_csv):
            return {"success": False, "error": f"输入文件不存在: {input_csv}"}

        if not os.path.exists(images_folder):
            return {"success": False, "error": f"影像文件夹不存在: {images_folder}"}

        os.makedirs(output_dir, exist_ok=True)

        # 获取影像图数量
        update_progress(5, "统计影像图数量...")
        image_count, image_files = get_image_count_from_folder(images_folder)
        if image_count == 0:
            return {"success": False, "error": "未找到影像文件"}

        update_progress(10, f"检测到序列长度: {image_count}")

        # 确定参考TIF文件
        ref_tif = None
        if add_geo:
            tif_files = [f for f in image_files if f.lower().endswith(('.tif', '.tiff'))]
            if tif_files:
                ref_tif = tif_files[0]

        # 生成输出文件路径
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        processed_csv = os.path.join(output_dir, f"{base_name}_processed.csv")
        output_txt = os.path.join(output_dir, f"{base_name}_converted.txt")
        output_envi = os.path.join(output_dir, f"{base_name}_result.dat")

        # 步骤1: CSV处理
        update_progress(20, "处理CSV文件...")
        if not process_csv_pandas(input_csv, processed_csv, image_count, max_sequence_length):
            return {"success": False, "error": "CSV处理失败"}

        # 步骤2: CSV转TXT
        update_progress(50, "转换CSV为TXT...")
        if not csv_to_txt(processed_csv, output_txt):
            return {"success": False, "error": "CSV转TXT失败"}

        # 步骤3: TXT转ENVI
        update_progress(75, "转换TXT为ENVI格式...")
        if not txt_to_envi_with_metadata(output_txt, output_envi, ref_tif):
            return {"success": False, "error": "TXT转ENVI失败"}

        update_progress(100, "处理完成!")

        return {
            "success": True,
            "processed_csv": processed_csv,
            "output_txt": output_txt,
            "output_envi": output_envi,
            "hdr_file": output_envi + '.hdr',
            "valid_pixels_ratio": 0.0,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
