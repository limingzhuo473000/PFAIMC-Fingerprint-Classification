"""
步骤5: 重分类优化 - 核心处理逻辑

从 05重分类优化.py 提取封装
"""

import os
import re
import colorsys
from pathlib import Path

import numpy as np


# ============================================================
# 辅助函数
# ============================================================

def read_envi_header(hdr_path: str):
    """从 ENVI .hdr 文件中读取所有参数"""
    if not os.path.exists(hdr_path):
        if not hdr_path.endswith('.hdr'):
            hdr_path_with_ext = hdr_path + '.hdr'
            if os.path.exists(hdr_path_with_ext):
                hdr_path = hdr_path_with_ext
            else:
                raise FileNotFoundError(f"找不到HDR文件: {hdr_path}")
        else:
            raise FileNotFoundError(f"找不到HDR文件: {hdr_path}")

    content = None
    for encoding in ['gbk', 'utf-8']:
        try:
            with open(hdr_path, 'r', encoding=encoding) as f:
                content = f.read()
            if content.strip():
                break
        except UnicodeDecodeError:
            continue

    if not content:
        raise ValueError(f"无法读取HDR文件: {hdr_path}")

    samples = int(re.search(r'samples\s*=\s*(\d+)', content, re.IGNORECASE).group(1))
    lines = int(re.search(r'lines\s*=\s*(\d+)', content, re.IGNORECASE).group(1))
    bands = int(re.search(r'bands\s*=\s*(\d+)', content, re.IGNORECASE).group(1))

    data_type_map = {
        '1': np.uint8, '2': np.int16, '3': np.int32, '4': np.float32,
        '5': np.float64, '6': np.complex64, '9': np.complex128,
        '12': np.uint16, '13': np.uint32, '14': np.int64, '15': np.uint64,
    }
    data_type_code = re.search(r'data type\s*=\s*(\d+)', content, re.IGNORECASE)
    dtype = data_type_map.get(data_type_code.group(1), np.int16) if data_type_code else np.int16

    geographic_info = {}
    geographic_patterns = [
        (r'map info\s*=\s*\{([^}]*)\}', 'map_info'),
        (r'coordinate system string\s*=\s*\{([^}]*)\}', 'coordinate_system'),
        (r'projection info\s*=\s*\{([^}]*)\}', 'projection_info'),
        (r'x start\s*=\s*([^\n]+)', 'x_start'),
        (r'y start\s*=\s*([^\n]+)', 'y_start'),
    ]

    for pattern, key in geographic_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            geographic_info[key] = match.group(1).strip()

    return samples, lines, bands, dtype, geographic_info


def reclassify_with_geospatial_info(
    dat_path: str,
    class_value_groups: list[list[int]],
    class_names: list[str],
) -> dict:
    """
    对 ENVI 格式的 .dat 文件进行重分类

    返回:
        dict: 包含 success, new_dat_path, new_hdr_path 等字段
    """
    # 1. 检查并修复 dat_path
    if not os.path.exists(dat_path):
        if not dat_path.endswith('.dat'):
            possible_dat = dat_path + '.dat'
            if os.path.exists(possible_dat):
                dat_path = possible_dat
            else:
                return {"success": False, "error": f"找不到DAT文件: {dat_path}"}

    # 2. 构建 hdr_path
    if dat_path.endswith('.dat'):
        hdr_path = dat_path[:-4] + '.hdr'
    else:
        hdr_path = dat_path + '.hdr'

    if not os.path.exists(hdr_path):
        return {"success": False, "error": f"找不到HDR文件: {hdr_path}"}

    # 3. 读取 HDR
    try:
        samples, lines, bands, dtype, geo_info = read_envi_header(hdr_path)
    except Exception as e:
        return {"success": False, "error": f"读取HDR失败: {e}"}

    # 4. 读取 DAT 数据
    try:
        data = np.fromfile(dat_path, dtype=dtype)
    except Exception as e:
        return {"success": False, "error": f"读取DAT失败: {e}"}

    # 5. 重塑数据
    total_elements = lines * samples
    if bands > 1:
        total_elements_with_bands = total_elements * bands
        if data.size != total_elements_with_bands:
            if data.size == total_elements:
                bands = 1
            else:
                return {"success": False, "error": f"数据大小不匹配: 期望 {total_elements_with_bands}, 实际 {data.size}"}
        data = data.reshape((bands, lines, samples))
        if bands > 1:
            data = data[0]
    else:
        if data.size != total_elements:
            return {"success": False, "error": f"数据大小不匹配: 期望 {total_elements}, 实际 {data.size}"}
        data = data.reshape((lines, samples))

    # 6. 重分类
    reclassified = np.zeros_like(data, dtype=np.uint8)

    # 边框处理
    border_mask = data == -1
    reclassified[border_mask] = 1
    border_count = np.sum(border_mask)

    # 根据数值组重分类（从类别2开始）
    classified_stats = []
    for i, value_group in enumerate(class_value_groups):
        for value in value_group:
            if value == -1 or value == 0:
                continue
            mask = (data == value) & (reclassified == 0)
            count = np.sum(mask)
            if count > 0:
                reclassified[mask] = i + 2
                classified_stats.append({
                    "class_name": class_names[i],
                    "value": value,
                    "count": int(count),
                })

    # 统计
    unclassified_count = np.sum((reclassified == 0) & (data != -1))
    border_count_final = np.sum(reclassified == 1)
    total_pixels = lines * samples
    classified_count = total_pixels - unclassified_count - border_count_final

    # 7. 保存 DAT
    base_name = os.path.splitext(dat_path)[0]
    if base_name.endswith('.dat'):
        base_name = base_name[:-4]
    new_dat_path = base_name + '_reclass.dat'

    try:
        reclassified.tofile(new_dat_path)
    except Exception as e:
        return {"success": False, "error": f"保存DAT失败: {e}"}

    # 8. 创建 HDR
    colors = [
        '128, 128, 128', '0, 0, 0', '0, 255, 0', '0, 0, 255',
        '255, 0, 0', '34, 139, 34', '255, 255, 0', '255, 0, 255',
        '0, 255, 255', '128, 0, 128', '255, 165, 0', '128, 128, 0',
    ]

    total_classes = len(class_value_groups) + 2

    header_content = f"""ENVI
description = {{Reclassification result - {os.path.basename(dat_path)}}}
samples = {samples}
lines = {lines}
bands = 1
header offset = 0
file type = ENVI Classification
data type = 1
interleave = bsq
byte order = 0
classes = {total_classes}
class names = {{Unclassified, Border, {", ".join(class_names)}}}
class lookup = {{"""

    for i in range(total_classes):
        color = colors[i] if i < len(colors) else f"{i * 30 % 255}, {i * 50 % 255}, {i * 70 % 255}"
        header_content += color
        if i < total_classes - 1:
            header_content += ", "

    header_content += "}\n"

    if geo_info:
        for key, value in geo_info.items():
            if key == 'map_info':
                header_content += f"map info = {{{value}}}\n"
            elif key == 'coordinate_system':
                header_content += f"coordinate system string = {{{value}}}\n"
            elif key == 'x_start':
                header_content += f"x start = {value}\n"
            elif key == 'y_start':
                header_content += f"y start = {value}\n"

    new_hdr_path = new_dat_path.replace('.dat', '.hdr')
    try:
        with open(new_hdr_path, 'w', encoding='gbk') as f:
            f.write(header_content)
    except Exception as e:
        return {"success": False, "error": f"保存HDR失败: {e}"}

    return {
        "success": True,
        "new_dat_path": new_dat_path,
        "new_hdr_path": new_hdr_path,
        "stats": {
            "total_pixels": int(total_pixels),
            "classified_pixels": int(classified_count),
            "border_pixels": int(border_count_final),
            "unclassified_pixels": int(unclassified_count),
        },
        "classified_details": classified_stats,
    }


# ============================================================
# 主函数
# ============================================================

def run_step5(
    task_manager,
    task_id: str,
    dat_path: str,
    classes: list[dict],
) -> dict:
    """
    执行步骤5处理

    参数:
        task_manager: 任务管理器实例
        task_id: 任务ID
        dat_path: ENVI .dat 文件路径
        classes: 地物类别定义 [{"name": str, "values": [int]}]
    """
    def update_progress(progress: float, message: str):
        task_manager.update_task(task_id, progress=progress, message=message)

    try:
        update_progress(10, "读取ENVI文件...")

        class_value_groups = [c["values"] for c in classes]
        class_names = [c["name"] for c in classes]

        update_progress(30, "执行重分类...")

        result = reclassify_with_geospatial_info(
            dat_path,
            class_value_groups,
            class_names,
        )

        if result["success"]:
            update_progress(100, "重分类完成!")
        else:
            update_progress(0, f"重分类失败: {result.get('error')}")

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
