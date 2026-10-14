"""
步骤1: 影像填充合并 - 核心处理逻辑

从 01填充合并.py 提取封装
"""

import os
import re
import csv
import shutil
import time
import tempfile
import sys as _sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ---- 多进程池初始化 ----
_BACKEND_DIR = str(Path(__file__).parent.parent)

def _init_pool():
    """ProcessPoolExecutor 子进程初始化：将 backend/ 加入 sys.path"""
    if _BACKEND_DIR not in _sys.path:
        _sys.path.insert(0, _BACKEND_DIR)

_MAX_WORKERS = min(16, max(8, int((os.cpu_count() or 4) * 1.5)))  # 优化：利用16核CPU


def _check_pause(task_manager, task_id: str) -> None:
    """阻塞检查暂停状态。如果任务被暂停，会循环等待直到恢复。
    
    由步骤处理线程调用。轮询 task_manager 检查 PAUSED 状态，
    暂停时每 1 秒检查一次，状态变为非 PAUSED 后立即返回。
    """
    from core.task_manager import TaskStatus
    while True:
        task = task_manager.get_task(task_id)
        if not task or task.status != TaskStatus.PAUSED:
            return  # 未被暂停，继续执行
        time.sleep(1)  # 暂停中，等待恢复


def make_pause_check(task_manager, task_id: str) -> Callable[[], None]:
    """创建一个暂停检查回调，供批处理函数在循环中调用。"""
    def _check():
        _check_pause(task_manager, task_id)
    return _check


# ============================================================
# 辅助函数
# ============================================================

def lazy_import_rasterio():
    """延迟导入 rasterio"""
    try:
        import rasterio
        return rasterio
    except ImportError:
        return None


def delete_specific_bands(input_folder: str, delete_ranges: list[tuple] = None):
    """删除指定波段范围的文件"""
    if delete_ranges is None:
        delete_ranges = [
            (1, 7), (58, 76), (120, 128), (170, 213), (225, 242)
        ]

    pattern = re.compile(r'(?:band|b|_)(\d+)', re.IGNORECASE)
    deleted_count = 0

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if not os.path.isfile(file_path):
            continue

        match = pattern.search(filename)
        if not match:
            continue

        try:
            band_num = int(match.group(1))
        except ValueError:
            continue

        to_delete = any(start <= band_num <= end for start, end in delete_ranges)
        if to_delete:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception:
                pass

    return deleted_count


def process_single_image(image_path: str, output_dir: str) -> bool:
    """处理单张影像并转换为CSV"""
    try:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{filename}.csv")

        rasterio = lazy_import_rasterio()
        if rasterio is not None:
            with rasterio.open(image_path) as src:
                band_count = src.count
                if band_count > 1:
                    for b in range(1, band_count + 1):
                        data_b = src.read(b)
                        band_output = os.path.join(output_dir, f"{filename}_B{b}.csv")
                        pd.DataFrame(data_b).to_csv(band_output, index=False, header=False)
                    return True
                data = src.read(1)
        else:
            from PIL import Image
            with Image.open(image_path) as img:
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    img.seek(0)
                data = np.array(img)
            if data.ndim == 3:
                if data.shape[2] >= 1:
                    data = data[:, :, 0]
                else:
                    return False

        pd.DataFrame(data).to_csv(output_path, index=False, header=False)
        return True

    except Exception as e:
        print(f"处理影像出错: {e}")
        return False


def batch_process_images(input_dir: str, output_dir: str, progress_callback=None, pause_check=None):
    """批量处理影像文件（多进程并行）
    
    pause_check: 可选暂停检查回调，每处理完一批就调用一次，暂停时会阻塞等待
    """
    supported_extensions = ['.tif', '.tiff', '.img']
    image_files = []
    for ext in supported_extensions:
        image_files.extend(glob(os.path.join(input_dir, f'*{ext}')))
        image_files.extend(glob(os.path.join(input_dir, f'*{ext.upper()}')))
    image_files = sorted(set(image_files))

    if not image_files:
        return 0

    success_count = 0
    total = len(image_files)
    done = 0
    batch_size = max(1, _MAX_WORKERS * 4)  # 每处理约32个就检查一次暂停

    with ProcessPoolExecutor(max_workers=_MAX_WORKERS, initializer=_init_pool) as ex:
        futures = {ex.submit(process_single_image, img, output_dir): img for img in image_files}
        for fut in as_completed(futures):
            try:
                if fut.result():
                    success_count += 1
            except Exception as e:
                print(f"影像处理异常: {e}")
            done += 1
            if done % batch_size == 0 and pause_check:
                pause_check()  # 暂停检查
            if progress_callback:
                progress_callback(done / total * 100)

    return success_count


def process_single_csv(csv_path: str, output_dir: str) -> bool:
    """处理单个CSV文件（零值替换）"""
    try:
        df = pd.read_csv(csv_path, header=None)
        df = df.replace(0, "")
        filename = os.path.basename(csv_path)
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False, header=False)
        return True
    except Exception:
        return False


def batch_process_csv(input_dir: str, output_dir: str, progress_callback=None, pause_check=None):
    """批量处理CSV文件（多进程并行）
    
    pause_check: 可选暂停检查回调，每处理完一批就调用一次
    """
    try:
        csv_files = glob(os.path.join(input_dir, "*.csv"))
    except Exception as e:
        raise RuntimeError(f"batch_csv: glob输入目录失败 [{input_dir}]: {e}") from e
    if not csv_files:
        return 0

    success_count = 0
    total = len(csv_files)
    done = 0
    batch_size = max(1, _MAX_WORKERS * 4)

    with ProcessPoolExecutor(max_workers=_MAX_WORKERS, initializer=_init_pool) as ex:
        futures = {ex.submit(process_single_csv, f, output_dir): f for f in csv_files}
        for fut in as_completed(futures):
            try:
                if fut.result():
                    success_count += 1
            except Exception as e:
                print(f"CSV处理异常: {e}")
            done += 1
            if done % batch_size == 0 and pause_check:
                pause_check()
            if progress_callback:
                progress_callback(done / total * 100)

    return success_count


def _classify_cell(cell, thresholds, num_classes):
    """单格值分类（顶层函数，可被多进程序列化）"""
    try:
        x = float(cell)
        if num_classes == 1:
            return '1'
        for i, threshold in enumerate(thresholds):
            if x <= threshold:
                return str(i + 1)
        return str(num_classes)
    except Exception:
        return str(cell)


def _process_single_quantile_file(args):
    """处理单个CSV分位数分类（顶层，可被多进程序列化）"""
    input_path, output_path, num_classes = args
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 2 or len(rows[0]) < 2:
            return False

        data_values = []
        for row in rows:
            for cell in row:
                try:
                    val = float(cell)
                    if not np.isnan(val):
                        data_values.append(val)
                except (ValueError, TypeError):
                    pass

        if not data_values:
            return False

        if num_classes > 1:
            quantiles = [(i + 1) / num_classes for i in range(num_classes - 1)]
            thresholds = sorted(np.quantile(data_values, quantiles).tolist())
        else:
            thresholds = []

        new_rows = [[_classify_cell(cell, thresholds, num_classes) for cell in row] for row in rows]

        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)

        return True
    except Exception as e:
        print(f"分类文件处理出错 {os.path.basename(input_path)}: {e}")
        return False


def process_large_csv_files_with_progress(
    input_dir: str,
    output_dir: str,
    num_classes: int = 5,
    progress_callback=None,
    pause_check=None,
):
    """分位数分类处理（多进程并行）
    
    pause_check: 可选暂停检查回调
    """
    try:
        csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    except Exception as e:
        raise RuntimeError(f"分位数分类: 读取输入目录失败 [{input_dir}]: {e}") from e
    if not csv_files:
        return 0

    args_list = [
        (os.path.join(input_dir, fn), os.path.join(output_dir, fn), num_classes)
        for fn in csv_files
    ]

    processed_count = 0
    total = len(args_list)
    done = 0
    batch_size = max(1, _MAX_WORKERS * 4)

    with ProcessPoolExecutor(max_workers=_MAX_WORKERS, initializer=_init_pool) as ex:
        futures = {ex.submit(_process_single_quantile_file, args): args for args in args_list}
        for fut in as_completed(futures):
            try:
                if fut.result():
                    processed_count += 1
            except Exception as e:
                print(f"分类任务异常: {e}")
            done += 1
            if progress_callback:
                progress_callback(done / total * 100)
            if done % batch_size == 0 and pause_check:
                pause_check()

    return processed_count


def merge_csv_cells_by_position(
    input_dir: str,
    output_dir: str,
    filename: str,
    adjust_sizes: bool = True,
    progress_callback=None
):
    """按像元位置合并CSV"""
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    if not filename.lower().endswith('.csv'):
        filename += '.csv'
    output_file = os.path.join(output_dir, filename)

    csv_files = []
    try:
        csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    except Exception as e:
        raise RuntimeError(f"merge_csv: 读取输入目录失败 [{input_dir}]: {e}") from e
    if not csv_files:
        return False

    merged_data = None
    target_rows = 0
    target_cols = 0
    file_paths = [os.path.join(input_dir, f) for f in csv_files]
    valid_files = 0

    for idx, file_path in enumerate(file_paths):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            if not rows or not rows[0]:
                continue

            current_rows = len(rows)
            current_cols = max(len(row) for row in rows) if rows else 0

            if target_rows == 0:
                target_rows = current_rows
                target_cols = current_cols
                merged_data = [[[] for _ in range(target_cols)] for _ in range(target_rows)]

            if adjust_sizes:
                if current_rows > target_rows:
                    rows = rows[:target_rows]
                elif current_rows < target_rows:
                    for _ in range(target_rows - current_rows):
                        rows.append([''] * min(current_cols, target_cols))

                for i in range(len(rows)):
                    row = rows[i]
                    if len(row) > target_cols:
                        rows[i] = row[:target_cols]
                    elif len(row) < target_cols:
                        rows[i].extend([''] * (target_cols - len(row)))

            for i in range(min(target_rows, len(rows))):
                for j in range(min(target_cols, len(rows[i]))):
                    cell = rows[i][j] if j < len(rows[i]) else ''
                    try:
                        val = f"{float(cell):.6g}" if cell != '' else ''
                        if val:
                            merged_data[i][j].append(val)
                    except (ValueError, TypeError):
                        if cell != '':
                            merged_data[i][j].append(str(cell))

            valid_files += 1

        except Exception:
            pass

        if progress_callback:
            progress_callback((idx + 1) / len(file_paths) * 50)  # 前50%进度

    if merged_data is None:
        return False

    # 构建最终结果
    final_rows = []
    for i in range(target_rows):
        new_row = []
        for j in range(target_cols):
            cell = merged_data[i][j]
            if isinstance(cell, list):
                new_row.append(" ".join(cell))
            else:
                new_row.append(str(cell))
        final_rows.append(new_row)

    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(final_rows)
        return True
    except Exception as e:
        raise RuntimeError(f"merge_csv: 写入输出文件失败 [{output_file}]: {e}") from e


def create_temp_dir(base_dir: str = None) -> str:
    """创建临时目录（自动使用系统临时目录，避免硬编码D盘）"""
    if base_dir is None:
        base_dir = os.path.join(tempfile.gettempdir(), "csv_processing")
    os.makedirs(base_dir, exist_ok=True)
    timestamp = int(datetime.now().timestamp() * 1000) % 1000000
    temp_dir = os.path.join(base_dir, f"temp_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def cleanup_temp_dir(temp_dir: str):
    """清理临时目录"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"临时目录已清理: {temp_dir}")
    except Exception as e:
        print(f"清理临时目录失败: {e}")


# ============================================================
# 主函数
# ============================================================

def run_step1(
    task_manager,
    task_id: str,
    input_img_dir: str,
    output_dir: str,
    filename: str,
    num_classes: int = 5,
    delete_ranges: list = None,
    adjust_sizes: bool = True,
    skip_discretization: bool = False,
) -> dict:
    """
    执行步骤1处理

    参数:
        task_manager: 任务管理器实例
        task_id: 任务ID
        input_img_dir: 影像文件夹路径
        output_dir: 输出目录路径
        filename: 输出文件名
        num_classes: 分位数分类数量
        delete_ranges: 要删除的波段范围
        adjust_sizes: 是否调整不同尺寸
        skip_discretization: 消融实验1 — 跳过五级分位数离散化，
            为 True 时不进行分位数分类，直接合并零值处理后的数值文件

    返回:
        dict: 包含 success, output_file, processed_files 等字段
    """
    def update_progress(progress: float, message: str):
        """更新任务进度 — 同时更新顶层 progress 和 result.step_progress，
        确保前端轮询时步骤进度条能正常推进。"""
        task_manager.update_task(
            task_id,
            progress=progress,
            message=message,
            result={
                "current_step": 1,
                "step_message": message,
                "step_progress": progress,
                "current_round": 0,
            },
        )

    # 创建暂停检查回调
    _pause_check = make_pause_check(task_manager, task_id)

    try:
        # ── 前置路径校验 ──
        print(f"[Step1] input_img_dir={input_img_dir}, exists={os.path.exists(input_img_dir)}, isdir={os.path.isdir(input_img_dir) if os.path.exists(input_img_dir) else False}")
        print(f"[Step1] output_dir={output_dir}")
        if not os.path.exists(input_img_dir):
            return {"success": False, "error": f"影像根目录不存在: {input_img_dir}"}
        if not os.path.isdir(input_img_dir):
            return {"success": False, "error": f"影像根目录不是文件夹: {input_img_dir}"}

        # 确保输出目录存在（防御性检查，应对外部未创建的情况）
        print(f"[Step1] 创建输出目录: {output_dir}")
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            return {"success": False, "error": f"创建输出目录失败 [{output_dir}]: {type(e).__name__}: {e}"}
        if not os.path.isdir(output_dir):
            return {"success": False, "error": f"输出目录创建失败或不是目录: {output_dir}"}
        # 检查写入权限（用测试文件验证）
        _test_file = os.path.join(output_dir, ".write_test")
        try:
            with open(_test_file, "w") as _f:
                _f.write("ok")
            os.remove(_test_file)
        except Exception as e:
            return {"success": False, "error": f"输出目录无写入权限 [{output_dir}]: {type(e).__name__}: {e}。请右键文件夹→属性→安全，赋予当前用户完全控制权限。"}
        print(f"[Step1] 输出目录已就绪")

        # 创建临时目录
        print(f"[Step1] 创建临时目录...")
        try:
            temp_dir = create_temp_dir()
        except Exception as e:
            return {"success": False, "error": f"创建临时目录失败: {type(e).__name__}: {e}"}
        print(f"[Step1] 临时目录: {temp_dir}")
        update_progress(5, "临时目录创建完成")

        # 步骤1: 可选的删除波段
        if delete_ranges:
            update_progress(10, "删除指定波段...")
            _pause_check()
            try:
                delete_specific_bands(input_img_dir, delete_ranges)
            except Exception as e:
                return {"success": False, "error": f"删除波段失败: {e}"}
        update_progress(15, "波段处理完成")
        _pause_check()

        # 步骤2: 影像转CSV
        temp_csv_dir_02 = os.path.join(temp_dir, "step02_csv")
        try:
            os.makedirs(temp_csv_dir_02, exist_ok=True)
        except Exception as e:
            return {"success": False, "error": f"创建临时目录 step02 失败: {type(e).__name__}: {e}"}
        update_progress(20, "影像文件转CSV...")

        image_files = []
        for ext in ['.tif', '.tiff', '.img']:
            image_files.extend(glob(os.path.join(input_img_dir, f'*{ext}')))
            image_files.extend(glob(os.path.join(input_img_dir, f'*{ext.upper()}')))

        print(f"[Step1] 发现影像文件: {len(image_files)} 个")
        _pause_check()
        if image_files:
            try:
                batch_process_images(input_img_dir, temp_csv_dir_02, pause_check=_pause_check)
            except Exception as e:
                return {"success": False, "error": f"影像转CSV失败: {e}"}
        else:
            # 没有影像文件，检查CSV
            csv_files = glob(os.path.join(input_img_dir, "*.csv"))
            print(f"[Step1] 发现CSV文件: {len(csv_files)} 个")
            for csv_file in csv_files:
                try:
                    shutil.copy(csv_file, temp_csv_dir_02)
                except Exception as e:
                    return {"success": False, "error": f"复制CSV失败 [{csv_file}]: {e}"}
        update_progress(35, "影像转CSV完成")
        _pause_check()

        # 步骤3: 零值处理
        temp_csv_dir_03 = os.path.join(temp_dir, "step03_csv")
        try:
            os.makedirs(temp_csv_dir_03, exist_ok=True)
        except Exception as e:
            return {"success": False, "error": f"创建临时目录 step03 失败: {type(e).__name__}: {e}"}
        update_progress(40, "零值处理...")
        _pause_check()

        try:
            batch_process_csv(temp_csv_dir_02, temp_csv_dir_03, pause_check=_pause_check)
        except Exception as e:
            return {"success": False, "error": f"零值处理失败: {e}"}
        update_progress(50, "零值处理完成")
        _pause_check()

        # 步骤4: 分位数分类（消融实验1: skip_discretization=True 时跳过，直接合并数值文件）
        if skip_discretization:
            update_progress(80, "消融实验1: 跳过离散化，直接合并数值文件")
            print("[Step1] skip_discretization=True, 跳过分位数分类，直接使用零值处理后的数值文件进行合并")
            merge_source_dir = temp_csv_dir_03
        else:
            temp_csv_dir_04 = os.path.join(temp_dir, "step04_csv")
            try:
                os.makedirs(temp_csv_dir_04, exist_ok=True)
            except Exception as e:
                return {"success": False, "error": f"创建临时目录 step04 失败: {type(e).__name__}: {e}"}
            update_progress(55, f"分位数分类 (共{num_classes}类)...")

            def class_progress(p):
                update_progress(55 + p * 0.25, f"分类进度 {int(p)}%")

            try:
                process_large_csv_files_with_progress(
                    temp_csv_dir_03,
                    temp_csv_dir_04,
                    num_classes,
                    class_progress,
                    pause_check=_pause_check
                )
            except Exception as e:
                return {"success": False, "error": f"分位数分类失败: {e}"}
            update_progress(80, "分类完成")
            merge_source_dir = temp_csv_dir_04

        # 步骤5: 合并CSV
        update_progress(85, "合并CSV文件...")
        print(f"[Step1] 合并CSV: input={merge_source_dir}, output={output_dir}, filename={filename}")
        try:
            result = merge_csv_cells_by_position(
                merge_source_dir,
                output_dir,
                filename,
                adjust_sizes,
            )
        except Exception as e:
            return {"success": False, "error": f"合并CSV失败: {e}"}

        if result:
            output_file = os.path.join(output_dir, filename if filename.endswith('.csv') else filename + '.csv')
            print(f"[Step1] 输出文件: {output_file}")
            if not os.path.isfile(output_file):
                return {"success": False, "error": f"输出文件未生成: {output_file}"}
            total_pixels = 0
            for row in open(output_file, 'r', encoding='utf-8'):
                total_pixels += len(row.split())

            update_progress(100, "处理完成!")
            return {
                "success": True,
                "output_file": output_file,
                "processed_files": len(image_files) if image_files else 0,
                "total_pixels": total_pixels,
            }
        else:
            return {
                "success": False,
                "error": "CSV合并失败（无有效数据或输入为空）",
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        if 'temp_dir' in locals():
            cleanup_temp_dir(temp_dir)
