"""
步骤3: 匹配测试 - 核心处理逻辑（NumPy 向量化 + GPU批量加速版）

核心优化：
1. 参考序列预构建为 NumPy 矩阵，避免每个单元格重复 Python 循环
2. 汉明距离用 (ref_matrix != current).sum(axis=1) 向量化计算
3. 并行度改为 max(8, CPU核心数)
4. GPU批量模式: 矩阵化汉明距离计算，比逐元素快5-8倍 (RTX 4050)
5. CPU/GPU自动切换: GPU不可用时无缝回退到多进程CPU
"""

import os
import csv
import time as _time
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed, CancelledError

import numpy as np

# ── GPU 加速支持 (延迟初始化) ─────────────────────────────────
_gpu_initialized = False
_gpu_available = False
cp = None

def _ensure_gpu_init():
    """延迟初始化GPU，避免模块导入时的副作用"""
    global _gpu_initialized, _gpu_available, cp
    if _gpu_initialized:
        return _gpu_available
    _gpu_initialized = True
    try:
        from scripts.gpu_optimize import init_gpu, USE_GPU
        init_gpu()
        if USE_GPU:
            import cupy as _cp
            cp = _cp
            _gpu_available = True
        else:
            _gpu_available = False
    except Exception as e:
        print(f"[GPU加速] 初始化失败，使用CPU: {e}")
        _gpu_available = False
    return _gpu_available

# ── 多进程子进程初始化 ─────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path

_BACKEND_DIR = str(_Path(__file__).parent.parent)


def _init_pool():
    """ProcessPoolExecutor 子进程初始化回调：将 backend/ 加入 sys.path"""
    if _BACKEND_DIR not in _sys.path:
        _sys.path.insert(0, _BACKEND_DIR)


class _Step3PausedSignal(Exception):
    """步骤3内部暂停信号：在多进程执行中途检测到暂停请求时抛出，由 run_step3 捕获。"""
    pass



# ============================================================
# 参考序列预处理（在主进程中一次性完成）
# ============================================================

def _build_ref_arrays(all_reference_sequences, current_replacements, target_sequence_length,
                      distance_metric='hamming', hamming_tolerance=0.0):
    """
    将 Python list 格式的参考序列转换为 NumPy 数组，便于向量化计算。

    参数:
        distance_metric: 距离度量，'hamming'（汉明距离，整数编码）或
            'euclidean'（欧氏距离，连续数值，消融实验2）
        hamming_tolerance: 汉明距离的容差 ε（仅在 distance_metric='hamming' 时生效）。
            0 = 严格逐位相等判定（经典汉明距离）；
            >0 = ε-容差汉明：|ref - current| <= ε 即视为该波段"相等"，不计入距离。
            用于消融实验1（跳过离散化）：连续浮点序列用严格相等判定会导致
            距离恒等于序列长度、任何阈值下都无法匹配（OA=0），必须放宽为容差判定。
            建议先对连续值做逐波段标准化，此时 ε 的单位为标准差。

    返回:
        ref_matrix:  (total_refs, seq_len)  hamming→np.int16 / euclidean 或 ε>0→np.float32
        repl_ids:    (total_refs,)          np.int32  每个参考序列对应的 replacement 索引
        uniq_repls:  list[str]              去重后的 replacement 值列表（按原始出现顺序）
    """
    # ── 安全检查：长度必须一致 ──
    if len(current_replacements) != len(all_reference_sequences):
        import sys as _sys
        print(
            f"[步骤3 警告] _build_ref_arrays: 长度不匹配! "
            f"current_replacements({len(current_replacements)}) vs "
            f"all_reference_sequences({len(all_reference_sequences)})",
            file=_sys.stderr, flush=True
        )
        # 截断到较短的长度，避免 IndexError
        min_len = min(len(current_replacements), len(all_reference_sequences))
        current_replacements = current_replacements[:min_len]
        all_reference_sequences = all_reference_sequences[:min_len]

    # 建立 replacement 值 -> id 的映射
    uniq_repls = []
    repl_to_id = {}
    for r in current_replacements:
        if r not in repl_to_id:
            repl_to_id[r] = len(uniq_repls)
            uniq_repls.append(r)

    ref_list = []
    repl_id_list = []

    for sample_idx, ref_sequences in enumerate(all_reference_sequences):
        repl_id = repl_to_id[current_replacements[sample_idx]]
        for ref in ref_sequences:
            if len(ref) == target_sequence_length:
                ref_list.append(ref)
                repl_id_list.append(repl_id)

    if not ref_list:
        return None, None, uniq_repls


    _use_tol = (distance_metric == 'hamming' and hamming_tolerance > 0)

    if distance_metric == 'euclidean':
        # 欧氏距离模式（消融实验2）：参考序列保持连续数值，用 float32 存储
        ref_matrix = np.array(ref_list, dtype=np.float32)   # (total_refs, seq_len)
    elif _use_tol:
        # ε-容差汉明（消融实验1的配套判定）：需要做减法比较，必须用 float32
        print(
            f"[步骤3] ε-容差汉明距离: ε={hamming_tolerance}，参考矩阵改用 float32",
            flush=True
        )
        ref_matrix = np.array(ref_list, dtype=np.float32)   # (total_refs, seq_len)
    else:
        # 汉明距离模式：离散整数编码用 int16 存储；
        # 若序列含非整数值（消融实验1跳过离散化后的连续数值），改用 float32，
        # 汉明距离的逐位不等比较对浮点同样成立，避免 int() 截断破坏数值
        all_integral = all(
            float(v).is_integer()
            for seq in ref_list for v in seq
        )
        if all_integral:
            ref_matrix = np.array(ref_list, dtype=np.int16)   # (total_refs, seq_len)
        else:
            print("[步骤3] 检测到非整数序列（可能为跳过离散化的数值模式），汉明距离改用float32比较", flush=True)
            ref_matrix = np.array(ref_list, dtype=np.float32)  # (total_refs, seq_len)
    repl_ids = np.array(repl_id_list, dtype=np.int32)    # (total_refs,)
    return ref_matrix, repl_ids, uniq_repls


# ============================================================
# 多进程 worker（NumPy 向量化版）
# ============================================================

def _match_block_vectorized(row_indices, data_rows, target_sequence_length,
                          ref_matrix, repl_ids, uniq_repls, threshold,
                          use_gpu=False, distance_metric='hamming',
                          hamming_tolerance=0.0):
    """
    多进程 worker：使用向量化计算距离（自动选择CPU/GPU、汉明/欧氏距离）

    参数:
        use_gpu: 是否使用GPU加速
        distance_metric: 'hamming'（汉明距离，整数编码）或 'euclidean'（欧氏距离，消融实验2）
        hamming_tolerance: 汉明距离容差 ε，>0 时逐波段判定放宽为 |ref-current| <= ε 即相等

    返回: [(row_idx, new_row, empty_count, ambiguous_count, modified_count), ...]
    """
    import numpy as np
    _tol = float(hamming_tolerance) if hamming_tolerance else 0.0

    if use_gpu:
        try:
            import cupy as cp
            
            # 将数据移到GPU
            if not isinstance(ref_matrix, cp.ndarray):
                ref_matrix_gpu = cp.asarray(ref_matrix)
            else:
                ref_matrix_gpu = ref_matrix
            
            if not isinstance(repl_ids, cp.ndarray):
                repl_ids_gpu = cp.asarray(repl_ids)
            else:
                repl_ids_gpu = repl_ids
            
            results = []
            total_refs = ref_matrix_gpu.shape[0] if ref_matrix_gpu is not None else 0
            
            for row_idx, row in zip(row_indices, data_rows):
                new_row = list(row)
                empty_assigned = 0
                ambiguous_count = 0
                modified_count = 0
                
                for col_idx, cell in enumerate(row):
                    if not cell.strip():
                        new_row[col_idx] = "-1"
                        empty_assigned += 1
                        continue
                    
                    try:
                        vals = cell.split()
                        if len(vals) != target_sequence_length:
                            continue
                        
                        # GPU上的数组构建：dtype 跟随参考矩阵
                        # （int16=离散编码，float32=欧氏距离或跳过离散化的连续数值）
                        if ref_matrix_gpu.dtype == cp.float32:
                            current_arr = cp.fromiter(
                                (float(v) for v in vals),
                                dtype=cp.float32,
                                count=target_sequence_length
                            )
                        else:
                            current_arr = cp.fromiter(
                                (int(float(v)) for v in vals),
                                dtype=cp.int16,
                                count=target_sequence_length
                            )
                    except ValueError:
                        continue
                    
                    if total_refs == 0:
                        continue
                    
                    # ── GPU上向量化距离计算 ──
                    if distance_metric == 'euclidean':
                        # 欧氏距离（消融实验2）：sqrt(Σ(ref-current)^2)
                        diff = ref_matrix_gpu - current_arr
                        dists = cp.sqrt((diff * diff).sum(axis=1))
                    elif _tol > 0:
                        # ε-容差汉明（消融实验1）：|ref-current| > ε 才计入不一致波段数
                        # 距离仍为「不一致波段数」，量纲与经典汉明一致（整数计数）
                        dists = (cp.abs(ref_matrix_gpu - current_arr) > _tol).sum(axis=1)
                    else:
                        # 汉明距离：逐位不等计数
                        dists = cp.not_equal(ref_matrix_gpu, current_arr).sum(axis=1)
                    min_dist = float(cp.min(dists).get())  # 传回CPU
                    
                    if min_dist > threshold:
                        continue
                    
                    # 找出所有最小距离的索引
                    min_mask = dists == min_dist
                    matched_repl_ids = cp.unique(repl_ids_gpu[min_mask]).get()
                    
                    if len(matched_repl_ids) > 1:
                        ambiguous_count += 1
                        continue
                    
                    new_row[col_idx] = uniq_repls[int(matched_repl_ids[0])]
                    modified_count += 1
                
                results.append((row_idx, new_row, empty_assigned, ambiguous_count, modified_count))
            return results
            
        except Exception as e:
            print(f"[GPU加速] GPU计算失败，回退到CPU: {e}")
            # 回退到CPU版本
    
    # CPU版本（NumPy）
    results = []
    total_refs = ref_matrix.shape[0] if ref_matrix is not None else 0
    
    for row_idx, row in zip(row_indices, data_rows):
        new_row = list(row)
        empty_assigned = 0
        ambiguous_count = 0
        modified_count = 0
        
        for col_idx, cell in enumerate(row):
            if not cell.strip():
                new_row[col_idx] = "-1"
                empty_assigned += 1
                continue
            
            try:
                vals = cell.split()
                if len(vals) != target_sequence_length:
                    continue
                # dtype 跟随参考矩阵：int16=离散编码，float32=欧氏距离或跳过离散化的连续数值
                if ref_matrix.dtype == np.float32:
                    current_arr = np.fromiter(
                        (float(v) for v in vals),
                        dtype=np.float32,
                        count=target_sequence_length
                    )
                else:
                    current_arr = np.fromiter(
                        (int(float(v)) for v in vals),
                        dtype=np.int16,
                        count=target_sequence_length
                    )
            except ValueError:
                continue
            
            if total_refs == 0:
                continue
            
            # ── 向量化距离计算 ──
            if distance_metric == 'euclidean':
                # 欧氏距离（消融实验2）：sqrt(Σ(ref-current)^2)
                diff = ref_matrix - current_arr
                dists = np.sqrt((diff * diff).sum(axis=1))
            elif _tol > 0:
                # ε-容差汉明（消融实验1）：|ref-current| > ε 才计入不一致波段数
                # 距离仍为「不一致波段数」，量纲与经典汉明一致（整数计数）
                dists = (np.abs(ref_matrix - current_arr) > _tol).sum(axis=1)
            else:
                # 汉明距离：逐位不等计数
                dists = np.not_equal(ref_matrix, current_arr).sum(axis=1)
            min_dist = float(dists.min())
            
            if min_dist > threshold:
                continue
            
            # 找出所有最小距离的索引
            min_mask = dists == min_dist
            matched_repl_ids = np.unique(repl_ids[min_mask])
            
            if matched_repl_ids.size > 1:
                ambiguous_count += 1
                continue
            
            new_row[col_idx] = uniq_repls[int(matched_repl_ids[0])]
            modified_count += 1
        
        results.append((row_idx, new_row, empty_assigned, ambiguous_count, modified_count))
    return results
    results = []
    total_refs = ref_matrix.shape[0] if ref_matrix is not None else 0

    for row_idx, row in zip(row_indices, data_rows):
        new_row = list(row)
        empty_assigned = 0
        ambiguous_count = 0
        modified_count = 0

        for col_idx, cell in enumerate(row):
            if not cell.strip():
                new_row[col_idx] = "-1"
                empty_assigned += 1
                continue

            try:
                vals = cell.split()
                if len(vals) != target_sequence_length:
                    continue
                # 直接构建 numpy 数组，避免 Python list 中转
                current_arr = np.fromiter(
                    (int(float(v)) for v in vals),
                    dtype=np.int16,
                    count=target_sequence_length
                )
            except ValueError:
                continue

            if total_refs == 0:
                continue

            # ── 向量化汉明距离计算 ──
            # (total_refs, seq_len) != (seq_len,)  → 广播 → (total_refs, seq_len)
            # sum(axis=1) → (total_refs,)
            dists = np.not_equal(ref_matrix, current_arr).sum(axis=1)
            min_dist = int(dists.min())

            if min_dist > threshold:
                continue

            # 找出所有最小距离的索引
            min_mask = dists == min_dist
            matched_repl_ids = np.unique(repl_ids[min_mask])

            if matched_repl_ids.size > 1:
                ambiguous_count += 1
                continue

            new_row[col_idx] = uniq_repls[int(matched_repl_ids[0])]
            modified_count += 1

        results.append((row_idx, new_row, empty_assigned, ambiguous_count, modified_count))
    return results


# ============================================================
# GPU 批量匹配（核心加速路径）
# ============================================================

def _match_round_gpu_batch(
    modified_data,            # list[list[str]] - 当前数据
    target_sequence_length,   # int - 序列长度
    ref_matrix,               # (n_refs, seq_len) np.int16/np.float32 - 参考矩阵
    repl_ids,                 # (n_refs,) np.int32 - 类别ID
    uniq_repls,               # list[str] - 类别名称列表
    threshold,                # float - 距离阈值（汉明:整数位差数 / 欧氏:连续距离值）
    task_manager=None,        # 任务管理器（暂停检查）
    task_id=None,             # 任务ID
    progress_callback=None,   # 进度回调 fn(progress, message)
    distance_metric='hamming',  # 距离度量: 'hamming' / 'euclidean'（消融实验2）
    hamming_tolerance=0.0,    # 汉明距离容差 ε（>0 时对连续值启用容差判定，消融实验1）
):
    """
    GPU批量匹配 - 一整轮匹配的核心GPU加速函数
    
    策略:
    1. 收集所有非空单元格，构建为NumPy矩阵
    2. 调用GPU批量距离计算（汉明/欧氏）
    3. 根据阈值和歧义检查，写回结果
    
    返回:
        (modified_in_round, empty_assigned_in_round, ambiguous_in_round)
    """
    from scripts.gpu_optimize import batch_hamming_distance_gpu, estimate_batch_size, USE_GPU, cp as _cp
    
    if not USE_GPU or _cp is None:
        raise RuntimeError("GPU not available, should use CPU path")
    
    n_refs = ref_matrix.shape[0]
    if n_refs == 0:
        return 0, 0, 0
    
    seq_len = target_sequence_length
    
    # ── Step 1: 收集所有待匹配单元格 ──
    cell_positions = []  # [(row_idx, col_idx), ...]
    cell_values = []     # [str, str, ...]
    empty_count = 0
    
    for row_idx, row in enumerate(modified_data):
        for col_idx, cell in enumerate(row):
            if not cell.strip():
                if modified_data[row_idx][col_idx] != "-1":
                    modified_data[row_idx][col_idx] = "-1"
                empty_count += 1
                continue
            try:
                vals = cell.split()
                if len(vals) != seq_len:
                    continue
                # dtype 跟随参考矩阵：int16=离散编码，float32=欧氏/连续数值
                if ref_matrix.dtype == np.float32:
                    [float(v) for v in vals]
                else:
                    [int(float(v)) for v in vals]
                cell_positions.append((row_idx, col_idx))
                cell_values.append(cell)
            except ValueError:
                continue
    
    n_cells = len(cell_positions)
    if n_cells == 0:
        return 0, empty_count, 0
    
    # ── Step 2: 构建像元矩阵（dtype 与参考矩阵一致） ──
    if ref_matrix.dtype == np.float32:
        cells_matrix = np.empty((n_cells, seq_len), dtype=np.float32)
        for i, cell in enumerate(cell_values):
            vals = cell.split()
            cells_matrix[i] = [float(v) for v in vals]
    else:
        cells_matrix = np.empty((n_cells, seq_len), dtype=np.int16)
        for i, cell in enumerate(cell_values):
            vals = cell.split()
            cells_matrix[i] = [int(float(v)) for v in vals]
    
    # ── Step 3: 估算批量大小（欧氏距离中间张量为float32，显存占用更大） ──
    batch_size = estimate_batch_size(n_refs, seq_len, safety_factor=0.6,
                                     distance_metric=distance_metric)
    
    # ── Step 4: GPU批量计算 ──
    def _gpu_progress(current, total):
        if progress_callback:
            pct = current / total
            progress_callback(pct, f"GPU匹配中... {current}/{total} ({pct*100:.0f}%)")
        if task_manager and task_id:
            _task = task_manager.get_task(task_id)
            if _task and _task.status.value == 'paused':
                raise _Step3PausedSignal()
    
    try:
        result = batch_hamming_distance_gpu(
            cells_matrix, ref_matrix, repl_ids,
            batch_size=batch_size,
            threshold=threshold,
            progress_callback=_gpu_progress,
            distance_metric=distance_metric,
            hamming_tolerance=hamming_tolerance,
        )
    except _Step3PausedSignal:
        raise
    except Exception as e:
        print(f"[GPU加速] GPU批量计算失败，回退到CPU: {e}")
        raise
    
    # ── Step 5: 写回结果 ──
    min_dists = result['min_distances']
    matched_ids = result['matched_repl_ids']
    ambiguous_flags = result['ambiguous']
    
    modified_count = 0
    ambiguous_count = 0
    
    for i, (row_idx, col_idx) in enumerate(cell_positions):
        if ambiguous_flags[i]:
            ambiguous_count += 1
            continue
        if matched_ids[i] >= 0:
            modified_data[row_idx][col_idx] = uniq_repls[int(matched_ids[i])]
            modified_count += 1
    
    return modified_count, empty_count, ambiguous_count


def get_image_count_from_folder(folder_path: str):
    """从文件夹中统计影像图数量"""
    supported_extensions = ['.tif', '.tiff', '.img']
    image_files = []
    for ext in supported_extensions:
        image_files.extend(glob(os.path.join(folder_path, f'*{ext}')))
        image_files.extend(glob(os.path.join(folder_path, f'*{ext.upper()}')))
    image_files = sorted(set(image_files))
    return len(image_files), image_files


def get_image_spatial_dims(folder_path: str):
    """从文件夹中的第一个影像文件获取空间尺寸（高度×宽度）。

    返回:
        (height, width) 或 (None, None)（无法读取时）
    """
    _, image_files = get_image_count_from_folder(folder_path)
    if not image_files:
        return None, None
    # 优先用第一个 TIF 文件
    ref_file = None
    for f in image_files:
        if f.lower().endswith(('.tif', '.tiff')):
            ref_file = f
            break
    if ref_file is None:
        ref_file = image_files[0]
    try:
        from scripts.process01 import lazy_import_rasterio
        rasterio = lazy_import_rasterio()
        if rasterio is not None:
            with rasterio.open(ref_file) as src:
                return src.height, src.width
        else:
            from PIL import Image
            with Image.open(ref_file) as img:
                w, h = img.size  # PIL 返回 (width, height)
                return h, w
    except Exception as e:
        print(f"[步骤3] 读取影像空间尺寸失败: {e}")
        return None, None


def _resize_csv_rows(target_rows: list, target_height: int, target_width: int) -> list:
    """将 CSV 行数据调整为指定尺寸（行数×列数）。

    - 行数不足：末尾补空行
    - 行数过多：截断到目标行数
    - 列数不足：每行末尾补空字符串
    - 列数过多：截断到目标列数

    返回调整后的新行列表。
    """
    current_rows = len(target_rows)
    current_cols = max((len(row) for row in target_rows), default=0)

    resized = []
    for i in range(target_height):
        if i < current_rows:
            row = list(target_rows[i])  # 拷贝
        else:
            row = [''] * target_width   # 新增空行
        # 调整列数
        if len(row) > target_width:
            row = row[:target_width]
        elif len(row) < target_width:
            row.extend([''] * (target_width - len(row)))
        resized.append(row)

    return resized


# ============================================================
# 主函数
# ============================================================

def run_step3(
    task_manager,
    task_id: str,
    images_folder: str,
    samples_folder: str,
    target_file: str,
    output_file: str,
    match_order: list[str],
    rounds: list[dict],
    execution_mode: str = 'multiprocess',
    num_processes: int = 0,
    distance_metric: str = 'hamming',
    hamming_tolerance: float = 0.0,
) -> dict:
    """
    执行步骤3处理（NumPy 向量化优化版）

    参数:
        num_processes: 多进程模式下使用的进程数。0 表示自动（max(8, CPU核心数)）。
        distance_metric: 距离度量，'hamming'（默认，汉明距离）或
            'euclidean'（欧氏距离，消融实验2）。欧氏模式下序列保持连续数值，
            阈值语义为连续距离值而非整数位差数。
        hamming_tolerance: 汉明距离容差 ε（仅 hamming 模式生效，默认 0）。
            消融实验1（跳过离散化）必须配合 ε>0：连续浮点序列用严格相等判定时
            距离恒等于序列长度，任何阈值下都不会命中。
            ε>0 时逐波段判定为 |ref - current| <= ε 即相等，距离仍是不一致波段数，
            阈值语义与经典汉明一致。建议输入序列先做逐波段标准化，此时 ε 单位为标准差。
    """
    def update_progress(progress: float, message: str):
        task_manager.update_task(task_id, progress=progress, message=message)

    try:
        # 获取影像图数量
        update_progress(5, "统计影像图数量...")
        image_count, image_files = get_image_count_from_folder(images_folder)
        if image_count == 0:
            return {"success": False, "error": "未找到影像文件"}

        # ★ 关键修复：序列长度以 merged.csv 的实际长度为权威。
        #   优先从 step1 写的 .meta.json 读；其次从 merged.csv 第一格推断；
        #   最后才退回 count_total_bands / 文件数。
        #   这样无论 Sentinel（111→127）/Landsat（340→实际可能335）都自动对齐。
        from scripts.process01 import resolve_required_sequence_length
        target_sequence_length, length_source = resolve_required_sequence_length(
            experiment_data_path=target_file,
            tif_folder=images_folder,
        )
        if target_sequence_length == 0:
            target_sequence_length = image_count
            length_source = f"图片数兜底({images_folder})"
        update_progress(10, f"检测到序列长度: {target_sequence_length} ← {length_source}")
        print(
            f"[步骤3] ★ 序列长度={target_sequence_length} ← {length_source}",
            flush=True,
        )

        # 读取待匹配文件
        update_progress(15, "读取待匹配文件...")
        target_rows = None
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1']

        for encoding in encodings:
            try:
                with open(target_file, 'r', newline='', encoding=encoding) as f:
                    target_rows = list(csv.reader(f))
                break
            except Exception:
                continue

        if target_rows is None:
            _exists = os.path.exists(target_file) if target_file else False
            _size = os.path.getsize(target_file) if _exists else 0
            print(
                f"[步骤3错误] 无法读取待匹配文件: target_file={target_file}, "
                f"exists={_exists}, size={_size}, "
                f"尝试编码={encodings}",
                flush=True,
            )
            return {"success": False, "error": f"无法读取待匹配文件: {target_file} (存在={_exists}, 大小={_size})"}

        # ── 自动尺寸适配：将 CSV 调整为影像文件的空间尺寸 ──
        img_height, img_width = get_image_spatial_dims(images_folder)
        csv_rows = len(target_rows)
        csv_cols = max((len(row) for row in target_rows), default=0)

        if img_height is not None and img_width is not None:
            if csv_rows != img_height or csv_cols != img_width:
                print(
                    f"[步骤3] CSV尺寸({csv_rows}×{csv_cols})与影像尺寸({img_height}×{img_width})不匹配，"
                    f"自动调整CSV尺寸为影像尺寸",
                    flush=True,
                )
                update_progress(
                    17,
                    f"CSV尺寸({csv_rows}×{csv_cols})→影像尺寸({img_height}×{img_width})，自动调整..."
                )
                target_rows = _resize_csv_rows(target_rows, img_height, img_width)
                csv_rows = img_height
                csv_cols = img_width

        # 读取参考序列
        update_progress(20, "读取参考序列...")
        all_reference_sequences = []
        sample_names = []
        total_cells = sum(len(row) for row in target_rows if row)

        # ── 诊断：列出 samples_folder 内容 ──
        _diag_lines = []
        _diag_lines.append(f"[步骤3 诊断] 读取参考序列")
        _diag_lines.append(f"  samples_folder = {samples_folder}")
        _diag_lines.append(f"  target_sequence_length = {target_sequence_length}")
        _diag_lines.append(f"  match_order = {match_order}")
        try:
            _actual_files = sorted(os.listdir(samples_folder)) if os.path.isdir(samples_folder) else []
            _diag_lines.append(f"  samples_folder 实际文件: {_actual_files}")
        except Exception as _e:
            _diag_lines.append(f"  ⚠ 无法列出 samples_folder: {_e}")

        for idx, sample_file in enumerate(match_order):
            sample_path = os.path.join(samples_folder, sample_file)
            _diag_lines.append(f"  [{idx}] sample_file={sample_file}")
            _diag_lines.append(f"       sample_path={sample_path}")
            _diag_lines.append(f"       os.path.exists={os.path.exists(sample_path)}")

            if not os.path.exists(sample_path):
                all_reference_sequences.append([])
                sample_names.append(sample_file)
                _diag_lines.append(f"       ❌ 文件不存在，跳过")
                continue

            sample_names.append(sample_file)

            sample_rows = None
            _used_encoding = None
            for encoding in encodings:
                try:
                    with open(sample_path, 'r', newline='', encoding=encoding) as f:
                        sample_rows = list(csv.reader(f))
                    _used_encoding = encoding
                    break
                except Exception:
                    continue

            if sample_rows is None:
                _diag_lines.append(f"       ❌ 所有编码都无法读取: {os.path.getsize(sample_path) if os.path.isfile(sample_path) else 'N/A'} bytes")
                all_reference_sequences.append([])
                continue

            _diag_lines.append(f"       编码={_used_encoding}, 行数={len(sample_rows)}")

            if len(sample_rows) < 2:
                _diag_lines.append(f"       ❌ CSV 行数<2（仅表头或空文件），跳过")
                all_reference_sequences.append([])
                continue

            header_sample = sample_rows[0]
            _diag_lines.append(f"       表头列: {header_sample}")
            sequence_column_idx = -1
            for i, col_name in enumerate(header_sample):
                if col_name.strip() == '提取值':
                    sequence_column_idx = i
                    break

            if sequence_column_idx == -1:
                _diag_lines.append(f"       ❌ 未找到'提取值'列，跳过")
                all_reference_sequences.append([])
                continue

            reference_sequences = []
            _cell_attempts = 0
            _cell_empty = 0
            _cell_single = 0
            _cell_wrong_len = 0
            _cell_invalid = 0
            for i in range(1, len(sample_rows)):
                cell = sample_rows[i][sequence_column_idx].strip()
                _cell_attempts += 1
                if not cell:
                    _cell_empty += 1
                    continue
                try:
                    float_val = float(cell)
                    if cell.count(' ') == 0:
                        _cell_single += 1
                        continue
                except ValueError:
                    pass
                try:
                    values = cell.split()
                    if len(values) == target_sequence_length:
                        # 统一按浮点解析（离散编码如"3"也能解析），
                        # 存储类型（int16/float32）由 _build_ref_arrays 根据数值特征决定：
                        # 离散整数编码 → int16（汉明）；连续数值（欧氏/跳过离散化）→ float32
                        seq = [float(v) for v in values]
                        reference_sequences.append(seq)
                    else:
                        _cell_wrong_len += 1
                except Exception:
                    _cell_invalid += 1
                    continue

            _diag_lines.append(
                f"       提取结果: {len(reference_sequences)} 条有效序列 "
                f"(共{_cell_attempts}个单元格, 空={_cell_empty}, "
                f"单值={_cell_single}, 长度不匹配={_cell_wrong_len}, 无效={_cell_invalid})"
            )

            all_reference_sequences.append(reference_sequences)
            update_progress(20 + (idx + 1) / len(match_order) * 15, f"读取样本 {sample_file}")

        total_sequences = sum(len(seqs) for seqs in all_reference_sequences)
        _diag_lines.append(f"  ── 汇总: total_sequences={total_sequences} ──")

        if total_sequences == 0:
            # ── 构建详细错误摘要 ──
            _diag_text = "\n".join(_diag_lines)
            print(_diag_text, flush=True)

            # 为每个样本文件生成一行摘要
            _per_file_summary = []
            for idx, sample_file in enumerate(match_order):
                file_exists = os.path.exists(os.path.join(samples_folder, sample_file))
                if idx < len(all_reference_sequences):
                    seqs = all_reference_sequences[idx]
                    _per_file_summary.append(
                        f"{sample_file}(存在={file_exists}, 有效={len(seqs) if seqs else 0})"
                    )
                else:
                    _per_file_summary.append(f"{sample_file}(??? 未处理)")

            # 尝试取第一个有数据的CSV的前几行"提取值"列值来展示格式
            _sample_preview = ""
            for idx, sample_file in enumerate(match_order):
                sp = os.path.join(samples_folder, sample_file)
                if not os.path.isfile(sp):
                    continue
                try:
                    with open(sp, 'r', encoding='utf-8-sig') as _sf:
                        _srows = list(csv.reader(_sf))
                    if len(_srows) < 2:
                        continue
                    _seq_col = -1
                    _header = _srows[0]
                    for _ci, _cn in enumerate(_header):
                        if _cn.strip() == '提取值':
                            _seq_col = _ci
                            break
                    if _seq_col == -1:
                        continue
                    _previews = []
                    for _ri in range(1, min(6, len(_srows))):
                        _cv = _srows[_ri][_seq_col].strip() if _seq_col < len(_srows[_ri]) else ''
                        if _cv:
                            _previews.append(f"行{_ri}: [{_cv[:60]}{'...' if len(_cv)>60 else ''}]")
                    if _previews:
                        _sample_preview = f"  {sample_file} 提取值列前5行: " + "; ".join(_previews)
                        break
                except Exception:
                    pass

            # 写诊断日志文件
            _diag_file = os.path.join(os.path.dirname(samples_folder), "step3_diag_error.log")
            try:
                with open(_diag_file, "w", encoding="utf-8") as _df:
                    _df.write(_diag_text)
            except Exception:
                _diag_file = "(写入失败)"
                pass

            error_msg = (
                f"未找到有效参考序列。"
                f"共{len(match_order)}个样本: " + "; ".join(_per_file_summary) + "。"
                f"序列长度要求={target_sequence_length}。"
                + (_sample_preview or "无法预览CSV内容。")
                + f" 完整诊断: {_diag_file}"
            )
            return {"success": False, "error": error_msg}

        update_progress(35, "开始多轮匹配...")

        # 多轮匹配
        modified_data = [row[:] for row in target_rows]
        total_modified = 0
        total_empty_assigned = 0
        total_ambiguous = 0

        # 并行度：优先使用传入的 num_processes；0 表示自动（max(8, CPU核心数)）
        if num_processes and num_processes > 0:
            NUM_PROCESSES = num_processes
        else:
            # 优化：使用CPU核心数的1.5倍，最多16个进程
            cpu_count = os.cpu_count() or 4
            NUM_PROCESSES = min(16, max(8, int(cpu_count * 1.5)))

        for round_num, config in enumerate(rounds):
            threshold = config.get('threshold', 999)
            replacements_dict = config.get('replacements', {})

            current_replacements = []
            for sample_file in match_order:
                current_replacements.append(replacements_dict.get(sample_file, ''))

            # ── 关键优化：预构建 NumPy 参考矩阵 ──
            ref_matrix, repl_ids, uniq_repls = _build_ref_arrays(
                all_reference_sequences, current_replacements, target_sequence_length,
                distance_metric=distance_metric,
                hamming_tolerance=hamming_tolerance,
            )

            # ── GPU可用性检测 ──
            gpu_ready = _ensure_gpu_init()
            use_gpu_mode = (execution_mode == 'gpu') or (gpu_ready and execution_mode != 'serial' and execution_mode != 'multiprocess')

            modified_in_round = 0
            empty_assigned_in_round = 0
            ambiguous_in_round = 0

            if use_gpu_mode and gpu_ready and ref_matrix is not None:
                # ═══════════════════════════════════════════════════
                # GPU批量模式 (核心加速路径)
                # ═══════════════════════════════════════════════════
                from scripts.gpu_optimize import get_gpu_info
                gpu_info = get_gpu_info()
                mode_label = f"GPU加速 ({gpu_info.get('name', 'Unknown')})"
                update_progress(
                    35 + round_num / len(rounds) * 50,
                    f"第 {round_num + 1} 轮匹配（{mode_label}）..."
                )

                try:
                    modified_in_round, empty_assigned_in_round, ambiguous_in_round = _match_round_gpu_batch(
                        modified_data, target_sequence_length,
                        ref_matrix, repl_ids, uniq_repls, threshold,
                        task_manager=task_manager, task_id=task_id,
                        progress_callback=lambda pct, msg: update_progress(
                            35 + round_num / len(rounds) * 50 + pct * 50 / len(rounds),
                            f"第 {round_num + 1} 轮 {msg}"
                        ),
                        distance_metric=distance_metric,
                        hamming_tolerance=hamming_tolerance,
                    )
                except _Step3PausedSignal:
                    raise
                except Exception as e:
                    print(f"[GPU加速] GPU匹配失败，回退到CPU多进程: {e}", flush=True)
                    use_gpu_mode = False  # 标记本轮回退到CPU
                    # 重新构建数据（GPU可能已部分修改）
                    # 继续走CPU路径

            if not use_gpu_mode:
                # ═══════════════════════════════════════════════════
                # CPU模式 (串行或多进程)
                # ═══════════════════════════════════════════════════
                mode_label = f"多进程 x{NUM_PROCESSES}" if execution_mode == 'multiprocess' else "串行"
                if execution_mode == 'serial':
                    mode_label = "串行（CPU向量化）"
                update_progress(
                    35 + round_num / len(rounds) * 50,
                    f"第 {round_num + 1} 轮匹配（{mode_label}）..."
                )

                if execution_mode == 'serial':
                    # 串行模式使用向量化
                    results = _match_block_vectorized(
                        list(range(len(modified_data))), modified_data,
                        target_sequence_length, ref_matrix, repl_ids, uniq_repls, threshold,
                        use_gpu=False, distance_metric=distance_metric,
                        hamming_tolerance=hamming_tolerance,
                    )
                    for row_idx, new_row, empty_count, ambiguous_count, modified_count in results:
                        modified_data[row_idx] = new_row
                        empty_assigned_in_round += empty_count
                        ambiguous_in_round += ambiguous_count
                        modified_in_round += modified_count

                else:
                    # 多进程模式
                    num_rows = len(modified_data)
                    chunk_size = (num_rows + NUM_PROCESSES - 1) // NUM_PROCESSES
                    chunks = []
                    for i in range(NUM_PROCESSES):
                        start = i * chunk_size
                        end = min(start + chunk_size, num_rows)
                        if start < end:
                            row_indices = list(range(start, end))
                            data_rows = [modified_data[idx] for idx in row_indices]
                            chunks.append((row_indices, data_rows))

                    with ProcessPoolExecutor(max_workers=NUM_PROCESSES, initializer=_init_pool) as executor:
                        futures = []
                        for row_indices, data_rows in chunks:
                            future = executor.submit(
                                _match_block_vectorized, row_indices, data_rows,
                                target_sequence_length, ref_matrix, repl_ids,
                                uniq_repls, threshold,
                                use_gpu=False, distance_metric=distance_metric,
                                hamming_tolerance=hamming_tolerance,
                            )
                            futures.append(future)

                        pending = set(futures)
                        while pending:
                            _task = task_manager.get_task(task_id) if task_manager else None
                            if _task and _task.status.value == 'paused':
                                for f in pending:
                                    f.cancel()
                                executor.shutdown(wait=False, cancel_futures=True)
                                print(f"[步骤3] 检测到暂停信号，中断第{round_num+1}轮多进程匹配", flush=True)
                                raise _Step3PausedSignal()

                            done_now = set()
                            for f in list(pending):
                                if f.done():
                                    done_now.add(f)
                            if not done_now:
                                _time.sleep(0.25)
                                continue
                            for future in done_now:
                                pending.discard(future)
                                try:
                                    results = future.result()
                                    for row_idx, new_row, empty_count, ambiguous_count, modified_count in results:
                                        modified_data[row_idx] = new_row
                                        empty_assigned_in_round += empty_count
                                        ambiguous_in_round += ambiguous_count
                                        modified_in_round += modified_count
                                except CancelledError:
                                    pass
                                except Exception as e:
                                    print(f"[步骤3错误] 多进程处理失败: {e}")
                                    raise

            total_empty_assigned += empty_assigned_in_round
            total_ambiguous += ambiguous_in_round
            total_modified += modified_in_round

            update_progress(
                35 + (round_num + 1) / len(rounds) * 50,
                f"第 {round_num + 1} 轮完成: 修改 {modified_in_round} 个单元格"
            )

        # 保存最终结果
        update_progress(95, "保存最终结果...")
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        # ★ 关键修复：确保所有数值单元格保持整数格式，避免浮点数精度问题
        def _format_cell(cell):
            """格式化单元格值：将浮点数格式的字符串转换为整数格式"""
            if not isinstance(cell, str):
                cell = str(cell)
            cell = cell.strip()
            if not cell:
                return cell
            # 检查是否为纯数字（可能包含小数点）
            try:
                # 尝试转换为float再转int，去除不必要的小数部分
                float_val = float(cell)
                int_val = int(float_val)
                # 如果转换后值相等，使用整数格式
                if float_val == int_val:
                    return str(int_val)
                # 否则保留原始值（可能是真正的浮点数）
                return cell
            except (ValueError, OverflowError):
                # 非数字值，保持原样
                return cell
        
        # 格式化所有数据行
        formatted_data = []
        for row in modified_data:
            formatted_row = [_format_cell(cell) for cell in row]
            formatted_data.append(formatted_row)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(formatted_data)

        update_progress(100, "匹配完成!")

        return {
            "success": True,
            "output_file": output_file,
            "total_cells": total_cells,
            "total_sequences": total_sequences,
            "total_modified": total_modified,
            "total_empty_assigned": total_empty_assigned,
            "total_ambiguous": total_ambiguous,
        }

    except _Step3PausedSignal:
        # 步骤3被用户暂停（多进程执行中途）
        # 返回特殊标志，让 pipeline 的主循环感知并执行暂停流程
        print("[步骤3] 暂停信号已传递给 pipeline 主循环", flush=True)
        return {"success": False, "paused": True, "error": "用户暂停（步骤3执行中）"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
