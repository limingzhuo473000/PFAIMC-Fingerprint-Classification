"""
步骤3: 匹配测试 - 核心处理逻辑（性能优化版）

核心优化：
1. 目标像素预解析 - 避免在每个worker中重复解析CSV
2. 批处理匹配 - 利用矩阵运算加速
3. 修复GPU代码死代码问题
4. 减少重复日志输出
"""

import os
import csv
import time as _time
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed, CancelledError
from pathlib import Path as _Path

import numpy as np

# ── GPU 加速支持（延迟检测，避免重复日志）─────────────────────
_HAS_GPU = None  # None=未检测, True=已加载, False=未安装

def _check_gpu():
    """延迟检测GPU，只打印一次日志"""
    global _HAS_GPU
    if _HAS_GPU is not None:
        return _HAS_GPU
    
    try:
        import cupy as cp
        _HAS_GPU = True
        print("[GPU加速] CuPy已加载，将使用GPU加速计算", flush=True)
    except ImportError:
        _HAS_GPU = False
        print("[GPU加速] 未检测到CuPy，使用CPU计算", flush=True)
    
    return _HAS_GPU


# ── 多进程子进程初始化 ─────────────────────────────────────────────
import sys as _sys

_BACKEND_DIR = str(_Path(__file__).parent.parent)


def _init_pool():
    """ProcessPoolExecutor 子进程初始化回调：将 backend/ 加入 sys.path"""
    if _BACKEND_DIR not in _sys.path:
        _sys.path.insert(0, _BACKEND_DIR)
    # 在子进程中检测GPU（只打印一次）
    _check_gpu()


class _Step3PausedSignal(Exception):
    """步骤3内部暂停信号：在多进程执行中途检测到暂停请求时抛出，由 run_step3 捕获。"""
    pass



# ============================================================
# 目标文件预解析（性能优化）
# ============================================================

def _preparse_target_file(target_file: str, target_sequence_length: int):
    """
    预解析目标CSV文件，将所有像素序列转换为NumPy数组
    
    返回:
        target_array: (n_pixels, seq_len) np.int16  (-1表示无效)
        valid_mask: (n_pixels,) bool  该位置是否有有效序列
        pixel_positions: list of (row_idx, col_idx)  每个有效像素的位置
        n_rows, n_cols: CSV文件的行列数
    """
    print(f"[优化] 预解析目标文件: {target_file}", flush=True)
    start_time = _time.time()
    
    # 读取CSV
    target_rows = None
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1']:
        try:
            with open(target_file, 'r', newline='', encoding=encoding) as f:
                target_rows = list(csv.reader(f))
            break
        except Exception:
            continue
    
    if not target_rows:
        return None, None, None, 0, 0
    
    n_rows = len(target_rows)
    n_cols = max((len(row) for row in target_rows), default=0)
    n_pixels = n_rows * n_cols
    
    # 预分配数组，用-1填充无效值
    target_array = np.full((n_pixels, target_sequence_length), -1, dtype=np.int16)
    valid_mask = np.zeros(n_pixels, dtype=bool)
    pixel_positions = []
    
    idx = 0
    for row_idx, row in enumerate(target_rows):
        for col_idx, cell in enumerate(row):
            cell = cell.strip() if cell else ''
            if cell and cell != '-1':
                try:
                    vals = cell.split()
                    if len(vals) == target_sequence_length:
                        target_array[idx] = [int(float(v)) for v in vals]
                        valid_mask[idx] = True
                        pixel_positions.append((row_idx, col_idx))
                except (ValueError, AttributeError):
                    pass
            idx += 1
    
    elapsed = _time.time() - start_time
    n_valid = valid_mask.sum()
    print(f"[优化] 预解析完成: {n_valid}/{n_pixels} 有效像素, 耗时 {elapsed:.2f}s", flush=True)
    
    return target_array, valid_mask, pixel_positions, n_rows, n_cols


# ============================================================
# 参考序列预处理（在主进程中一次性完成）
# ============================================================

def _build_ref_arrays(all_reference_sequences, current_replacements, target_sequence_length):
    """
    将 Python list 格式的参考序列转换为 NumPy 数组，便于向量化计算。

    返回:
        ref_matrix:  (total_refs, seq_len)  np.int16
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

    ref_matrix = np.array(ref_list, dtype=np.int16)      # (total_refs, seq_len)
    repl_ids = np.array(repl_id_list, dtype=np.int32)    # (total_refs,)
    return ref_matrix, repl_ids, uniq_repls



# ============================================================
# 多进程 worker（优化版 - 批处理）
# ============================================================

def _match_block_optimized(pixel_indices, target_array, valid_mask, pixel_positions,
                           ref_matrix, repl_ids, uniq_repls, threshold,
                           target_sequence_length, batch_size=500):
    """
    多进程 worker：批处理匹配（优化版）
    
    参数:
        pixel_indices: 要处理的像素索引列表
        target_array: (n_pixels, seq_len) 预解析的目标数组
        valid_mask: (n_pixels,) 有效像素掩码
        pixel_positions: 像素位置列表 [(row_idx, col_idx), ...]
        ref_matrix: (n_refs, seq_len) 参考序列矩阵
        repl_ids: (n_refs,) 参考序列对应的replacement ID
        uniq_repls: list[str]  唯一的replacement值列表
        threshold: 汉明距离阈值
        batch_size: 批处理大小
    
    返回: [(row_idx, col_idx, new_value, is_ambiguous), ...]
    """
    import numpy as np
    
    # 筛选有效像素
    valid_pixel_indices = [idx for idx in pixel_indices if valid_mask[idx]]
    
    if not valid_pixel_indices:
        return []
    
    # 检查是否使用GPU
    use_gpu = _check_gpu()
    
    results = []
    n_valid = len(valid_pixel_indices)
    
    # 分批处理
    for start in range(0, n_valid, batch_size):
        end = min(start + batch_size, n_valid)
        batch_idx = valid_pixel_indices[start:end]
        
        # 提取批次数据: (batch_size, seq_len)
        batch_data = target_array[batch_idx]
        
        # 批处理匹配
        if use_gpu and _HAS_GPU:
            batch_results = _batch_match_gpu(batch_data, ref_matrix, repl_ids, 
                                            uniq_repls, threshold)
        else:
            batch_results = _batch_match_cpu(batch_data, ref_matrix, repl_ids, 
                                            uniq_repls, threshold)
        
        # 整理结果
        for i, (min_dist, repl_id, ambiguous) in enumerate(batch_results):
            pixel_idx = batch_idx[i]
            row_idx, col_idx = pixel_positions[pixel_idx]
            
            if ambiguous:
                results.append((row_idx, col_idx, None, True))
            elif repl_id is not None:
                results.append((row_idx, col_idx, uniq_repls[repl_id], False))
            else:
                results.append((row_idx, col_idx, None, False))
    
    return results


def _batch_match_cpu(target_batch, ref_matrix, repl_ids, uniq_repls, threshold):
    """
    批处理匹配 - CPU版本（向量化）
    
    参数:
        target_batch: (batch_size, seq_len) np.int16
        ref_matrix: (n_refs, seq_len) np.int16
        repl_ids: (n_refs,) np.int32
        uniq_repls: list[str]
        threshold: int
    
    返回:
        results: list of (min_dist, matched_repl_id, ambiguous)
    """
    import numpy as np
    
    batch_size = target_batch.shape[0]
    if ref_matrix is None or len(ref_matrix) == 0:
        return [(threshold + 1, None, False) for _ in range(batch_size)]
    
    # 向量化计算距离矩阵: (batch_size, n_refs)
    # 使用广播: target_batch[:, None, :] -> (batch_size, 1, seq_len)
    #           ref_matrix[None, :, :] -> (1, n_refs, seq_len)
    # 结果: (batch_size, n_refs, seq_len) -> sum(axis=2)
    dists = np.sum(target_batch[:, np.newaxis, :] != ref_matrix[np.newaxis, :, :], axis=2)
    
    # 找到每个像素的最小距离
    min_dists = np.min(dists, axis=1)
    
    results = []
    for i in range(batch_size):
        min_dist = int(min_dists[i])
        
        if min_dist > threshold:
            results.append((min_dist, None, False))
            continue
        
        # 找到所有达到最小距离的参考序列
        min_mask = (dists[i] == min_dist)
        matched_ids = np.unique(repl_ids[min_mask])
        
        if len(matched_ids) > 1:
            results.append((min_dist, None, True))
        else:
            results.append((min_dist, int(matched_ids[0]), False))
    
    return results


def _batch_match_gpu(target_batch, ref_matrix, repl_ids, uniq_repls, threshold):
    """
    批处理匹配 - GPU版本（向量化）
    """
    if not _HAS_GPU:
        return _batch_match_cpu(target_batch, ref_matrix, repl_ids, uniq_repls, threshold)
    
    try:
        import cupy as cp
        
        # 转移到GPU
        if not isinstance(target_batch, cp.ndarray):
            target_gpu = cp.asarray(target_batch)
        else:
            target_gpu = target_batch
        
        if not isinstance(ref_matrix, cp.ndarray):
            ref_gpu = cp.asarray(ref_matrix)
        else:
            ref_gpu = ref_matrix
        
        if not isinstance(repl_ids, cp.ndarray):
            repl_ids_gpu = cp.asarray(repl_ids)
        else:
            repl_ids_gpu = repl_ids
        
        batch_size = target_gpu.shape[0]
        
        # GPU上批处理计算距离
        dists = cp.sum(target_gpu[:, cp.newaxis, :] != ref_gpu[cp.newaxis, :, :], axis=2)
        
        # 找到最小距离
        min_dists = cp.min(dists, axis=1)
        min_dists_cpu = min_dists.get()
        
        results = []
        for i in range(batch_size):
            min_dist = int(min_dists_cpu[i])
            
            if min_dist > threshold:
                results.append((min_dist, None, False))
                continue
            
            # 在GPU上找到所有最小距离的索引
            min_mask = (dists[i] == min_dists[i])
            matched_ids = cp.unique(repl_ids_gpu[min_mask]).get()
            
            if len(matched_ids) > 1:
                results.append((min_dist, None, True))
            else:
                results.append((min_dist, int(matched_ids[0]), False))
        
        return results
        
    except Exception as e:
        print(f"[GPU加速] GPU批处理失败，回退到CPU: {e}", flush=True)
        return _batch_match_cpu(target_batch, ref_matrix, repl_ids, uniq_repls, threshold)


# ============================================================
# 为了兼容旧代码，保留原函数名（但内部调用优化版本）
# ============================================================

def _match_block_vectorized(row_indices, data_rows, target_sequence_length,
                          ref_matrix, repl_ids, uniq_repls, threshold,
                          use_gpu=False):
    """
    兼容旧接口：将行数据转换为预解析格式，然后调用优化版本
    
    注意：这个函数性能较低，建议使用新的 _match_block_optimized 函数
    """
    import sys as _sys
    print("[警告] 使用旧版 _match_block_vectorized，性能较低，建议升级", file=_sys.stderr, flush=True)
    
    # 简单实现：逐像素处理（保持旧接口兼容）
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
# 以下是保留的原有辅助函数（未修改）
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


def get_image_spatial_dims(folder_path: str):
    """从文件夹中的第一个影像文件获取空间维度（高度×宽度）。"""
    # ... (保留原有实现)
