"""
步骤2: 样本进阶 - 核心处理逻辑

从 02样本进阶.py 提取封装
"""

import os
import sys as _sys
import gc
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional
from glob import glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import chardet

# ---- 多进程池初始化 ----
_BACKEND_DIR = str(Path(__file__).parent.parent)

def _init_pool():
    """ProcessPoolExecutor 子进程初始化：将 backend/ 加入 sys.path"""
    if _BACKEND_DIR not in _sys.path:
        _sys.path.insert(0, _BACKEND_DIR)

_MAX_WORKERS = 8


# ============================================================
# 辅助函数
# ============================================================

def detect_encoding(file_path: str) -> str:
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(50000)
        result = chardet.detect(raw_data)
    return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'


def count_tif_files(tif_folder: str) -> int:
    """统计TIF文件数量"""
    tif_files = sorted(f for f in os.listdir(tif_folder) if f.lower().endswith(('.tif', '.tiff')))
    return len(tif_files)


def create_temp_dir(base_dir: str = None) -> str:
    """创建临时目录，支持D盘优先+自动回退"""
    candidates = []
    if base_dir is not None:
        candidates.append(base_dir)
    else:
        # 优先D盘，回退到E盘output/temp，最后系统temp
        candidates = [
            r"D:\temp_sample_processing",
            r"E:\output\temp_sample_processing",
            os.path.join(tempfile.gettempdir(), "temp_sample_processing"),
        ]

    last_err = None
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            timestamp = int(datetime.now().timestamp() * 1000) % 1000000
            temp_dir = os.path.join(d, f"temp_{timestamp}")
            os.makedirs(temp_dir, exist_ok=True)
            print(f"[create_temp_dir] 使用: {temp_dir}")
            return temp_dir
        except Exception as e:
            last_err = e
            print(f"[create_temp_dir] {d} 失败: {type(e).__name__}: {e}，尝试下一个...")
            continue

    raise RuntimeError(
        f"无法创建临时目录。尝试了: {candidates}。最后一个错误: {type(last_err).__name__}: {last_err}"
    )


def cleanup_temp_dir(temp_dir: str):
    """清理临时目录"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception:
        pass


def _extract_single_shp(args):
    """提取单个 SHP 文件的像素坐标（顶层，可被多进程序列化）"""
    shp_path, raster_path, output_dir, batch_size, global_diag_dir = args
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.transform import rowcol as _rowcol
    except ImportError as e:
        raise RuntimeError(
            f"地理空间库导入失败: {e}。请确保已安装 geopandas、rasterio、shapely、pyproj。"
            f"运行: pip install geopandas rasterio shapely pyproj"
        ) from e

    class_name = os.path.splitext(os.path.basename(shp_path))[0]
    output_csv = os.path.join(output_dir, f"{class_name}.csv")

    try:
        with rasterio.open(raster_path) as src:
            transform = src.transform
            band = src.read(1)
            nodata = src.nodata
            height, width = band.shape
            crs_raster = src.crs

        gdf = gpd.read_file(shp_path)
        if len(gdf) == 0:
            _diag = f"[步骤2 诊断] {class_name}: SHP 文件包含0个要素，将生成空CSV"
            print(_diag, flush=True)
            _write_diag_file(output_dir, _diag)
            # 写入空CSV（仅表头）
            empty_df = pd.DataFrame(columns=['pixel_value','pixel_row','pixel_col','point_x','point_y'])
            empty_df.to_csv(output_csv, index=False, encoding='utf-8')
            return True, _diag

        # ═══════════════════════════════════════════════════════
        # ★ 关键修复：SHP → 栅格 CRS 坐标系重投影 ★
        # 如果SHP和栅格坐标系不同，必须先转换，否则 rowcol() 全落在范围外
        # ═══════════════════════════════════════════════════════
        _shp_crs_orig = gdf.crs
        if _shp_crs_orig is None:
            # SHP 没有定义 CRS，无法自动转换，假设与栅格一致
            _crs_msg = f"SHP_CRS=未定义(假设与栅格一致), 栅格CRS={crs_raster}"
        elif _shp_crs_orig != crs_raster:
            try:
                gdf = gdf.to_crs(crs_raster)
                _crs_msg = f"SHP_CRS_原始={_shp_crs_orig}, 栅格CRS={crs_raster}, 已重投影({len(gdf)}要素)"
            except Exception as _crserr:
                _crs_msg = f"SHP_CRS={_shp_crs_orig}, 栅格CRS={crs_raster}, 重投影失败: {_crserr}"
        else:
            _crs_msg = f"SHP_CRS={_shp_crs_orig}, 栅格CRS={crs_raster}, 坐标系一致(无需转换)"

        # ── 诊断：SHP基本信息 ──
        _diag = (
            f"[步骤2 诊断] {class_name}: {len(gdf)}要素, "
            f"geom_type={set(str(g.geom_type) for g in gdf.geometry[:5])}, "
            f"{_crs_msg}, 栅格尺寸={width}×{height}"
        )
        print(_diag, flush=True)
        _write_diag_file(output_dir, _diag)

        original_count = len(gdf)
        pixel_values = [None] * original_count
        pixel_rows = [-1] * original_count
        pixel_cols = [-1] * original_count
        point_x = [0.0] * original_count
        point_y = [0.0] * original_count

        for i in range(0, original_count, batch_size):
            end_idx = min(i + batch_size, original_count)
            batch_geoms = gdf.geometry.iloc[i:end_idx]
            for j, geom in enumerate(batch_geoms):
                idx_geom = i + j
                try:
                    # 支持点(Point)和多边形(Polygon/MultiPolygon)要素
                    if geom.geom_type == 'Point':
                        x, y = geom.x, geom.y
                    elif hasattr(geom, 'centroid'):
                        x, y = geom.centroid.x, geom.centroid.y
                    else:
                        # 兜底：尝试直接访问（兼容部分shapely版本）
                        x, y = geom.x, geom.y
                except Exception:
                    # 无法提取坐标，跳过
                    continue
                point_x[idx_geom] = x
                point_y[idx_geom] = y
                row, col = _rowcol(transform, x, y)
                pixel_rows[idx_geom] = row
                pixel_cols[idx_geom] = col
                if 0 <= row < height and 0 <= col < width:
                    val = band[row, col]
                    is_nodata = False
                    if nodata is not None:
                        _nod = float(nodata)
                        # ★ 2026-09-30 修复：数值型 nodata 与合法 0 值冲突
                        #   旧写法 np.isclose(float(val), float(nodata)) 在 nodata=0.0 时
                        #   会把合法的 0 值（如 NDBI 恰好等于 0，即 NIR==SWIR1）误判为
                        #   NoData 而剔除，导致该样本点序列长度少 1 位（199≠200）
                        #   被 filter_by_sequence_length 整体丢弃。
                        #   实测影响：评价样本人工表面 52 → 51 点。
                        #   修法：仅当 nodata 是显式 NaN，或 val 本身为 NaN 时才判 NaN；
                        #   数值型 nodata 只在 val 与其"非零"相等时才算命中。
                        if np.isnan(_nod):
                            is_nodata = bool(np.isnan(float(val)))
                        elif np.isnan(float(val)):
                            is_nodata = False   # val 为 NaN 由后续序列长度过滤处理
                        else:
                            is_nodata = (float(val) == _nod) and (_nod != 0.0)
                    if not is_nodata:
                        pixel_values[idx_geom] = int(val)

        # ── 诊断：提取统计 ──
        _valid_pixels = sum(1 for v in pixel_values if v is not None)
        _valid_rows_count = sum(1 for r in pixel_rows if r >= 0)
        _valid_cols_count = sum(1 for c in pixel_cols if c >= 0)
        _diag_extract = (
            f"[步骤2 诊断] {class_name}: 像素提取 — 总要素={original_count}, "
            f"有效像素值={_valid_pixels}, 有效行坐标={_valid_rows_count}, 有效列坐标={_valid_cols_count}, "
            f"nodata={nodata}, 栅格波段1范围=[{band.min()},{band.max()}]"
        )
        print(_diag_extract, flush=True)
        _write_diag_file(output_dir, _diag_extract)
        if global_diag_dir:
            _write_diag_file(global_diag_dir, _diag_extract)

        # ── 额外诊断：如果有效像素为0，打印前3个坐标的波段1值 ──
        if _valid_pixels == 0 and original_count > 0:
            _sample_details = []
            for _k in range(min(3, original_count)):
                _r, _c = pixel_rows[_k], pixel_cols[_k]
                if 0 <= _r < height and 0 <= _c < width:
                    _raw_val = band[_r, _c]
                    _sample_details.append(f"({pixel_rows[_k]},{pixel_cols[_k]}) band1={_raw_val}")
            _diag_extra = f"[步骤2 诊断] {class_name}: 有效像素=0! 采样: {'; '.join(_sample_details) if _sample_details else '无有效坐标'}"
            print(_diag_extra, flush=True)
            _write_diag_file(output_dir, _diag_extra)
            if global_diag_dir:
                _write_diag_file(global_diag_dir, _diag_extra)

        result_df = pd.DataFrame({
            'pixel_value': pixel_values,
            'pixel_row': pixel_rows,
            'pixel_col': pixel_cols,
            'point_x': point_x,
            'point_y': point_y
        })

        for col in gdf.columns:
            if col != 'geometry':
                result_df[col] = gdf[col].values

        # ── 收集诊断摘要 ──
        _diag_summary = (
            f"{class_name}: 要素={original_count}, 有效像素={_valid_pixels}, "
            f"有效行={_valid_rows_count}, 有效列={_valid_cols_count}, "
            f"行范围=[{min(pixel_rows) if _valid_rows_count else 'N/A'},"
            f"{max(pixel_rows) if _valid_rows_count else 'N/A'}], "
            f"列范围=[{min(pixel_cols) if _valid_cols_count else 'N/A'},"
            f"{max(pixel_cols) if _valid_cols_count else 'N/A'}]"
        )

        result_df.to_csv(output_csv, index=False, encoding='utf-8')
        del gdf, result_df
        return True, _diag_summary

    except Exception as e:
        _diag_err = f"处理 {class_name} 失败: {type(e).__name__}: {e}"
        print(_diag_err, flush=True)
        return False, _diag_err


def extract_points_with_pixel_location_batch(
    shp_input: str,
    raster_path: str,
    output_dir: str,
    batch_size: int = 10000,
    progress_callback=None,
    global_diag_dir: str = None
) -> bool:
    """批量提取样本点对应影像像素值及行列位置（多进程并行）

    global_diag_dir: 额外诊断输出目录（不会被清理），写入 step2_diag.log
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取SHP文件列表
    if os.path.isfile(shp_input) and shp_input.lower().endswith('.shp'):
        shp_files = [shp_input]
    elif os.path.isdir(shp_input):
        shp_files = [os.path.join(shp_input, f) for f in os.listdir(shp_input) if f.lower().endswith('.shp')]
    else:
        raise ValueError(
            f"SHP输入无效: [{shp_input}] 不是有效的SHP文件或文件夹。"
            f"请确认填写的是 .shp  shapefile 文件路径，而不是 CSV 或其他格式。"
        )

    if not shp_files:
        raise ValueError(
            f"在 [{shp_input}] 中未找到任何 .shp 文件。"
            f"请确认该文件夹内包含有效的 shapefile 样本文件。"
        )

    args_list = [(shp, raster_path, output_dir, batch_size, global_diag_dir) for shp in shp_files]
    total = len(args_list)
    done = 0
    diag_messages = []

    with ProcessPoolExecutor(max_workers=_MAX_WORKERS, initializer=_init_pool) as ex:
        futures = {ex.submit(_extract_single_shp, args): args for args in args_list}
        for fut in as_completed(futures):
            try:
                result = fut.result()
                if isinstance(result, tuple) and len(result) == 2:
                    ok, diag = result
                    diag_messages.append(diag)
                else:
                    ok = result
            except Exception as e:
                diag_messages.append(f"SHP提取任务异常: {type(e).__name__}: {e}")
            done += 1
            if progress_callback:
                progress_callback(done / total * 100)

    # ── 汇总诊断 ──
    if diag_messages:
        summary = "\n".join(f"  [{i+1}] {msg}" for i, msg in enumerate(diag_messages))
        print(f"[步骤2 诊断汇总] SHP提取完成 {total} 个文件:\n{summary}", flush=True)
        # 同时写入文件，便于事后查看
        try:
            diag_file = os.path.join(output_dir, "step2_shp_diag.log")
            with open(diag_file, "w", encoding="utf-8") as _df:
                _df.write(summary)
        except Exception:
            pass

    return True


def parse_cell_value_keep_integers(cell_value):
    """解析单元格值，保持整数格式"""
    if pd.isna(cell_value) or cell_value == '':
        return None
    cell_str = str(cell_value).strip()
    if ' ' in cell_str:
        return cell_str
    else:
        try:
            return str(int(float(cell_str)))
        except ValueError:
            return None


def _crop_cell_sequence(cell_value: str, target_seq_len: int) -> str:
    """将单元格中的多值序列裁剪到目标长度"""
    if cell_value is None:
        return None
    parts = str(cell_value).split(' ')
    if len(parts) <= target_seq_len:
        return cell_value
    return ' '.join(parts[:target_seq_len])


def load_experiment_data_as_matrix(csv_path: str, target_shape: tuple = None, target_seq_len: int = None):
    """加载实验数据并解析多值序列

    自动检测格式：
    - 标准格式（Step1输出）：每个单元格包含空格分隔的序列 "v1 v2 ... vN"
    - 带偏移格式：前几行/列为空元数据，实际数据从某行某列开始

    参数:
        csv_path: CSV 文件路径
        target_shape: (target_rows, target_cols) — 如果提供，将 CSV 裁剪/填充到此尺寸
        target_seq_len: 每个单元格的序列长度目标 — 如果提供，截断过长序列
    """
    encoding = detect_encoding(csv_path)
    df = pd.read_csv(csv_path, header=None, dtype=str)

    raw_shape = df.shape
    _total_cells = raw_shape[0] * raw_shape[1]

    # ── 直接使用原始坐标构建解析后的数据矩阵 ──
    # 注意：不再做偏移检测/校正。偏移会导致坐标系不匹配：
    # load_experiment 用偏移后的坐标存数据，但 extract_values_from_parsed_matrix
    # 用原始影像坐标查询，两者不匹配，导致样本提取全为None。
    # 步骤1输出的CSV本身就是与影像1:1对应的矩阵，无需偏移。
    parsed_data = {}
    effective_rows, effective_cols = raw_shape

    for i in range(raw_shape[0]):
        for j in range(raw_shape[1]):
            cell_value = df.iloc[i, j]
            parsed_val = parse_cell_value_keep_integers(cell_value)
            # 序列裁剪（如果目标长度小于实际长度）
            if parsed_val is not None and target_seq_len is not None and ' ' in str(parsed_val):
                parsed_val = _crop_cell_sequence(parsed_val, target_seq_len)
            # 使用原始坐标作为key，与影像坐标系保持一致
            parsed_data[(i, j)] = parsed_val

    effective_shape = raw_shape

    # ── ★ 尺寸适配：裁剪/填充到 target_shape ──
    if target_shape is not None:
        target_rows, target_cols = target_shape
        current_rows, current_cols = effective_shape

        crop_msg_parts = []
        adapted_data = {}

        # 确定适配后的行列范围（取交集 → 裁剪 + 拓展边界 → 填充）
        max_r = min(current_rows, target_rows)
        max_c = min(current_cols, target_cols)

        # 复制在交集范围内的数据
        for (r, c), v in parsed_data.items():
            if 0 <= r < max_r and 0 <= c < max_c:
                adapted_data[(r, c)] = v

        if current_rows > target_rows or current_cols > target_cols:
            crop_msg_parts.append(
                f"裁剪: {current_rows}×{current_cols} → {max_r}×{max_c}"
                f"（削减行={current_rows - max_r}, 列={current_cols - max_c}）"
            )
        if current_rows < target_rows or current_cols < target_cols:
            crop_msg_parts.append(
                f"有效区域 {current_rows}×{current_cols}"
                f" < 目标 {target_rows}×{target_cols}，不足部分用None填充"
            )

        if crop_msg_parts:
            print(
                f"[load_experiment] ★ 尺寸适配: {'; '.join(crop_msg_parts)}",
                flush=True,
            )

        parsed_data = adapted_data
        effective_shape = target_shape

    # ── 格式验证 ──
    # 统计有效数据单元格数和序列长度分布
    _valid_cells = 0
    _seq_lengths = []
    _sample_coords = []
    for (r, c), v in list(parsed_data.items())[:min(500, len(parsed_data))]:
        if v is not None:
            _valid_cells += 1
            if ' ' in str(v):
                _seq_lengths.append(len(str(v).split()))
                if len(_sample_coords) < 3:
                    _sample_coords.append((r, c, len(str(v).split())))

    if _seq_lengths:
        _avg_len = sum(_seq_lengths) / len(_seq_lengths)
        _min_len = min(_seq_lengths)
        _max_len = max(_seq_lengths)
        print(
            f"[load_experiment] 格式检查: 有效单元格={_valid_cells}/{_total_cells}, "
            f"序列长度 avg={_avg_len:.1f} range=[{_min_len}, {_max_len}], "
            f"有效矩阵尺寸={effective_shape}",
            flush=True,
        )

    return parsed_data, effective_shape


def extract_values_from_parsed_matrix(coords_df, parsed_data, matrix_shape, required_length, raw_df=None, output_dir=None, class_name=None):
    """从解析后的数据中提取值（含邻近像素fallback + 诊断）

    raw_df: 原始 DataFrame（用于诊断打印单元格内容）
    output_dir: 诊断日志输出目录
    class_name: 类别名（用于诊断）
    """
    total_points = len(coords_df)
    extracted = [None] * total_points
    fallback_used = [False] * total_points
    rows = coords_df['pixel_row'].values
    cols = coords_df['pixel_col'].values

    _none_cell_samples = []  # 收集None单元格样本用于诊断
    _neighbor_found = 0

    def _neighbor_search(r0, c0, max_radius=5):
        """在 (r0,c0) 半径 max_radius 范围内搜索有效序列"""
        for radius in range(1, max_radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r0 + dr, c0 + dc
                    if not (0 <= nr < matrix_shape[0] and 0 <= nc < matrix_shape[1]):
                        continue
                    pv = parsed_data.get((nr, nc), None)
                    if pv is not None and ' ' in pv:
                        parts = pv.split(' ')
                        if len(parts) == required_length:
                            return pv, (nr, nc), radius
        return None, None, None

    for i in range(total_points):
        r = int(rows[i]) if not pd.isna(rows[i]) else -1
        c = int(cols[i]) if not pd.isna(cols[i]) else -1
        if 0 <= r < matrix_shape[0] and 0 <= c < matrix_shape[1]:
            parsed_val = parsed_data.get((r, c), None)
            if parsed_val is not None:
                if ' ' in parsed_val:
                    parts = parsed_val.split(' ')
                    if len(parts) == required_length:
                        extracted[i] = parsed_val
                    # 长度不匹配 → 不提取（会在后续诊断统计）
                else:
                    extracted[i] = parsed_val
            else:
                # ── 单元格为空/None → 尝试邻近搜索 ──
                nb_val, nb_pos, nb_radius = _neighbor_search(r, c, max_radius=5)
                if nb_val is not None:
                    extracted[i] = nb_val
                    fallback_used[i] = True
                    _neighbor_found += 1
                else:
                    # 收集前5个None样本用于诊断
                    if raw_df is not None and len(_none_cell_samples) < 5:
                        try:
                            raw_cell = raw_df.iloc[r, c]
                            _none_cell_samples.append(f"(r={r},c={c}): raw={repr(raw_cell)[:120]}")
                        except Exception:
                            _none_cell_samples.append(f"(r={r},c={c}): 无法读取")

    # ── 诊断输出 ──
    if output_dir and class_name:
        _diag_lines = []
        if _none_cell_samples:
            _diag_lines.append(
                f"[步骤2 诊断] {class_name}: 空单元格样本 — {'; '.join(_none_cell_samples)}"
            )
        if _neighbor_found > 0:
            _diag_lines.append(
                f"[步骤2 诊断] {class_name}: 邻近像素fallback已激活 — {_neighbor_found}/{total_points} 个点通过邻近搜索获得有效序列"
            )
        if not _none_cell_samples and _neighbor_found == 0:
            # 没有None也没有fallback → 正常
            pass
        for line in _diag_lines:
            print(line, flush=True)
            _write_diag_file(output_dir, line)

    return extracted


def filter_by_sequence_length(df, sequence_column, required_length):
    """按序列长度过滤样本"""
    def count_sequence_length(sequence_str):
        if pd.isna(sequence_str) or sequence_str == '':
            return 0
        parts = str(sequence_str).split(' ')
        parts = [part.strip() for part in parts if part.strip()]
        return len(parts)

    df['sequence_length'] = df[sequence_column].apply(count_sequence_length)
    filtered_df = df[df['sequence_length'] == required_length].copy()
    filtered_df = filtered_df.drop(columns=['sequence_length'])
    return filtered_df


def process_samples_optimized(
    sample_coords_path: str,
    experiment_data_path: str,
    output_dir: str,
    required_length: int,
    target_shape: tuple = None,
):
    """优化处理样本
    
    参数:
        target_shape: (rows, cols) 目标影像尺寸，用于自动裁切/填充CSV矩阵
    """
    base_name = os.path.splitext(os.path.basename(sample_coords_path))[0]
    filtered_output = os.path.join(output_dir, f"{base_name}.csv")

    coords_df = pd.read_csv(sample_coords_path)
    if len(coords_df) == 0:
        empty_df = pd.DataFrame(columns=coords_df.columns.tolist() if len(coords_df.columns) > 0 else ['pixel_value','pixel_row','pixel_col','point_x','point_y'])
        empty_df.to_csv(filtered_output, index=False, encoding='utf-8-sig')
        _diag = f"[步骤2 诊断] {base_name}: coords_df 为空（SHP提取未产生任何有效点），输出空CSV（仅表头）"
        print(_diag, flush=True)
        _write_diag_file(output_dir, _diag)
        return True, [], _diag

    # ── ★ 尺寸诊断：打印 CSV 原始尺寸和目标尺寸 ──
    _raw_df_for_shape = None
    try:
        _enc = detect_encoding(experiment_data_path)
        _raw_df_for_shape = pd.read_csv(experiment_data_path, header=None, dtype=str, encoding=_enc)
        _raw_shape = _raw_df_for_shape.shape
        print(
            f"[load_experiment] CSV 原始尺寸={_raw_shape[0]}×{_raw_shape[1]}, "
            f"目标尺寸={target_shape}, 目标序列长度={required_length}",
            flush=True,
        )
    except Exception:
        pass

    # 加载实验数据（传入目标尺寸和序列长度）
    parsed_data, matrix_shape = load_experiment_data_as_matrix(
        experiment_data_path,
        target_shape=target_shape,
        target_seq_len=required_length,
    )

    # ── 同时加载原始 DataFrame（用于诊断单元格内容）──
    _raw_df = None
    try:
        _encoding_raw = detect_encoding(experiment_data_path)
        _raw_df = pd.read_csv(experiment_data_path, header=None, dtype=str, encoding=_encoding_raw)
    except Exception:
        pass

    # ── 诊断：坐标范围与矩阵尺寸对比 ──
    _sample_rows = coords_df['pixel_row'].values
    _sample_cols = coords_df['pixel_col'].values
    _valid_rows = [r for r in _sample_rows if not pd.isna(r) and r >= 0]
    _valid_cols = [c for c in _sample_cols if not pd.isna(c) and c >= 0]
    _in_bounds = sum(
        1 for r, c in zip(_sample_rows, _sample_cols)
        if (not pd.isna(r) and not pd.isna(c) and
            0 <= int(r) < matrix_shape[0] and 0 <= int(c) < matrix_shape[1])
    )
    _diag_coords = (
        f"[步骤2 诊断] {base_name}: coords_df行数={len(coords_df)}, "
        f"矩阵尺寸={matrix_shape[0]}×{matrix_shape[1]}, "
        f"有效行坐标={len(_valid_rows)}(范围{min(_valid_rows) if _valid_rows else 'N/A'}~{max(_valid_rows) if _valid_rows else 'N/A'}), "
        f"有效列坐标={len(_valid_cols)}(范围{min(_valid_cols) if _valid_cols else 'N/A'}~{max(_valid_cols) if _valid_cols else 'N/A'}), "
        f"在矩阵范围内={_in_bounds}/{len(coords_df)}"
    )
    print(_diag_coords, flush=True)
    _write_diag_file(output_dir, _diag_coords)

    # 提取值
    extracted_values = extract_values_from_parsed_matrix(
        coords_df, parsed_data, matrix_shape, required_length,
        raw_df=_raw_df, output_dir=output_dir, class_name=base_name
    )
    coords_df['提取值'] = extracted_values

    # ── ★ 关键验证：如果全部提取为None，给出明确原因 ──
    _extracted_none = sum(1 for v in extracted_values if v is None)
    if _extracted_none == len(extracted_values) and len(extracted_values) > 0:
        _diag_fatal = (
            f"[步骤2 诊断] {base_name}: ⚠️ 致命错误 — 所有样本点提取值均为None！\n"
            f"  这通常意味着「已有步骤1结果CSV」的格式与当前影像不匹配：\n"
            f"  - 影像尺寸: {matrix_shape[0]}×{matrix_shape[1]}, 需要序列长度={required_length}\n"
            f"  - CSV文件: {experiment_data_path}\n"
            f"  - 样本坐标范围: rows[{min(_valid_rows)}~{max(_valid_rows)}], cols[{min(_valid_cols)}~{max(_valid_cols)}]\n"
            f"\n"
            f"  解决方案:\n"
            f"  1. 清空「已有步骤1结果CSV」输入框，让系统自动运行Step1重新生成\n"
            f"  2. 或确保填写的CSV是当前影像文件夹({required_length}个TIF)的正确Step1输出"
        )
        print(_diag_fatal, flush=True)
        _write_diag_file(output_dir, _diag_fatal)
        # 写入特殊标记文件供上层检测
        _fatal_marker = os.path.join(output_dir, f"_FATAL_FORMAT_MISMATCH_{base_name}.txt")
        with open(_fatal_marker, 'w', encoding='utf-8') as _fm:
            _fm.write(_diag_fatal)

    # 过滤
    filtered_df = filter_by_sequence_length(coords_df, '提取值', required_length)

    # ── 诊断：提取结果统计 ──
    _extracted_none = sum(1 for v in extracted_values if v is None)
    _extracted_single = sum(1 for v in extracted_values if v is not None and ' ' not in str(v))
    _extracted_seq = sum(1 for v in extracted_values if v is not None and ' ' in str(v))
    _filtered_count = len(filtered_df)
    _diag_stats = (
        f"[步骤2 诊断] {base_name}: 提取值统计 — None={_extracted_none}, "
        f"单值={_extracted_single}, 序列={_extracted_seq}, "
        f"过滤后保留={_filtered_count}/{len(coords_df)} (要求长度={required_length})"
    )
    print(_diag_stats, flush=True)
    _write_diag_file(output_dir, _diag_stats)

    if len(filtered_df) > 0:
        filtered_df.to_csv(filtered_output, index=False, encoding='utf-8-sig')
    else:
        empty_df = pd.DataFrame(columns=coords_df.columns.tolist())
        empty_df.to_csv(filtered_output, index=False, encoding='utf-8-sig')
        _diag_empty = f"[步骤2 诊断] {base_name}: 过滤后无有效数据，输出空CSV（仅表头）"
        print(_diag_empty, flush=True)
        _write_diag_file(output_dir, _diag_empty)

    extracted_samples = filtered_df['提取值'].dropna().tolist()
    _combined_diag = f"{_diag_coords}; {_diag_stats}"
    del coords_df, filtered_df, parsed_data
    gc.collect()

    return True, extracted_samples, _combined_diag


def calculate_hamming_distance(seq1, seq2):
    """计算两个序列的汉明距离（支持整数编码与连续浮点数值序列）"""
    if len(seq1) != len(seq2):
        return max(len(seq1), len(seq2))
    return sum(a != b for a, b in zip(seq1, seq2))


def find_best_representative(sequences):
    """组内对比：寻找差异性最小的样本作为代表

    兼容两种序列格式：
    - 离散整数编码（默认流程，五级分位数离散化后）
    - 连续数值序列（消融实验1：跳过离散化后为浮点值）
    """
    if len(sequences) == 1:
        return sequences[0]

    try:
        # 优先按整数解析（离散编码），失败则按浮点解析（跳过离散化的连续数值）
        try:
            num_sequences = [[int(x) for x in seq.split()] for seq in sequences]
        except ValueError:
            num_sequences = [[float(x) for x in seq.split()] for seq in sequences]
    except ValueError:
        return sequences[0]

    min_total_dist = float('inf')
    best_idx = 0

    for i in range(len(num_sequences)):
        total_dist = 0
        seq_i = num_sequences[i]
        for j in range(len(num_sequences)):
            if i == j:
                continue
            seq_j = num_sequences[j]
            total_dist += calculate_hamming_distance(seq_i, seq_j)

        if total_dist < min_total_dist:
            min_total_dist = total_dist
            best_idx = i

    return sequences[best_idx]


def _process_samples_worker(args):
    """顶层包装：多进程调用 process_samples_optimized"""
    csv_path, experiment_data_path, output_dir, required_length, target_shape = args
    return process_samples_optimized(csv_path, experiment_data_path, output_dir, required_length, target_shape)
    # 返回 (success_bool, samples_list, diag_string)


# ============================================================
# 诊断辅助（子进程写文件，主进程可读取）
# ============================================================

def _write_diag_file(output_dir: str, message: str):
    """将诊断信息追加写入 step2_diag.log（多进程安全，尽力而为）"""
    try:
        diag_path = os.path.join(output_dir, "step2_diag.log")
        with open(diag_path, "a", encoding="utf-8") as _df:
            _df.write(message + "\n")
    except Exception:
        pass


# ============================================================
# 主函数
# ============================================================

def run_step2(
    task_manager,
    task_id: str,
    shp_input: str,
    tif_folder: str,
    experiment_data_path: str,
    output_dir: str,
) -> dict:
    """
    执行步骤2处理
    """
    def update_progress(progress: float, message: str):
        """更新任务进度 — 同时更新顶层 progress 和 result.step_progress，
        确保前端轮询时步骤进度条能正常推进。"""
        task_manager.update_task(
            task_id,
            progress=progress,
            message=message,
            result={
                "current_step": 2,
                "step_message": message,
                "step_progress": progress,
                "current_round": 0,
            },
        )

    try:
        os.makedirs(output_dir, exist_ok=True)

        # 获取序列长度
        update_progress(5, "统计影像波段数...")
        # ★ 关键修复：以 merged.csv 的实际序列长度为权威来源（而不是文件系统计数）。
        #   哨兵：111文件→127子波段→127 CSV → 长度=127 ✓
        #   Landsat：340文件→340子波段→实际可能只有 335 CSV 成功 → 长度=335 ✓
        #   旧逻辑：count_total_bands 数的是 340，会把 merged.csv 的 335 长 cell 全部
        #   strict-check 掉，导致所有样本返回 None。
        from scripts.process01 import resolve_required_sequence_length
        required_length, length_source = resolve_required_sequence_length(
            experiment_data_path=experiment_data_path,
            tif_folder=tif_folder,
        )
        if required_length == 0:
            return {"success": False, "error": "未找到TIF文件或merged.csv无法推断序列长度"}
        update_progress(10, f"检测到序列长度: {required_length} ← {length_source}")
        print(
            f"[步骤2] ★ 序列长度={required_length} ← {length_source}",
            flush=True,
        )

        # 创建临时目录
        temp_dir = create_temp_dir()

        # 提取样本点
        update_progress(15, "提取样本点像素值...")
        tif_files = [f for f in os.listdir(tif_folder) if f.lower().endswith(('.tif', '.tiff'))]
        if not tif_files:
            return {"success": False, "error": f"影像文件夹 [{tif_folder}] 中未找到TIF文件"}
        raster_path = os.path.join(tif_folder, sorted(tif_files)[0])

        # ── ★ 读取影像实际尺寸（行列数），用于步骤1 CSV 尺寸适配 ──
        img_height = 0
        img_width = 0
        try:
            import rasterio as _rio
            with _rio.open(raster_path) as _src:
                img_data = _src.read(1)
                img_height, img_width = img_data.shape
            print(f"[步骤2] 影像尺寸: {img_height}×{img_width}, 序列长度需求={required_length}", flush=True)
        except Exception as _e:
            print(f"[步骤2] 警告: 无法读取影像尺寸: {_e}, 将跳过CSV尺寸适配", flush=True)

        try:
            extract_points_with_pixel_location_batch(shp_input, raster_path, temp_dir,
                                                     global_diag_dir=output_dir)
        except ValueError as ve:
            return {"success": False, "error": str(ve)}
        except Exception as e:
            return {"success": False, "error": f"提取样本点失败: {type(e).__name__}: {e}"}
        update_progress(40, "样本点提取完成")

        # 处理样本
        step06_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv') and f != 'summary.csv']
        all_class_data = {}

        if not step06_files:
            # 列出临时目录内容以便诊断
            temp_contents = os.listdir(temp_dir) if os.path.isdir(temp_dir) else []
            return {
                "success": False,
                "error": (
                    f"未找到提取的样本CSV文件。临时目录 [{temp_dir}] 内容: {temp_contents or '空'}。"
                    f"请确认 SHP 文件 [{shp_input}] 包含有效的样本数据，且与影像坐标系匹配。"
                )
            }

        update_progress(45, "处理样本并提取序列（多进程）...")

        args_list = [
            (os.path.join(temp_dir, csv_file), experiment_data_path, output_dir, required_length, (img_height, img_width))
            for csv_file in step06_files
        ]

        # 用 Path 提取 class_name，跨平台安全
        total_args = len(args_list)
        sample_diags = []
        with ProcessPoolExecutor(max_workers=min(_MAX_WORKERS, total_args), initializer=_init_pool) as ex:
            futures = {
                ex.submit(_process_samples_worker, args): Path(args[0]).stem
                for args in args_list
            }
            done = 0
            for fut in as_completed(futures):
                class_name = futures[fut]
                try:
                    result = fut.result()
                    if isinstance(result, tuple) and len(result) == 3:
                        success, samples, diag = result
                        sample_diags.append(diag)
                    elif isinstance(result, tuple) and len(result) == 2:
                        success, samples = result
                    else:
                        success = result
                    if success and samples:
                        all_class_data[class_name] = samples
                except Exception as e:
                    sample_diags.append(f"样本处理异常 {class_name}: {type(e).__name__}: {e}")
                done += 1
                update_progress(45 + done / total_args * 20, f"已完成 {done}/{total_args} 个类别")

        # ── 汇总样本处理诊断 ──
        if sample_diags:
            _summary = "\n".join(f"  {msg}" for msg in sample_diags)
            print(f"[步骤2 诊断汇总] 样本处理:\n{_summary}", flush=True)

        update_progress(65, "样本处理完成")

        # 计算各类别的光谱指纹（平均光谱曲线）
        spectral_profiles = {}
        for class_name, sequences in all_class_data.items():
            if not sequences:
                continue
            # 将字符串序列转换为数值矩阵
            numeric_matrix = []
            for seq in sequences:
                try:
                    vals = [float(x) for x in str(seq).split()]
                    if vals:
                        numeric_matrix.append(vals)
                except ValueError:
                    continue
            if numeric_matrix:
                arr = np.array(numeric_matrix)
                mean_profile = arr.mean(axis=0).tolist()
                spectral_profiles[class_name] = mean_profile

        if len(all_class_data) < 2:
            return {
                "success": True,
                "output_files": [os.path.join(output_dir, f) for f in os.listdir(output_dir)],
                "recommendations": [],
                "message": "样本类别不足，无法计算差异性",
            }

        # 组内对比：找代表值
        update_progress(70, "组内差异分析...")
        class_representatives = {}
        for classname, sequences in all_class_data.items():
            if sequences:
                class_representatives[classname] = find_best_representative(sequences)

        # 组间对比：算差异
        update_progress(80, "组间差异分析...")
        class_variance = defaultdict(float)
        class_names = list(class_representatives.keys())

        for i, class_a in enumerate(class_names):
            total_distance = 0
            for j, class_b in enumerate(class_names):
                if i == j:
                    continue
                # 兼容离散整数编码与连续数值序列（消融实验1：跳过离散化）
                try:
                    vec_a = [int(x) for x in class_representatives[class_a].split()]
                    vec_b = [int(x) for x in class_representatives[class_b].split()]
                except ValueError:
                    vec_a = [float(x) for x in class_representatives[class_a].split()]
                    vec_b = [float(x) for x in class_representatives[class_b].split()]
                dist = calculate_hamming_distance(vec_a, vec_b)
                total_distance += dist

            avg_distance = total_distance / (len(class_names) - 1)
            class_variance[class_a] = avg_distance

        # 排序生成推荐
        sorted_classes = sorted(class_variance.items(), key=lambda x: x[1])
        recommendations = [
            {
                "class_name": name,
                "rank": idx + 1,
                "confusion_score": round(score, 2),
                "representative_sequence": class_representatives[name][:50] + "..."
                if len(class_representatives[name]) > 50
                else class_representatives[name],
            }
            for idx, (name, score) in enumerate(sorted_classes)
        ]

        # 保存推荐文件
        rec_path = os.path.join(output_dir, "匹配顺序建议.txt")
        with open(rec_path, 'w', encoding='utf-8') as f:
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"序列长度: {required_length}\n")
            order_str = " -> ".join([r["class_name"] for r in recommendations])
            f.write(f"推荐匹配顺序: {order_str}\n\n")
            f.write("详细评分:\n")
            for name, score in sorted_classes:
                f.write(f"{name}: {score:.2f}\n")

        update_progress(100, "处理完成!")

        total_samples = sum(len(s) for s in all_class_data.values())
        valid_samples = total_samples  # 已经过滤处理，均为有效样本

        return {
            "success": True,
            "output_files": [os.path.join(output_dir, f) for f in os.listdir(output_dir)],
            "recommendations": recommendations,
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "spectral_profiles": spectral_profiles,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    finally:
        if 'temp_dir' in locals():
            cleanup_temp_dir(temp_dir)
