"""
步骤1: 影像填充合并 - 核心处理逻辑

从 01填充合并.py 提取封装
"""

import os
import re
import csv
import json
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


def count_total_bands(input_dir: str) -> int:
    """统计文件夹中所有影像的子波段总数（多波段文件按实际波段数计）。

    背景：process_single_image 会把多波段影像（如哨兵 TCI 真彩色文件，含3个波段）
    拆成多个 CSV（_B1/_B2/_B3），因此 merged.csv 中每个单元格的序列长度等于
    子波段总数，而不是影像文件数。所有步骤的序列长度校验（步骤2截断/过滤、
    步骤3严格相等校验、阈值缩放基准）必须与之一致，否则会出现
    「序列127位 vs 校验111位」的不匹配，导致所有单元格被跳过、精度永远为0。

    返回:
        子波段总数；无影像文件时返回 0；无法读取的文件按 1 个波段计。
    """
    supported_extensions = ['.tif', '.tiff', '.img']
    image_files = []
    for ext in supported_extensions:
        image_files.extend(glob(os.path.join(input_dir, f'*{ext}')))
        image_files.extend(glob(os.path.join(input_dir, f'*{ext.upper()}')))
    image_files = sorted(set(image_files))
    if not image_files:
        return 0

    rasterio = lazy_import_rasterio()
    total = 0
    for fp in image_files:
        n = 1
        if rasterio is not None:
            try:
                with rasterio.open(fp) as src:
                    n = src.count
            except Exception:
                n = 1
        total += max(1, n)
    return total


def detect_sequence_length_from_csv(csv_path: str, max_scan_rows: int = 20) -> int:
    """从 merged.csv 第一个非空格分隔的 cell 推断实际序列长度。

    为什么需要这个？ count_total_bands() 是按文件系统+TIF子波段数算的，
    但 step1 实际生成的 per-band CSV 数量常常少于子波段总数（部分 TIF 读取失败、
    越界、空文件等），导致 merged.csv 中每个 cell 的序列长度 < count_total_bands。
    若 step2/3 仍按 count_total_bands 校验，len(parts) != required 会把
    所有 cell 过滤掉，匹配数为0，精度永远0%。

    解决：以 merged.csv 的实际内容为权威。统计第一行所有含空格 cell 的
    token 数并取「众数」作为序列长度。

    ⚠️ 为什么不能用「第一个 cell」（2026-09-05 事故教训）：
    merged.csv 首行第 0 列（以及最后一列）存在只有 60 个 token 的残缺重复列，
    若取第一个含空格的 cell 会得到 60 而非真实的 460，meta.json 记录错误长度后
    Step2 按 60 截取指纹、Step3 按 460 匹配，长度不一致 → 全部匹配失败 →
    重分类全为 0 → 精度 %。取众数可天然忽略个别残缺列。

    ⚠️ 为什么不能只看「第一行」（2026-09-18 全指数数据教训）：
    全指数数据（460子波段）的 merged.csv 整个首行 576 个格子全部是 60 token
    的残缺列（第 2 行起才正常为 457），若只统计第一行，众数仍是 60，
    会复现完全相同的精度 0% 故障。因此改为扫描前 N 行（默认 20 行），
    对所有含空格 cell 的 token 数取全局众数（并列取较大值），
    用大量正常行淹没残缺行。
    """
    if not csv_path or not os.path.isfile(csv_path):
        return 0
    try:
        from collections import Counter
        cnt = Counter()
        with open(csv_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= max_scan_rows:
                    break
                for cell in row:
                    s = cell.strip()
                    if s and ' ' in s:
                        cnt[len(s.split())] += 1
        if cnt:
            # 全局众数（最常见 token 数）；并列时取较大值，避免被残缺短行拉低
            max_freq = max(cnt.values())
            return max(n for n, c in cnt.items() if c == max_freq)
    except Exception:
        return 0
    return 0


def write_merged_metadata(
    csv_path: str,
    source_image_dir: str = "",
    source_image_count: int = 0,
    extra: dict = None,
) -> bool:
    """在 merged.csv 旁写一个 .meta.json，记录「真实」序列长度等元数据。

    step2/3 会优先从这里读序列长度（而不是文件系统计数），从而：
    - 哨兵数据（111文件→127子波段→127 CSV）一致 ✓
    - Landsat 数据（340文件→340子波段→实际只有 335 CSV 成功）也一致 ✓
    - 用户换了/挪了文件夹：仍以 merged.csv 的实际长度为权威 ✓

    返回 True/False 表示是否成功写入。
    """
    if not csv_path or not os.path.isfile(csv_path):
        return False
    try:
        rows_count = 0
        cols_count = 0
        with open(csv_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                rows_count += 1
                if len(row) > cols_count:
                    cols_count = len(row)
        seq_len = detect_sequence_length_from_csv(csv_path)
        subband_count = 0
        if source_image_dir and os.path.isdir(source_image_dir):
            try:
                subband_count = count_total_bands(source_image_dir)
            except Exception:
                subband_count = 0
        meta = {
            "csv_path": os.path.abspath(csv_path),
            "sequence_length": seq_len,
            "image_height": rows_count,
            "image_width": cols_count,
            "source_image_dir": os.path.abspath(source_image_dir) if source_image_dir else "",
            "source_image_count": int(source_image_count or 0),
            "source_subband_count": int(subband_count or 0),
            "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if extra:
            meta.update(extra)
        meta_path = csv_path + ".meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(
            f"[Step1] ✓ 写入元数据: {meta_path} "
            f"(序列长度={seq_len}, 影像={rows_count}×{cols_count}, "
            f"影像文件={source_image_count}, 子波段={subband_count})",
            flush=True,
        )
        return True
    except Exception as e:
        print(f"[Step1] 警告: 写入元数据失败: {type(e).__name__}: {e}", flush=True)
        return False


def resolve_required_sequence_length(
    experiment_data_path: str = "",
    tif_folder: str = "",
) -> tuple:
    """解析步骤2/3 真正应该使用的「序列长度」。

    优先级（高→低）：
      1) merged.csv.meta.json  — step1 写入的最权威来源
      2) merged.csv 第一个非空 cell — 自动推断实际长度（不依赖 meta 文件）
      3) tif_folder 子波段数 (count_total_bands) — 兜底
      4) tif_folder 文件数 — 最终兜底

    返回 (length, source_description)；
    全部失败时返回 (0, "无来源")。
    """
    # 1) meta JSON
    if experiment_data_path and os.path.isfile(experiment_data_path + ".meta.json"):
        try:
            with open(experiment_data_path + ".meta.json", 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if 'sequence_length' in meta and int(meta['sequence_length'] or 0) > 0:
                return int(meta['sequence_length']), f"元数据({experiment_data_path}.meta.json)"
        except Exception:
            pass
    # 2) auto-detect from merged.csv
    if experiment_data_path and os.path.isfile(experiment_data_path):
        detected = detect_sequence_length_from_csv(experiment_data_path)
        if detected > 0:
            return detected, f"merged.csv 推断({experiment_data_path})"
    # 3) count_total_bands
    if tif_folder and os.path.isdir(tif_folder):
        try:
            n = count_total_bands(tif_folder)
            if n > 0:
                return n, f"子波段计数({tif_folder})"
        except Exception:
            pass
        # 4) tif 文件数
        try:
            n = len([f for f in os.listdir(tif_folder)
                     if f.lower().endswith(('.tif', '.tiff', '.img'))])
            if n > 0:
                return n, f"文件计数({tif_folder})"
        except Exception:
            pass
    return 0, "无来源"


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
    input_path, output_path, num_classes, exclude_zero = args
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
                    # ★ 2026-09-23 改为可配置策略（exclude_zero）：
                    #   False（默认/auto 的默认兜底）→ 0 值【参与】分位数统计：
                    #       背景自然独占等级1，有效地物在 2~5 级上区分。
                    #       124036 全时相实测 OA=89.91%（优于排除 0 的 88.07%）。
                    #   True → 0 值【排除】出分位数统计：
                    #       分位点只反映有效地物，有效地物独占完整 5 级分辨率。
                    #       129034 九指数实测 OA=91.67%（优于包含 0 的 78.70%）。
                    #   两个数据集结论相反，故不能硬编码，由 zero_policy='auto'
                    #   在运行时按 0 值占比与等级占用情况自动择优。
                    if exclude_zero:
                        if (not np.isnan(val)) and val > 0:
                            data_values.append(val)
                    else:
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


def _decide_zero_policy(
    input_dir: str,
    num_classes: int,
    policy: str = "include",
    sample_files: int = 6,
    sample_rows: int = 400,
) -> dict:
    """决定分位数统计是否把 0 值（NoData / 背景）排除在外。

    policy:
        'include' —— 0 参与统计（等级严格等分，背景独占等级1）【默认】
        'exclude' —— 0 排除出统计（分位点只反映有效地物）
        'auto'    —— 自动择优

    默认取值变更（2026-09-30）：
        默认由 'exclude' 改回 'include'，与 92.66% 历史高精度实验口径一致。
        124036 全时相 2 级离散实测对照：
          评价样本 + include  → OA 98.17%（Kappa 0.9721）
          实验样本_7月版 + include → OA 95.71%（Kappa 0.9408）
          实验样本_7月版 + exclude → OA 89.91%（Kappa 0.8497）
        include 口径在两个样本集上均显著优于 exclude，且是历史实验所用口径。

    自动判据（2026-09-23 依据两组实测标定，仅 policy='auto' 时生效）：
        1. 采样统计 0 值占比 p；
        2. 分别按两套口径求分位点，统计「非 0（有效）像元」实际占用了几个等级；
        3. 若包含 0 时非 0 值被挤掉等级（k_incl < k_excl），说明 0 吞掉了至少
           一个等级、有效地物的等级分辨率受损 → 选 exclude；
           否则保持 include（保守默认）。

        实测标定（旧口径参考值，保留供 auto 判据使用）：
          129034 九指数  p≈20.0%  k_incl=4 < k_excl=5 → exclude → OA 91.67%
          124036 全时相  p≈15.7%  k_incl=5 = k_excl=5 → include
    """
    info = {
        "policy": policy,
        "exclude_zero": False,
        "zero_ratio": None,
        "levels_incl": None,
        "levels_excl": None,
        "reason": "",
        "sampled_files": 0,
        "sampled_values": 0,
    }

    if policy == "include":
        info["reason"] = "手动指定：0 值参与分位数统计（默认口径，与 92.66% 历史实验一致）"
        return info
    if policy == "exclude":
        info["exclude_zero"] = True
        info["reason"] = "手动指定：0 值排除出分位数统计"
        return info

    # ---- auto：采样统计 ----
    try:
        csv_files = sorted(
            f for f in os.listdir(input_dir) if f.lower().endswith(".csv")
        )[:sample_files]
    except Exception as e:
        info["reason"] = f"读取输入目录失败，回退为包含 0（{e}）"
        return info

    vals_all = []
    for fn in csv_files:
        try:
            with open(os.path.join(input_dir, fn), "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= sample_rows:
                        break
                    for cell in row:
                        try:
                            v = float(cell)
                            if not np.isnan(v):
                                vals_all.append(v)
                        except (ValueError, TypeError):
                            pass
        except Exception:
            continue

    info["sampled_files"] = len(csv_files)
    info["sampled_values"] = len(vals_all)

    if len(vals_all) < 1000 or num_classes <= 1:
        info["reason"] = "采样量不足或 num_classes<=1，回退为包含 0"
        return info

    arr = np.asarray(vals_all, dtype=np.float64)
    zero_mask = (arr <= 0)
    nonzero = arr[~zero_mask]
    p = float(zero_mask.mean())
    info["zero_ratio"] = round(p, 6)

    if nonzero.size < 1000:
        info["reason"] = "非 0 值过少，回退为包含 0"
        return info

    q = [(i + 1) / num_classes for i in range(num_classes - 1)]
    thr_incl = sorted(np.quantile(arr, q).tolist())
    thr_excl = sorted(np.quantile(nonzero, q).tolist())

    lvl_incl = np.searchsorted(thr_incl, nonzero, side="left") + 1
    lvl_excl = np.searchsorted(thr_excl, nonzero, side="left") + 1
    k_incl = int(np.unique(lvl_incl).size)
    k_excl = int(np.unique(lvl_excl).size)
    info["levels_incl"] = k_incl
    info["levels_excl"] = k_excl

    if k_excl > k_incl:
        info["exclude_zero"] = True
        info["reason"] = (
            f"0 值占比 {p*100:.2f}% ≥ 1/{num_classes}，包含 0 时有效地物只占用 "
            f"{k_incl}/{num_classes} 个等级（损失 {k_excl-k_incl} 级分辨率）"
            f"→ 自动排除 0，让有效地物占用完整 {k_excl}/{num_classes} 级"
        )
    else:
        info["reason"] = (
            f"0 值占比 {p*100:.2f}% < 1/{num_classes}，两种口径下有效地物均占用 "
            f"{k_incl}/{num_classes} 个等级 → 保持包含 0（保守默认）"
        )
    return info


def process_large_csv_files_with_progress(
    input_dir: str,
    output_dir: str,
    num_classes: int = 5,
    progress_callback=None,
    pause_check=None,
    zero_policy: str = "include",
    decision_out: dict = None,
):
    """分位数分类处理（多进程并行）

    zero_policy: 'include'（默认，0 值参与统计）/ 'exclude' / 'auto'
    decision_out: 可选 dict，用于回传自动判定的依据（供写入元数据）

    pause_check: 可选暂停检查回调
    """
    try:
        csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    except Exception as e:
        raise RuntimeError(f"分位数分类: 读取输入目录失败 [{input_dir}]: {e}") from e
    if not csv_files:
        return 0

    decision = _decide_zero_policy(input_dir, num_classes, zero_policy)
    exclude_zero = bool(decision.get("exclude_zero"))
    print(
        f"[Step1-离散化] 0值处理策略={zero_policy} → "
        f"{'排除0值（分位点只反映有效地物）' if exclude_zero else '包含0值（严格等分）'} | "
        f"{decision.get('reason', '')}"
    )
    if decision_out is not None:
        decision_out.update(decision)

    args_list = [
        (os.path.join(input_dir, fn), os.path.join(output_dir, fn), num_classes, exclude_zero)
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

def standardize_band_csv_files(input_dir: str, output_dir: str, progress_callback=None) -> int:
    """
    逐波段（逐文件）z-score 标准化（消融实验1配套）。

    背景：跳过离散化后指纹为连续数值，反射率波段量级（~10^4）与归一化
    指数波段（-1~1）相差 4 个数量级，直接计算距离会被反射率波段完全主导，
    指数波段几乎不贡献。逐波段标准化后各波段权重相当，距离才真正反映
    光谱曲线形状差异，消融结论才严谨。

    注意：每个输入 CSV 文件对应一个子波段（merge 时按像元位置拼接），
    因此"逐文件标准化"等价于"逐波段标准化"。

    返回: 处理的文件数
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    total = len(csv_files)
    done = 0

    for idx, name in enumerate(csv_files):
        src = os.path.join(input_dir, name)
        dst = os.path.join(output_dir, name)
        try:
            with open(src, 'r', encoding='utf-8', newline='') as f:
                rows = list(csv.reader(f))
        except Exception as e:
            print(f"[Step1-标准化] 读取失败 {name}: {e}")
            continue

        # 第一遍：统计该波段的均值与标准差
        vals = []
        for row in rows:
            for cell in row:
                if cell == '':
                    continue
                try:
                    vals.append(float(cell))
                except (ValueError, TypeError):
                    continue
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            mean = float(arr.mean())
            std = float(arr.std())
        else:
            mean, std = 0.0, 0.0
        if std < 1e-12:
            std = 1.0
            print(f"[Step1-标准化] 波段 {name} 标准差为0，跳过缩放（均值={mean:.4g}）")

        # 第二遍：写回标准化结果
        out_rows = []
        for row in rows:
            new_row = []
            for cell in row:
                if cell == '':
                    new_row.append('')
                    continue
                try:
                    new_row.append(f"{(float(cell) - mean) / std:.6f}")
                except (ValueError, TypeError):
                    new_row.append(str(cell))
            out_rows.append(new_row)

        with open(dst, 'w', encoding='utf-8', newline='') as f:
            csv.writer(f).writerows(out_rows)
        done += 1

        if progress_callback:
            try:
                progress_callback((idx + 1) / max(1, total))
            except Exception:
                pass

    print(f"[Step1-标准化] 完成: {done}/{total} 个波段文件（逐波段 z-score）")
    return done


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
    normalize_continuous: bool = False,
    zero_policy: str = "include",
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
        zero_policy: 分位数统计中 0 值（NoData/背景）的处理策略 —
            'include'（默认，0 参与统计，等级严格等分）/
            'include'（0 参与统计，等级严格等分）/
            'exclude'（0 排除出统计，有效地物独占完整等级分辨率）
        skip_discretization: 消融实验1 — 跳过五级分位数离散化，
            为 True 时不进行分位数分类，直接合并零值处理后的数值文件
        normalize_continuous: 消融实验1配套 — 连续数值模式下对每个子波段做
            z-score 标准化（消除反射率波段与指数波段的量级差），仅在与
            skip_discretization=True 且使用欧氏距离匹配时有意义

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

    # 分位数离散化的 0 值处理判定结果（写入 meta.json 便于复现追溯）
    _zero_decision = {}

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

            # 连续数值标准化（消融实验1配套，可选）
            if normalize_continuous:
                update_progress(82, "逐波段标准化 (z-score)...")
                temp_csv_dir_03n = os.path.join(temp_dir, "step03_csv_norm")
                try:
                    os.makedirs(temp_csv_dir_03n, exist_ok=True)
                except Exception as e:
                    return {"success": False, "error": f"创建临时目录 step03_csv_norm 失败: {type(e).__name__}: {e}"}

                def norm_progress(p):
                    update_progress(82 + p * 2, f"标准化进度 {int(p * 100)}%")

                try:
                    standardize_band_csv_files(temp_csv_dir_03, temp_csv_dir_03n, norm_progress)
                except Exception as e:
                    return {"success": False, "error": f"逐波段标准化失败: {type(e).__name__}: {e}"}
                merge_source_dir = temp_csv_dir_03n
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
                _zero_decision = {}
                process_large_csv_files_with_progress(
                    temp_csv_dir_03,
                    temp_csv_dir_04,
                    num_classes,
                    class_progress,
                    pause_check=_pause_check,
                    zero_policy=zero_policy,
                    decision_out=_zero_decision,
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

            # ★ 写 merged.csv 旁的元数据 JSON（让 step2/3 知道真实序列长度）
            write_merged_metadata(
                output_file,
                source_image_dir=input_img_dir,
                source_image_count=len(image_files) if image_files else 0,
                extra={"zero_policy": _zero_decision} if _zero_decision else None,
            )

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
