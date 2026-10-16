"""
全流程处理 - API 路由实现

功能：将步骤1-6串联为一条完整的处理管道
数据流：
  Step1 output_file(CSV) → Step2 experiment_data_path
  Step2 output_dir/samples → Step3 samples_folder, recommendations → match_order

  迭代循环（每轮）：
    Step3(round_N) matched CSV → Step4 input_csv
    Step4(round_N) *.dat       → Step5 dat_path
    Step5(round_N) *_reclass.dat + unclassified_ratio → 判断
    Step6(round_N) metrics.kappa → 判断

  终止条件（级联判断，有先后顺序）：
    1. Kappa >= 阈值（默认0.8）→ 直接终止
    2. Kappa未达标 但 未分类像素比例 < 阈值（默认10%）→ 终止
    3. 两者均不满足 → 继续迭代
    4. 达到最大轮次 → 强制终止
    5. 所有类别均已冻结 → 提前终止
"""

import os
import sys
import math
import time
import asyncio
import threading
import traceback
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, model_validator

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.core.task_manager import task_manager, TaskStatus
from backend.core.path_utils import normalize_request_paths
from scripts.process01 import run_step1
from scripts.process02 import run_step2
from scripts.process03 import run_step3
from scripts.process04 import run_step4
from scripts.process05 import run_step5
from scripts.process06 import run_step6

router = APIRouter()


# ============================================================
# 地物类别检测（严格基于实际 .shp 文件）
# ============================================================

def detect_classes_from_shp_folder(shp_path: str) -> list[str]:
    """从SHP路径中严格检测地物类别，仅基于实际存在的 .shp 文件。

    这是整个系统中地物类别的「唯一真实来源」(Single Source of Truth)。
    所有其他来源（前端传入、旧缓存CSV、step2推荐）都必须以此为准进行过滤。
    """
    if not os.path.exists(shp_path):
        return []
    if os.path.isfile(shp_path) and shp_path.lower().endswith('.shp'):
        return [os.path.splitext(os.path.basename(shp_path))[0]]
    elif os.path.isdir(shp_path):
        return sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(shp_path)
            if f.lower().endswith('.shp')
        ])
    return []


def validate_classes_against_shp(classes_data: list, shp_input: str, shp_folder: str = None) -> tuple[list, list[str]]:
    """校验并过滤 classes_data，只保留在 SHP 样本文件夹中实际存在 .shp 的类别。

    ★ 注意：只使用 shp_input（样本SHP路径）做校验。
      shp_folder（验证SHP文件夹）是步骤6精度验证的独立数据，
      不应影响步骤1-5的类别有效性判断。

    Returns:
        (filtered_classes, removed_names): 过滤后的列表 + 被移除的类别名
    """
    # 只使用 shp_input（SHP样本路径），忽略 shp_folder（验证SHP）
    ground_truth = set(detect_classes_from_shp_folder(shp_input))

    filtered = []
    removed = []
    for c in classes_data:
        name = c.get("name", "") if isinstance(c, dict) else c
        if name in ground_truth:
            filtered.append(c)
        else:
            removed.append(name)

    return filtered, removed


# ============================================================
# 暂停/恢复/重开 — asyncio.Event 机制
# ============================================================

class PauseAction(str, Enum):
    """暂停后的用户操作"""
    RESUME = "resume"    # 继续执行
    RESTART = "restart"  # 重新开始

# {task_id: asyncio.Event}
_pause_events: dict[str, asyncio.Event] = {}
# {task_id: PauseAction}  — 用户选择的动作
_pause_actions: dict[str, PauseAction] = {}
_lock = threading.Lock()


def _create_pause_event(task_id: str) -> asyncio.Event:
    """为任务创建暂停事件（在后台任务中调用）"""
    event = asyncio.Event()
    with _lock:
        _pause_events[task_id] = event
        _pause_actions.pop(task_id, None)  # 清除旧动作
    return event


def signal_resume(task_id: str, action: PauseAction = PauseAction.RESUME) -> bool:
    """从 API 端点调用：唤醒暂停的任务。

    如果 pipeline 已创建 _pause_events，直接 event.set() 唤醒。
    如果 pipeline 尚未到达暂停检查点（event 未创建），则直接将状态
    改回 RUNNING——pipeline 后续检测到 RUNNING 会自然继续执行。
    """
    import time as _time
    # 先记录动作，然后尝试查找 event
    with _lock:
        _pause_actions[task_id] = action
    _deadline = _time.time() + 5.0
    while _time.time() < _deadline:
        with _lock:
            event = _pause_events.get(task_id)
        if event:
            event.set()
            return True
        _time.sleep(0.5)
    # 5 秒后 event 仍未创建 → pipeline 可能处于长步骤中，直接改状态
    task = task_manager.get_task(task_id)
    if task and task.status == TaskStatus.PAUSED:
        task_manager.resume_task(task_id)
        print(f"[signal_resume] 任务 {task_id} event 未就绪，直接恢复状态为 RUNNING", flush=True)
        return True
    return False


def _cleanup_pause_event(task_id: str):
    """清理暂停事件"""
    with _lock:
        _pause_events.pop(task_id, None)
        _pause_actions.pop(task_id, None)


class RestartPipelineSignal(Exception):
    """内部信号：用户选择重新开始，跳出迭代循环"""
    pass


# ============================================================
# 暂停检查辅助函数
# ============================================================

def _check_paused(task_id: str) -> bool:
    """检查任务是否已被暂停。返回 True 表示已暂停，调用方应退出执行。"""
    task = task_manager.get_task(task_id)
    if task and task.status == TaskStatus.PAUSED:
        return True
    return False


# ============================================================
# Request Models
# ============================================================

class BandRange(BaseModel):
    start: int = Field(ge=1)
    end: int = Field(ge=1)


class RoundConfig(BaseModel):
    threshold: int = Field(ge=0)
    replacements: dict[str, str]


class LandCoverClass(BaseModel):
    name: str
    values: list[int]
    # 冻结策略全局固定（连续2轮增幅<0.5%），以下字段保留兼容但后端不使用
    stop_pa: Optional[float] = Field(default=None, description="已废弃，冻结由全局固定策略控制")
    stop_improvement_gap: Optional[float] = Field(default=None, description="已废弃，固定0.5%")
    consecutive_rounds: Optional[int] = Field(default=None, description="已废弃，固定2轮")


class PipelineRequest(BaseModel):
    """全流程请求参数（所有步骤所需的初始输入）"""

    # ★ 路径规范化：把 "E://x//y"、"E:\\x\\y" 等写法统一收敛，
    #   防止 pandas 把 "E://" 误判为 fsspec 协议 URL（桌面版打包环境实测踩坑）。
    @model_validator(mode="before")
    @classmethod
    def _normalize_path_slashes(cls, data):
        return normalize_request_paths(data)

    # ── 步骤1 参数 ────────────────────────────────────────────
    input_img_dir: str = Field(description="影像文件夹路径（.tif/.tiff/.img）")
    output_dir: str = Field(description="所有步骤的公共输出根目录")
    existing_step1_csv: Optional[str] = Field(default=None, description="已有的步骤1输出CSV文件绝对路径，填写后跳过步骤1直接使用该文件")
    filename: str = Field(default="merged", description="步骤1输出文件名（不含扩展名）")
    num_classes: int = Field(default=5, ge=1, le=100, description="分位数分类数量（1-N可调）")
    skip_discretization: bool = Field(default=False, description="消融实验1：跳过五级分位数离散化，直接合并零值处理后的数值文件（后续步骤以连续数值进行距离计算）")
    normalize_continuous: bool = Field(default=False, description="消融实验1配套：跳过离散化时对连续数值做逐波段 z-score 标准化（消除反射率与指数波段的量级差），欧氏距离才有统一量纲")
    zero_policy: str = Field(default='include', description="分位数统计中 0 值（NoData/背景）的处理策略：'include'=0 参与统计（默认，等级严格等分，与 92.66% 历史实验口径一致）/ 'exclude'=排除（只取 >0 的值参与分位数统计，有效地物独占完整等级分辨率）/ 'auto'=按 0 值占比自动择优")
    delete_ranges: Optional[list[BandRange]] = Field(default=None, description="要删除的波段范围")
    adjust_sizes: bool = Field(default=True, description="是否自动调整不同尺寸的CSV")

    # ── 步骤2 参数 ────────────────────────────────────────────
    shp_input: str = Field(description="SHP文件或文件夹路径")
    # tif_folder 已合并到 input_img_dir，前端不再单独输入
    tif_folder: Optional[str] = Field(default=None, description="TIF影像文件夹路径（已废弃，自动复用input_img_dir）")

    # ── 步骤3 参数 ────────────────────────────────────────────
    target_file: Optional[str] = Field(default=None, description="待匹配的目标CSV文件路径（步骤1输出，可选，默认自动推导为 output_dir/step1/filename.csv）")
    threshold_mode: str = Field(default='manual', description="阈值模式：'manual'=初始阈值自动按影像文件数5%计算，'auto'=系统采样分析计算最优初始阈值（步长两种模式均按影像文件数2.5%自动计算）")
    initial_threshold: float = Field(default=15, ge=0, description="汉明距离初始阈值。【已废弃】2026-09-09 起改为自动计算：初始阈值=影像文件数(时相文件数)×5% 并向上取整为整数（例：50个文件→2.5→3）。此字段值不再生效，仅保留兼容旧前端。")
    threshold_step: float = Field(default=5, ge=0, le=100, description="每轮阈值递增步长。【已废弃】2026-09-09 起改为自动计算：步长=影像文件数(时相文件数)×2.5% 并向上取整为整数（例：50个文件→1.25→步长2）。此字段值不再生效，仅保留兼容旧前端。")
    auto_threshold_samples: int = Field(default=2000, ge=100, le=10000, description="自动阈值模式下，用于分析的像素采样数量")
    distance_metric: str = Field(default='hamming', description="消融实验2：距离度量，'hamming'=汉明距离（默认，位级匹配），'euclidean'=欧氏距离匹配")
    hamming_tolerance: float = Field(default=0.0, ge=0.0, le=1000.0, description="消融实验1配套：汉明距离容差 ε。0=严格逐位相等判定（经典汉明）；>0 时 |ref-current|<=ε 视为该波段相等。跳过离散化后序列为连续浮点值，必须设 ε>0 否则距离恒=序列长度、必然零匹配。输入做了逐波段标准化时，ε 的单位为标准差")
    execution_mode: str = Field(default='multiprocess', description="匹配执行模式：'gpu'=GPU加速, 'multiprocess'=多进程并行（默认），'serial'=主循环串行")
    num_processes: int = Field(default=0, ge=0, le=64, description="多进程模式的进程数，0=自动（max(8,CPU核心数)），建议范围4-32")
    # 兼容旧字段（已废弃，保留防止旧前端报错）
    rounds: list[RoundConfig] = Field(default=[], description="【已废弃】旧版手动轮次配置，由 initial_threshold 替代")

    # ── 步骤4 参数 ────────────────────────────────────────────
    add_geo: bool = Field(default=True, description="是否添加地理坐标")
    max_sequence_length: Optional[int] = Field(default=None, description="最大序列长度")

    # ── 步骤5 参数 ────────────────────────────────────────────
    classes: list[LandCoverClass] = Field(description="地物类别定义列表")

    # ── 步骤6 参数 ────────────────────────────────────────────
    shp_folder: str = Field(description="验证SHP文件夹路径")

    # ── 迭代控制参数 ──────────────────────────────────────────
    kappa_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Kappa系数终止阈值（≥此值终止）")
    unclassified_threshold: float = Field(default=10.0, ge=0.0, le=100.0, description="未分类像素比例终止阈值（<此值时达标），单位%")
    max_iterations: int = Field(default=25, ge=1, le=50, description="最大迭代轮次")
    fixed_rounds: bool = Field(default=True, description="固定轮次模式：忽略Kappa/未分类/类别冻结等提前终止条件，始终跑满max_iterations轮")
    consecutive_rounds: int = Field(default=3, ge=1, le=10, description="类别冻结连续轮次：连续N轮PA增幅<0.5%则冻结该类别")


# ============================================================
# Pipeline Steps Info
# ============================================================

PIPELINE_STEPS = [
    {"id": 1, "name": "影像填充合并"},
    {"id": 2, "name": "样本进阶分析"},
    {"id": 3, "name": "汉明距离匹配"},
    {"id": 4, "name": "ENVI格式转换"},
    {"id": 5, "name": "重分类优化"},
    {"id": 6, "name": "精度评价"},
]


# ============================================================
# Helper Functions
# ============================================================

def generate_auto_replacements(match_order: list[str], round_num: int, total_classes: int = 0) -> dict[str, str]:
    """
    根据匹配顺序和轮次自动生成替换值

    例如：match_order = ['水体.csv', '耕地.csv', '林地.csv', '建筑.csv'] (4个类别)
    - 第1轮: 水体=1, 耕地=2, 林地=3, 建筑=4
    - 第2轮: 水体=5, 耕地=6, 林地=7, 建筑=8
    - 第3轮: 水体=9, 耕地=10, 林地=11, 建筑=12
    ...

    参数:
        match_order: 匹配顺序列表（样本文件名），可以是全部或活跃类别子集
        round_num: 当前轮次（从1开始）
        total_classes: 总类别数（用于计算替换值偏移，避免冻结后值冲突）
                       若为0则使用 match_order 的长度（兼容旧行为）

    返回:
        {filename: replacement_value} 字典
    """
    effective_total = total_classes if total_classes > 0 else len(match_order)
    if effective_total == 0:
        return {}

    start_value = (round_num - 1) * effective_total + 1
    replacements = {}
    for idx, filename in enumerate(match_order):
        value = start_value + idx
        replacements[filename] = str(value)

    return replacements


def _format_time(seconds: float) -> str:
    """将秒数格式化为人类可读字符串"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h {m}m {s}s"


# ============================================================
# 自动阈值计算（基于数据特征分析）
# ============================================================

def _parse_cell_sequence(cell: str, target_length: int, as_float: bool = False):
    """解析单元格中的序列值，返回数值列表或 None

    参数:
        as_float: True 时保留原始浮点值（跳过离散化后的连续数值指纹必须如此，
            否则 int() 截断会破坏数值、容差汉明/欧氏距离失真）
    """
    if not cell or not cell.strip():
        return None
    try:
        vals = cell.strip().split()
        if len(vals) != target_length:
            return None
        if as_float:
            return [float(v) for v in vals]
        return [int(float(v)) for v in vals]
    except (ValueError, AttributeError):
        return None


def _calculate_optimal_threshold(
    target_file: str,
    samples_folder: str,
    match_order: list[str],
    target_sequence_length: int,
    sample_count: int = 2000,
    distance_metric: str = 'hamming',
    hamming_tolerance: float = 0.0,
) -> int:
    """
    自动计算最优初始距离阈值（支持经典汉明、ε-容差汉明与欧氏距离）。

    策略（2026-09-19 修订）：
    1. 从目标CSV中水库采样像素序列（只解析采中的格子，秒级完成）
    2. 计算每个采样像素到所有参考序列的「最近」距离与并列数
       距离度量：
         - euclidean  → sqrt(Σ(ref-current)^2)（连续值）
         - ε=0 汉明   → 严格逐位不等计数
         - ε>0 汉明   → |ref-current|>ε 的波段计数
    3. 在最近距离分布的分位数候选中，选择能平衡"分类覆盖率"和"匹配歧义率"的阈值

    返回推荐的初始阈值（汉明: 整数，范围 [2, max(30, seq_len*0.6)]）
    """
    import random
    import csv

    _is_euclid = (distance_metric == 'euclidean')
    _tol = float(hamming_tolerance) if hamming_tolerance else 0.0
    # 欧氏 / ε-容差模式下必须保留原始浮点值用 float32 参与减法比较
    _need_float = _is_euclid or _tol > 0

    # ── 1. 读取目标CSV并水库采样 ──
    # ★ 2026-09-19 性能修复：旧版把全部格子（可达42万个）解析成整数列表，
    #   约需2亿次 int 转换、耗时数分钟，前端看起来像卡死。
    #   改为水库采样：流式扫描原始字符串，仅对最终采中的 sample_count 个做解析。
    sampled_cells = []   # 水库：保存原始字符串
    seen = 0             # 已扫到的有效格子数
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1']:
        try:
            with open(target_file, 'r', newline='', encoding=encoding) as f:
                for row in csv.reader(f):
                    for cell in row:
                        if not cell or not cell.strip():
                            continue
                        # 廉价长度预检：token 数不符的直接跳过（与 _parse_cell_sequence 语义一致）
                        if len(cell.split()) != target_sequence_length:
                            continue
                        seen += 1
                        if len(sampled_cells) < sample_count:
                            sampled_cells.append(cell)
                        else:
                            j = random.randint(0, seen - 1)
                            if j < sample_count:
                                sampled_cells[j] = cell
            break
        except Exception:
            continue

    if not sampled_cells:
        return 10  # 默认回退

    sampled_sequences = []
    for cell in sampled_cells:
        try:
            if _need_float:
                sampled_sequences.append([float(v) for v in cell.split()])
            else:
                sampled_sequences.append([int(float(v)) for v in cell.split()])
        except (ValueError, AttributeError):
            continue

    if not sampled_sequences:
        return 10

    # ── 2. 读取参考序列 ──
    ref_sequences = []
    for sample_file in match_order:
        sample_path = os.path.join(samples_folder, sample_file)
        if not os.path.exists(sample_path):
            continue
        sample_rows = None
        for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                with open(sample_path, 'r', newline='', encoding=encoding) as f:
                    sample_rows = list(csv.reader(f))
                break
            except Exception:
                continue
        if not sample_rows or len(sample_rows) < 2:
            continue

        # 找到"提取值"列
        header = sample_rows[0]
        seq_col = -1
        for i, col_name in enumerate(header):
            if col_name.strip() == '提取值':
                seq_col = i
                break
        if seq_col == -1:
            continue

        for i in range(1, len(sample_rows)):
            seq = _parse_cell_sequence(sample_rows[i][seq_col] if seq_col < len(sample_rows[i]) else '', target_sequence_length, as_float=_need_float)
            if seq is not None:
                ref_sequences.append(seq)

    if not ref_sequences:
        return 10

    # ── 3. 计算每个采样像素的最近参考距离与并列数 ──
    # ★ 2026-09-19 修复：旧版统计的是「采样像素×全部参考」的 pairwise 距离分布，
    #   中位数/分位数被大量无关参考拉得极高，推荐阈值动辄数百，第1轮就把几乎
    #   全部像素匹配上、精度崩坏。匹配时真正决定成败的是每个像素的「最近距离」。
    _ref_dtype = np.float32 if _need_float else np.int16
    ref_matrix = np.array(ref_sequences, dtype=_ref_dtype)

    min_dists = []    # 每个采样像素到最近参考序列的距离（不一致波段数）
    tie_counts = []   # 最近距离并列的参考序列数（>1 即存在歧义）
    gpu_used = False
    try:
        import scripts.gpu_optimize as _gpu_mod
        _gpu_mod.init_gpu()
        if _gpu_mod.USE_GPU:
            import cupy as cp
            print(f"[自动阈值] 使用GPU加速计算 ({len(sampled_sequences)}像素 x {len(ref_sequences)}参考)", flush=True)
            ref_gpu = cp.asarray(ref_matrix)
            batch_size = 2000
            for i in range(0, len(sampled_sequences), batch_size):
                batch = cp.asarray(np.array(sampled_sequences[i:i + batch_size], dtype=_ref_dtype))
                if _is_euclid:
                    diff = batch[:, None, :] - ref_gpu[None, :, :]
                    dists = cp.sqrt((diff * diff).sum(axis=2, dtype=cp.float32))
                    del diff
                else:
                    diff = cp.not_equal(batch[:, None, :], ref_gpu[None, :, :])
                    dists = diff.sum(axis=2, dtype=cp.int32)
                min_d = cp.min(dists, axis=1)
                tie = (dists == min_d[:, None]).sum(axis=1)
                min_dists.extend(cp.asnumpy(min_d).tolist())
                tie_counts.extend(cp.asnumpy(tie).tolist())
                del batch, diff, dists, min_d, tie
            cp.cuda.Stream.null.synchronize()
            del ref_gpu
            gpu_used = True
    except Exception as e:
        print(f"[自动阈值] GPU不可用，使用CPU: {e}", flush=True)
        gpu_used = False

    if not gpu_used:
        # CPU计算（严格逐位不等计数）
        for seq in sampled_sequences:
            current_arr = np.array(seq, dtype=_ref_dtype)
            if _is_euclid:
                diff = ref_matrix - current_arr
                dists = np.sqrt((diff * diff).sum(axis=1, dtype=np.float32))
            else:
                dists = np.not_equal(ref_matrix, current_arr).sum(axis=1)
            if _is_euclid:
                _d = float(dists.min())
                min_dists.append(_d)
                tie_counts.append(int((dists <= _d + 1e-6).sum()))
            else:
                d_min_i = int(dists.min())
                min_dists.append(d_min_i)
                tie_counts.append(int((dists == d_min_i).sum()))

    _dist_dtype = np.float64 if _is_euclid else np.int64
    min_dists = np.array(min_dists, dtype=_dist_dtype)
    tie_counts = np.array(tie_counts, dtype=np.int64)

    # ── 4. 基于最近距离分布选择最优阈值 ──
    # 欧氏距离为连续值，保留小数（阈值语义=连续距离）；汉明为整数（不一致波段数）
    if _is_euclid:
        _r = lambda v: round(float(v), 3)
        d_min, d_max, d_median = _r(min_dists.min()), _r(min_dists.max()), _r(np.median(min_dists))
        d_q25, d_q75, d_q90 = _r(np.percentile(min_dists, 25)), _r(np.percentile(min_dists, 75)), _r(np.percentile(min_dists, 90))
    else:
        d_min = int(min_dists.min())
        d_max = int(min_dists.max())
        d_median = int(np.median(min_dists))
        d_q25 = int(np.percentile(min_dists, 25))
        d_q75 = int(np.percentile(min_dists, 75))
        d_q90 = int(np.percentile(min_dists, 90))
    d_mean = float(min_dists.mean())

    # 候选阈值取最近距离分布的分位数（覆盖保守到激进的策略）
    candidates = sorted(set([
        d_min,
        d_q25,
        d_median,
        d_q75,
        d_q90,
        max(2, target_sequence_length // 3),
    ]))

    best_threshold = d_median
    best_score = -1.0

    for t in candidates:
        # 覆盖率：多少采样像素的最近距离 ≤ t（第1轮即可被匹配分类）
        covered = min_dists <= t
        coverage = covered.mean()
        if coverage <= 0:
            score = 0.0
        else:
            # 歧义率：被覆盖像素中最近距离存在并列（无法唯一确定类别）的比例。
            # ★ 旧版 bug：ambiguous 不随 t 变化（常数惩罚），等价于直接选覆盖率
            #   最大的候选 → 总是推荐最大阈值。现在歧义率随 t 增大而上升，
            #   才能真正平衡「覆盖率」与「匹配精度」。
            ambiguous = (tie_counts[covered] > 1).mean()
            score = coverage - ambiguous * 2.0
        if score > best_score:
            best_score = score
            best_threshold = t

    # 限制在合理范围 — 上限随序列长度自适应
    # 旧代码硬编码 min(30, ...) 导致111波段的哨兵数据阈值被压到30，
    # 远小于实际需要的值（111波段需要阈值40~60才能有效匹配）
    # 欧氏距离为连续值：不做整数截断，只保证正值（量纲与汉明不同，不能套用同一上限）
    if _is_euclid:
        best_threshold = max(float(best_threshold), 1e-6)
    else:
        max_cap = max(30, int(target_sequence_length * 0.6))
        best_threshold = max(2, min(int(best_threshold), max_cap))

    print(
        f"[自动阈值] 分析完成: 采样{len(sampled_sequences)}个像素, {len(ref_sequences)}条参考序列, "
        f"最近距离范围[{d_min}-{d_max}], P25={d_q25}, 中位数={d_median}, 均值={d_mean:.1f}, "
        f"P75={d_q75}, P90={d_q90}. 推荐初始阈值={best_threshold} (得分={best_score:.3f})",
        flush=True,
    )

    return best_threshold


def _save_timing_report(output_dir: str, timing: dict, stop_reason: str, final_round: int,
                        distance_metric: str = 'hamming', hamming_tolerance: float = 0.0) -> str:
    """生成并保存计时报告TXT文件，返回文件路径"""
    report_path = os.path.join(output_dir, "timing_report.txt")
    lines = []
    lines.append("=" * 60)
    lines.append("  全流程处理计时报告")
    lines.append("=" * 60)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"终止原因: {stop_reason}")
    lines.append(f"总迭代轮次: {final_round}")
    # 记录本轮实际使用的匹配配置，避免报告口径与实验设置不符（消融实验留证）
    if distance_metric == 'euclidean':
        _metric_label = "欧氏距离"
    elif hamming_tolerance and hamming_tolerance > 0:
        _metric_label = f"ε-容差汉明 (ε={hamming_tolerance})"
    else:
        _metric_label = "汉明距离（严格逐位相等）"
    lines.append(f"距离度量: {_metric_label}")
    lines.append("")

    # 整体时间
    overall = timing.get("overall_seconds")
    if overall:
        lines.append(f"【整体运行时间】")
        lines.append(f"  总计: {_format_time(overall)}")
        lines.append("")

    # 步骤1-2时间
    st = timing.get("stage_times", {})
    if st:
        lines.append(f"【各阶段运行时间】")
        step_names = {"step1": "步骤1 影像填充合并", "step2": "步骤2 样本进阶分析",
                      "step3": f"步骤3 {_metric_label}匹配", "step4": "步骤4 CSV→ENVI转换",
                      "step5": "步骤5 ENVI重分类", "step6": "步骤6 精度评价"}
        for key in ["step1", "step2", "step3", "step4", "step5", "step6"]:
            if key in st:
                lines.append(f"  {step_names.get(key, key)}: {_format_time(st[key])}")
        lines.append("")

    # 每轮时间
    round_times = timing.get("round_times", [])
    round_stage_times = timing.get("round_stage_times", [])
    if round_times:
        lines.append(f"【每轮迭代运行时间】")
        for i, rt in enumerate(round_times, 1):
            lines.append(f"  第{i}轮: {_format_time(rt)}")
            if i <= len(round_stage_times):
                rst = round_stage_times[i - 1]
                for sk in ["step3", "step4", "step5", "step6"]:
                    if sk in rst:
                        lines.append(f"    - {step_names.get(sk, sk)}: {_format_time(rst[sk])}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("  报告结束")
    lines.append("=" * 60)

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        print(f"[TimingReport] 保存失败: {e}")
        return ""

    return report_path


# ============================================================
# API Endpoints
# ============================================================

class PreviewReplacementsRequest(BaseModel):
    """预览替换值请求"""
    input_img_dir: str = Field(description="影像文件夹路径（复用基础路径配置）")
    shp_input: str = Field(description="SHP文件或文件夹路径")
    num_rounds: int = Field(default=25, ge=1, le=50, description="预览轮次数")
    experiment_data_path: Optional[str] = Field(default=None, description="步骤1输出的实验数据CSV路径（若提供则按实际匹配顺序生成预览）")


@router.post("/preview-replacements")
async def preview_replacements(request: PreviewReplacementsRequest):
    """
    预览自动生成的替换值（供前端展示）

    若提供 experiment_data_path（步骤1输出），则调用步骤2的真实推荐算法
    获取匹配顺序，确保预览与全流程一致；否则按SHP文件名排序回退。
    """
    input_img_dir = request.input_img_dir
    shp_input = request.shp_input
    num_rounds = request.num_rounds
    experiment_data_path = request.experiment_data_path

    # 检查路径是否存在
    if not os.path.exists(input_img_dir):
        return {"error": f"影像文件夹不存在: {input_img_dir}"}

    if not os.path.exists(shp_input):
        return {"error": f"SHP路径不存在: {shp_input}"}

    # ── ★ 严格检测地物类别（基于实际 .shp 文件，唯一真实来源）──
    ground_truth_classes = detect_classes_from_shp_folder(shp_input)
    if not ground_truth_classes:
        return {
            "error": (
                f"在 [{shp_input}] 中未找到任何 .shp 文件。"
                f"请确认样本文件夹中包含 shapefile 文件（如 建筑用地.shp、林地.shp 等）。"
            ),
            "detected_classes": [],
        }

    # 统计TIF文件数量作为序列长度
    tif_files = sorted([
        f for f in os.listdir(input_img_dir)
        if f.lower().endswith(('.tif', '.tiff'))
    ])
    num_tifs = len(tif_files)

    if num_tifs == 0:
        return {"error": "未找到TIF文件", "detected_classes": ground_truth_classes}

    # ── match_order：按 SHP 文件名排序（唯一真实来源）──
    # 注意：不再在此处运行 step2 推荐算法，因其处理全量 TIF 影像（3241×1201×242 波段）
    # 耗时 30-40 秒，会严重阻塞自动检测 UI。步骤2 的匹配顺序在实际执行全流程时自然得到。
    match_order: list[str] = [f"{name}.csv" for name in ground_truth_classes]
    print(f"[Preview] 检测到 {len(ground_truth_classes)} 个类别: {ground_truth_classes}，序列长度={num_tifs}", flush=True)

    # 最终安全检查：确保 match_order 不超过 ground truth
    match_order_set = set(match_order)
    ground_set = {f"{n}.csv" for n in ground_truth_classes}
    if match_order_set - ground_set:
        print(f"[Preview] 警告：match_order 包含SHP中不存在的类别，已自动过滤", flush=True)
        match_order = [m for m in match_order if m in ground_set]

    # 生成替换值预览
    previews = []
    for round_num in range(1, num_rounds + 1):
        replacements = generate_auto_replacements(match_order, round_num)
        previews.append({
            "round": round_num,
            "replacements": replacements,
        })

    return {
        "previews": previews,
        "num_classes": len(match_order),
        "match_order": match_order,
        "sequence_length": num_tifs,
        "detected_classes": ground_truth_classes,  # ★ 新增：供前端直接显示类别列表
    }


@router.post("/run")
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """启动全流程处理任务（含多轮迭代）"""
    try:
        return await _run_pipeline_impl(request, background_tasks)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        print(f"[Pipeline Run Error] {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"启动全流程失败: {str(e)}")


async def _run_pipeline_impl(request: PipelineRequest, background_tasks: BackgroundTasks):
    """run_pipeline 的实际实现，独立出来以便统一异常处理"""
    # 基础路径校验
    if not os.path.exists(request.input_img_dir):
        raise HTTPException(status_code=400, detail=f"影像文件夹不存在: {request.input_img_dir}")
    if not os.path.exists(request.shp_input):
        raise HTTPException(status_code=400, detail=f"SHP路径不存在: {request.shp_input}")
    if not os.path.exists(request.shp_folder):
        raise HTTPException(status_code=400, detail=f"验证SHP文件夹不存在: {request.shp_folder}")

    # 消融实验参数校验
    if request.distance_metric not in ('hamming', 'euclidean'):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的距离度量: {request.distance_metric}（可选 'hamming' 或 'euclidean'）",
        )
    if request.skip_discretization and request.distance_metric == 'hamming':
        if request.hamming_tolerance > 0:
            # 消融实验1 的正确实现：去掉离散化编码，汉明判定放宽为 ε-容差，
            # 距离仍是「不一致波段数」，与基线（离散化+汉明）同量纲、可比。
            print(
                f"[消融实验1] 跳过离散化 + ε-容差汉明：ε={request.hamming_tolerance}，"
                f"|ref-current|<=ε 视为该波段相等；距离语义与基线一致（不一致波段数）。",
                flush=True,
            )
        else:
            # 消融实验1（跳过离散化）与严格汉明距离搭配时，序列为连续浮点值，
            # 汉明距离按位判断"不相等"，任意两序列距离恒≈序列长度，任何阈值下都不会命中。
            # 该组合用于消融留证（实测 124036：13 轮 OA 全 0.00%、109/109 未分类），
            # 此处只告警、不强制改写，尊重用户的度量选择。
            print(
                "[消融警告] 当前组合 = 跳过离散化 + 严格汉明距离（ε=0）：指纹为连续浮点序列，"
                "按位比较几乎处处不相等，距离恒≈序列长度，任何阈值下都不会命中，"
                "结果必然全部未分类（OA/Kappa=0）。若非刻意留证该组合失效，"
                "请设置 hamming_tolerance>0 或改用 distance_metric='euclidean'。",
                flush=True,
            )
    elif request.hamming_tolerance > 0 and request.distance_metric == 'euclidean':
        print(
            f"[提示] distance_metric='euclidean' 时 hamming_tolerance={request.hamming_tolerance} 不生效。",
            flush=True,
        )

    # 目标CSV文件自动推导（若前端未传则自动构造）
    target_file = request.target_file
    if not target_file:
        target_file = os.path.join(request.output_dir, "step1", f"{request.filename}.csv")
    elif not os.path.exists(target_file):
        # 前端传了但文件还不存在（步骤1尚未执行），这是正常的
        pass

    # 步骤2的影像文件夹自动复用基础影像文件夹
    tif_folder = request.tif_folder if request.tif_folder else request.input_img_dir

    # 创建输出目录（提前创建，确保路径可写）
    try:
        os.makedirs(request.output_dir, exist_ok=True)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"无法创建输出目录 [{request.output_dir}]: {type(e).__name__}: {e}。请确认路径合法且有写入权限。"
        )

    task_id = task_manager.create_task()

    delete_ranges = None
    if request.delete_ranges:
        delete_ranges = [(r.start, r.end) for r in request.delete_ranges]

    classes_data = [
        {
            "name": c.name,
            "values": c.values,
            # 冻结策略：增幅阈值固定0.5%，连续轮次由用户配置
            "stop_improvement_gap": 0.5,
            "consecutive_rounds": request.consecutive_rounds,
        }
        for c in request.classes
    ]

    # ── ★ 严格校验：过滤掉SHP文件夹中不存在的类别 ──
    if classes_data:
        filtered_classes, removed_names = validate_classes_against_shp(
            classes_data, request.shp_input, request.shp_folder
        )
        if removed_names:
            print(
                f"[Pipeline] 类别校验: 过滤了 {len(removed_names)} 个不在SHP文件夹中的类别 → {removed_names}",
                flush=True,
            )
        classes_data = filtered_classes
        if not classes_data:
            _shp_files = detect_classes_from_shp_folder(request.shp_input)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"校验失败：所有地物类别在 SHP样本路径 [{request.shp_input}] 中均无对应的 .shp 文件。"
                    f"检测到的 .shp 文件: {_shp_files if _shp_files else '无'}。"
                    f"请确保「SHP样本路径」中包含 shapefile（如 建筑用地.shp），"
                    f"且「地物类别配置」与 .shp 文件名一一对应。"
                ),
            )

    background_tasks.add_task(
        _run_pipeline_background,
        task_id,
        request.input_img_dir,
        request.output_dir,
        request.filename,
        request.num_classes,
        delete_ranges,
        request.adjust_sizes,
        request.shp_input,
        tif_folder,                  # 自动复用基础影像文件夹
        target_file,
        request.threshold_mode,
        request.initial_threshold,
        request.threshold_step,
        request.auto_threshold_samples,
        request.add_geo,
        request.max_sequence_length,
        classes_data,
        request.shp_folder,
        request.kappa_threshold,
        request.unclassified_threshold,
        request.max_iterations,
        request.consecutive_rounds,
        request.existing_step1_csv or None,  # 已有步骤1结果，有值则跳过步骤1
        request.execution_mode,               # 匹配执行模式：multiprocess/serial
        request.num_processes,                # 多进程模式进程数，0=自动
        request.fixed_rounds,                 # 固定轮次模式：True=跑满max_iterations轮，不提前终止
        request.skip_discretization,          # 消融实验1：跳过离散化，直接合并数值文件
        request.normalize_continuous,         # 消融实验1配套：连续数值逐波段标准化
        request.distance_metric,              # 消融实验2：距离度量 hamming/euclidean
        request.zero_policy,                  # 分位数统计 0 值处理策略 auto/include/exclude
        request.hamming_tolerance,            # 消融实验1配套：汉明距离容差 ε
    )

    return {"task_id": task_id, "message": "全流程任务已创建，正在后台执行"}


@router.get("/info")
async def get_pipeline_info():
    return {
        "name": "全流程处理（多轮迭代）",
        "description": "步骤1-2准备数据，步骤3-6多轮迭代；固定轮次模式（默认开启）下始终跑满最大轮次（默认25轮）",
        "steps": PIPELINE_STEPS,
        "termination": {
            "fixed_rounds": "固定轮次模式（默认开启）：忽略提前终止条件，始终跑满最大迭代轮次（默认25轮）",
            "kappa_threshold": "Kappa系数 ≥ 阈值（默认0.8）【仅非固定轮次模式生效】",
            "unclassified_threshold": "未分类像素比例 < 阈值（默认10%）【仅非固定轮次模式生效】",
            "class_freeze": "类别连续N轮PA增幅<0.5%冻结【仅非固定轮次模式生效】",
            "max_iterations": "达到最大轮次（默认25）时停止",
        },
    }


# ============================================================
# Progress Helpers
# ============================================================

def _progress(task_id: str, step: int, msg: str, pct: float, extra: dict = None):
    """更新任务进度（附带当前步骤信息和迭代轮次信息）"""
    result_data = {
        "current_step": step,
        "step_message": msg,
        "step_progress": pct,
    }
    if extra:
        result_data.update(extra)

    task_manager.update_task(
        task_id,
        status=TaskStatus.RUNNING,
        message=f"[步骤{step}/6] {msg}",
        progress=0.0,           # 进度由调用方单独更新
        result=result_data,
    )


def _update_global_progress(task_id: str, total_pct: float, msg: str):
    """单独更新全局进度百分比"""
    task_manager.update_task(
        task_id,
        progress=round(total_pct, 1),
        message=msg,
    )


# ============================================================
# Background Pipeline Execution
# ============================================================

async def _run_pipeline_background(
    task_id: str,
    input_img_dir: str,
    output_dir: str,
    filename: str,
    num_classes: int,
    delete_ranges,
    adjust_sizes: bool,
    shp_input: str,
    tif_folder: str,
    target_file: str,
    threshold_mode: str,
    initial_threshold: float,
    threshold_step: float,
    auto_threshold_samples: int,
    add_geo: bool,
    max_sequence_length,
    classes_data: list,
    shp_folder: str,
    kappa_threshold: float,
    unclassified_threshold: float,
    max_iterations: int,
    consecutive_rounds: int = 3,
    existing_step1_csv: Optional[str] = None,
    execution_mode: str = 'multiprocess',
    num_processes: int = 0,
    fixed_rounds: bool = True,
    skip_discretization: bool = False,
    normalize_continuous: bool = False,
    distance_metric: str = 'hamming',
    zero_policy: str = 'include',
    hamming_tolerance: float = 0.0,
):
    """全流程后台执行逻辑（多轮迭代版）

    消融实验参数：
        skip_discretization: 消融实验1 — 跳过五级分位数离散化，直接合并数值文件
        normalize_continuous: 消融实验1配套 — 连续数值逐波段 z-score 标准化
        distance_metric:     消融实验2 — 匹配距离度量，'hamming'（默认）或 'euclidean'
        zero_policy:         分位数统计中 0 值处理策略，'include' 为默认（0 参与统计）
        hamming_tolerance:   消融实验1配套 — 汉明距离容差 ε。0=严格逐位相等（经典汉明）；
                             >0 时 |ref-current|<=ε 视为该波段相等，不计入距离。
                             跳过离散化后序列为连续浮点值，必须设 ε>0。
    """

    try:
        # ── 立即标记任务为 RUNNING，防止前端一直显示"等待开始" ──
        task_manager.update_task(
            task_id,
            status=TaskStatus.RUNNING,
            message="全流程启动，正在初始化...",
            progress=0.0,
            result={"current_step": 1, "step_message": "启动中...", "step_progress": 0, "current_round": 0},
        )

        # ── 子目录划分 ──────────────────────────────────────────
        # 先强制创建 output_dir（不管它是否已存在），再 resolve() 规范化
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as _mkdir_err:
            raise RuntimeError(f"无法创建输出根目录 [{output_dir}]: {type(_mkdir_err).__name__}: {_mkdir_err}")
        # resolve() 仅在目录确认存在后调用，避免 Windows 下路径不存在时 resolve 失败
        output_dir = os.path.normpath(str(Path(output_dir).resolve()))
        step1_out = os.path.join(output_dir, "step1")
        step2_out = os.path.join(output_dir, "step2")
        for d in [step1_out, step2_out]:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as _mkdir_err:
                raise RuntimeError(f"无法创建子目录 [{d}]: {type(_mkdir_err).__name__}: {_mkdir_err}")

        # 迭代轮次结果记录
        all_rounds_results = []
        stop_reason = None
        final_result6 = None
        final_round = 0
        final_step5_stats = None

        # 仪表板指标数据（步骤1/2后填充静态信息，每轮步骤6后更新动态信息）
        dashboard_metrics: dict = {}

        # ════════════════════════════════════════════════════
        # 计时统计
        # ════════════════════════════════════════════════════
        timing = {
            "overall_start": time.time(),
            "overall_end": None,
            "overall_seconds": None,
            "stage_times": {},      # { 'step1': sec, 'step2': sec, ... }
            "round_times": [],      # [ sec, sec, ... ]
            "round_stage_times": [],  # [ { 'step3': sec, ... }, ... ]
            "report_file": None,
        }

        # ── 类别冻结追踪 ──────────────────────────────────────────
        # per-class PA history: { class_name: [pa_round1, pa_round2, ...] }
        class_pa_history: dict[str, list[float]] = {c["name"]: [] for c in classes_data}
        # frozen classes set (不再参与后续匹配)
        frozen_classes: set[str] = set()
        # 当前轮次活跃类别 — 在步骤2完成后、进入迭代前赋值
        active_classes: list[str] = []
        # ────────────────────────────────────────────────────
        # 步骤 1：影像填充合并（只执行一次）
        # 若用户已提供现有步骤1输出文件，则跳过步骤1直接使用该文件
        # ────────────────────────────────────────────────────
        if existing_step1_csv and os.path.isfile(existing_step1_csv):
            # ── 跳过步骤1，直接使用已有CSV ──
            step1_csv = existing_step1_csv
            task_manager.update_task(
                task_id,
                status=TaskStatus.RUNNING,
                message=f"[步骤1/6] 已跳过（使用已有文件: {Path(step1_csv).name}）",
                progress=8.0,
                result={"current_step": 1, "step_message": "已跳过（使用已有结果）", "step_progress": 100, "current_round": 0},
            )
            # 提取影像基础信息用于仪表板展示
            try:
                import pandas as pd
                df1 = pd.read_csv(step1_csv, header=None)
                dashboard_metrics["image_size"] = f"{df1.shape[1]}×{df1.shape[0]}"
            except Exception:
                dashboard_metrics["image_size"] = "--"
            dashboard_metrics["bands_count"] = "--"
        elif existing_step1_csv:
            # ── 指定了文件但不存在，给出警告后正常执行步骤1 ──
            task_manager.update_task(
                task_id,
                status=TaskStatus.RUNNING,
                message=f"[步骤1/6] 警告: 指定的步骤1结果文件不存在: {existing_step1_csv}，将重新执行步骤1",
                progress=0.0,
                result={"current_step": 1, "step_message": f"文件不存在，重新执行: {existing_step1_csv}", "step_progress": 0, "current_round": 0},
            )
            # ── 正常执行步骤1 ──
            t1_start = time.time()
            try:
                result1 = await asyncio.to_thread(
                    run_step1,
                    task_manager=task_manager,
                    task_id=task_id,
                    input_img_dir=input_img_dir,
                    output_dir=step1_out,
                    filename=filename,
                    num_classes=num_classes,
                    delete_ranges=delete_ranges,
                    adjust_sizes=adjust_sizes,
                    skip_discretization=skip_discretization,
                    normalize_continuous=normalize_continuous,
                    zero_policy=zero_policy,
                )
            except Exception as e:
                # 写入日志文件以便诊断
                err_log = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pipeline_error.log")
                try:
                    with open(err_log, "a", encoding="utf-8") as lf:
                        lf.write(f"\n{'='*60}\n")
                        lf.write(f"[{datetime.now()}] 步骤1异常 (路径: existing_step1_csv=no)\n")
                        lf.write(f"input_img_dir={input_img_dir}\n")
                        lf.write(f"step1_out={step1_out}\n")
                        traceback.print_exc(file=lf)
                        lf.write(f"\n{'='*60}\n")
                except Exception:
                    pass
                result1 = {"success": False, "error": f"{type(e).__name__}: {e}"}
            timing["stage_times"]["step1"] = round(time.time() - t1_start, 2)
            if not result1.get("success"):
                raise RuntimeError(f"步骤1失败: {result1.get('error', '未知错误')}")

            step1_csv = result1.get("output_file") or os.path.join(step1_out, f"{filename}.csv")

            # 提取影像基础信息用于仪表板展示
            try:
                import pandas as pd
                df1 = pd.read_csv(step1_csv, header=None)
                dashboard_metrics["image_size"] = f"{df1.shape[1]}×{df1.shape[0]}"
            except Exception:
                dashboard_metrics["image_size"] = "--"

            # 从输入影像读取波段数
            try:
                img_path = Path(input_img_dir)
                if img_path.is_file():
                    img_files = [img_path]
                else:
                    img_files = (
                        list(img_path.glob("*.tif"))
                        + list(img_path.glob("*.tiff"))
                        + list(img_path.glob("*.jpg"))
                        + list(img_path.glob("*.png"))
                    )
                if img_files:
                    from scripts.process01 import lazy_import_rasterio
                    rasterio = lazy_import_rasterio()
                    if rasterio:
                        with rasterio.open(str(img_files[0])) as src:
                            dashboard_metrics["bands_count"] = src.count
                    else:
                        from PIL import Image
                        with Image.open(str(img_files[0])) as img:
                            dashboard_metrics["bands_count"] = getattr(img, "n_frames", 1) if hasattr(img, "n_frames") and img.n_frames > 1 else 1
                else:
                    dashboard_metrics["bands_count"] = "--"
            except Exception:
                dashboard_metrics["bands_count"] = "--"

            task_manager.update_task(
                task_id,
                status=TaskStatus.RUNNING,
                message=f"[步骤1/6] 完成（重新生成），输出: {Path(step1_csv).name}",
                progress=8.0,
                result={
                    "current_step": 1,
                    "step_message": "完成",
                    "step_progress": 100,
                    "current_round": 0,
                    "dashboard_metrics": dashboard_metrics,
                },
            )
        else:
            # ── 正常执行步骤1 ──
            task_manager.update_task(
                task_id,
                status=TaskStatus.RUNNING,
                message="[步骤1/6] 开始影像填充合并...",
                progress=0.0,
                result={"current_step": 1, "step_message": "开始...", "step_progress": 0, "current_round": 0},
            )

            t1_start = time.time()
            try:
                result1 = await asyncio.to_thread(
                    run_step1,
                    task_manager=task_manager,
                    task_id=task_id,
                    input_img_dir=input_img_dir,
                    output_dir=step1_out,
                    filename=filename,
                    num_classes=num_classes,
                    delete_ranges=delete_ranges,
                    adjust_sizes=adjust_sizes,
                    skip_discretization=skip_discretization,
                    normalize_continuous=normalize_continuous,
                    zero_policy=zero_policy,
                )
            except Exception as e:
                err_log = os.path.join(output_dir, "pipeline_error.log")
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    with open(err_log, "a", encoding="utf-8") as lf:
                        lf.write(f"\n{'='*60}\n")
                        lf.write(f"[{datetime.now()}] 步骤1异常 (路径: main)\n")
                        lf.write(f"input_img_dir={input_img_dir}\n")
                        lf.write(f"step1_out={step1_out}\n")
                        traceback.print_exc(file=lf)
                        lf.write(f"\n{'='*60}\n")
                except Exception:
                    pass
                result1 = {"success": False, "error": f"{type(e).__name__}: {e}"}
            timing["stage_times"]["step1"] = round(time.time() - t1_start, 2)
            if not result1.get("success"):
                raise RuntimeError(f"步骤1失败: {result1.get('error', '未知错误')}")

            # ★ 检查是否在步骤1执行期间收到 RESTART 请求
            with _lock:
                if _pause_actions.get(task_id) == PauseAction.RESTART:
                    print("[Pipeline] 步骤1完成后检测到 RESTART 请求，终止当前流程", flush=True)
                    raise RestartPipelineSignal()

            step1_csv = result1.get("output_file") or os.path.join(step1_out, f"{filename}.csv")

            # 提取影像基础信息用于仪表板展示
            try:
                import pandas as pd
                df1 = pd.read_csv(step1_csv, header=None)
                dashboard_metrics["image_size"] = f"{df1.shape[1]}×{df1.shape[0]}"
            except Exception:
                dashboard_metrics["image_size"] = "--"

            # 从输入影像读取波段数
            try:
                img_path = Path(input_img_dir)
                if img_path.is_file():
                    img_files = [img_path]
                else:
                    img_files = (
                        list(img_path.glob("*.tif"))
                        + list(img_path.glob("*.tiff"))
                        + list(img_path.glob("*.jpg"))
                        + list(img_path.glob("*.png"))
                    )
                if img_files:
                    from scripts.process01 import lazy_import_rasterio
                    rasterio = lazy_import_rasterio()
                    if rasterio:
                        with rasterio.open(str(img_files[0])) as src:
                            dashboard_metrics["bands_count"] = src.count
                    else:
                        from PIL import Image
                        with Image.open(str(img_files[0])) as img:
                            dashboard_metrics["bands_count"] = getattr(img, "n_frames", 1) if hasattr(img, "n_frames") and img.n_frames > 1 else 1
                else:
                    dashboard_metrics["bands_count"] = "--"
            except Exception:
                dashboard_metrics["bands_count"] = "--"

            task_manager.update_task(
                task_id,
                status=TaskStatus.RUNNING,
                message=f"[步骤1/6] 完成，输出: {Path(step1_csv).name}",
                progress=8.0,
                result={
                    "current_step": 1,
                    "step_message": "完成",
                    "step_progress": 100,
                    "current_round": 0,
                    "dashboard_metrics": dashboard_metrics,
                },
            )

        # ── ★ 关键：将 target_file 同步为步骤1实际输出路径 ──
        # target_file 在 API 入口处根据用户输入自动推导，可能与步骤1实际输出路径不一致
        # （例如：用户提供了 existing_step1_csv 跳过步骤1、路径规范化差异、或步骤1输出了不同文件名）
        # 步骤3需要读取 target_file，必须确保它指向步骤1的真实输出文件
        target_file = step1_csv
        if not os.path.isfile(target_file):
            raise RuntimeError(
                f"步骤1输出文件不存在: {target_file}。"
                f"step1_csv={step1_csv}, 请检查步骤1是否正常执行。"
            )
        print(f"[Pipeline] target_file 已同步为步骤1输出: {target_file}", flush=True)

        # ────────────────────────────────────────────────────
        # 步骤 2：样本进阶分析（只执行一次）
        # ────────────────────────────────────────────────────
        task_manager.update_task(
            task_id,
            status=TaskStatus.RUNNING,
            message="[步骤2/6] 开始样本进阶分析...",
            progress=8.0,
            result={"current_step": 2, "step_message": "开始...", "step_progress": 0, "current_round": 0},
        )

        t2_start = time.time()
        result2 = await asyncio.to_thread(
            run_step2,
            task_manager=task_manager,
            task_id=task_id,
            shp_input=shp_input,
            tif_folder=tif_folder,
            experiment_data_path=step1_csv,
            output_dir=step2_out,
        )
        timing["stage_times"]["step2"] = round(time.time() - t2_start, 2)
        if not result2.get("success"):
            raise RuntimeError(f"步骤2失败: {result2.get('error', '未知错误')}")

        # 提取推荐顺序
        recommendations = result2.get("recommendations", [])
        if recommendations:
            match_order = [f"{r['class_name']}.csv" for r in recommendations]
        else:
            match_order = [f.name for f in Path(step2_out).glob("*.csv")]

        # ── ★ 严格校验：match_order 必须与实际 .shp 文件一致 ──
        _ground_truth = set(detect_classes_from_shp_folder(shp_input)) or set(detect_classes_from_shp_folder(shp_folder))
        _valid_match_order = [m for m in match_order if m.replace('.csv', '') in _ground_truth]
        _removed_by_shp = [m for m in match_order if m.replace('.csv', '') not in _ground_truth]
        if _removed_by_shp:
            print(
                f"[Pipeline Step2] 过滤了 {len(_removed_by_shp)} 个不在SHP文件夹中的类别: {_removed_by_shp}",
                flush=True,
            )
        match_order = _valid_match_order

        # 步骤2输出的类别数量（用于自动替换值生成）
        num_landcover_types = len(match_order)
        if num_landcover_types == 0:
            raise RuntimeError(
                f"步骤2后未发现任何有效类别。SHP路径={shp_input}, "
                f"检测到的.shp文件={list(_ground_truth) if _ground_truth else '无'}"
            )

        # 初始化活跃类别（全部类别均活跃，冻结后逐步缩小）
        active_classes = list(match_order)

        # ── 关键修复：按 match_order 顺序重新排序 classes_data ──
        # 问题：前端传来的 classes_data 顺序可能与 match_order 不一致，
        # 导致步骤5中数值到类别名称的映射错位（如水体被分为人工表面）
        # 解决：强制让 classes_data 的顺序与 match_order 保持一致
        name_to_class = {c["name"]: c for c in classes_data}
        sorted_classes_data = []
        for sample_file in match_order:
            class_name = sample_file.replace('.csv', '')
            if class_name in name_to_class:
                sorted_classes_data.append(name_to_class[class_name])
        # ★ 只保留在 SHP 文件夹中有对应 .shp 的类别，不再追加不存在的类
        classes_data = sorted_classes_data

        # 填充仪表板静态指标
        dashboard_metrics["class_count"] = num_landcover_types
        dashboard_metrics["sample_count"] = result2.get("total_samples", "--")

        task_manager.update_task(
            task_id,
            status=TaskStatus.RUNNING,
            message=f"[步骤2/6] 完成，推荐顺序: {', '.join(match_order[:3])}...",
            progress=15.0,
            result={
                "current_step": 2,
                "step_message": "完成",
                "step_progress": 100,
                "current_round": 0,
                "match_order": match_order,
                "num_landcover_types": num_landcover_types,
                "auto_replacements_preview": {
                    "round_1": generate_auto_replacements(match_order, 1),
                    "round_2": generate_auto_replacements(match_order, 2),
                },
                "dashboard_metrics": dashboard_metrics,
                "spectral_profiles": result2.get("spectral_profiles", {}),
            },
        )

        # ════════════════════════════════════════════════════
        # 多轮迭代循环：步骤3 → 步骤4 → 步骤5 → 步骤6 → 判断
        # ════════════════════════════════════════════════════

        # ── 自动阈值计算（如启用）────────────────────────────
        # 先检测波段数（后续多处使用）
        # ★ 关键修复：必须统计「子波段总数」而非影像文件数——多波段文件（如哨兵
        #   TCI 真彩色，3波段）在步骤1中会被拆成多个CSV（_B1/_B2/_B3），merged.csv
        #   的实际序列长度 = 子波段总数。此前按文件数统计（111 vs 实际127），步骤3的
        #   严格长度校验 len(vals) != target_sequence_length 会跳过所有单元格，
        #   导致任何阈值下匹配数为0、精度永远为0%。
        from scripts.process01 import count_total_bands
        detected_bands = count_total_bands(input_img_dir)
        _file_count = len([f for f in os.listdir(input_img_dir)
                           if f.lower().endswith(('.tif', '.tiff', '.img'))])
        if detected_bands and detected_bands != _file_count:
            print(
                f"[Pipeline] ★ 检测到多波段文件: 影像文件数={_file_count}, "
                f"子波段总数={detected_bands}（多波段文件已按实际波段数计入序列长度）",
                flush=True,
            )
        seq_len = detected_bands if detected_bands > 0 else 7

        if threshold_mode == 'auto':
            try:
                # ★ 2026-09-19 修复：自动阈值分析必须用 merged.csv 的权威序列长度，
                #   不能直接用 count_total_bands——多波段/失败TIF场景下两者可能差几个
                #   token（如 460 vs 457），长度校验 len(vals)!=target_length 会把全部
                #   格子和样本序列拒之门外，分析退化为默认值 10。
                from scripts.process01 import resolve_required_sequence_length
                _auth_len, _auth_src = resolve_required_sequence_length(
                    target_file, tif_folder=input_img_dir)
                if _auth_len and _auth_len > 0:
                    seq_len = _auth_len
                    print(f"[Pipeline] 自动阈值分析使用权威序列长度: {_auth_len} (来源: {_auth_src})", flush=True)
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.RUNNING,
                    message="[步骤3] 自动分析数据特征，计算最优初始阈值...",
                    progress=10.0,
                    result={"current_step": 3, "step_message": "自动阈值分析中...", "step_progress": 0, "current_round": 0},
                )
                auto_threshold = _calculate_optimal_threshold(
                    target_file=target_file,
                    samples_folder=step2_out,
                    match_order=match_order,
                    target_sequence_length=seq_len,
                    sample_count=auto_threshold_samples,
                    distance_metric=distance_metric,
                    hamming_tolerance=hamming_tolerance,
                )
                initial_threshold = auto_threshold
                print(f"[Pipeline] 自动阈值模式: 初始阈值设置为 {initial_threshold}", flush=True)
            except Exception as e:
                print(f"[Pipeline] 自动阈值计算失败，回退到手动默认值15: {e}", flush=True)
                initial_threshold = 15

        # ── ★ 2026-09-09 变更：初始阈值/步长按影像文件数（时相文件数）百分比自动计算 ──
        #   初始阈值 = 影像文件数 × 5%，【2026-09-19 变更】向上取整为整数
        #              例：50个时相文件 → 2.5 → 初始阈值 3
        #   递增步长 = 影像文件数 × 2.5%，【2026-09-19 变更】向上取整为整数
        #              例：50个时相文件 → 1.25 → 步长 2；41个文件 → 1.025 → 步长 2
        #   说明：
        #     1) 手动模式（manual）：初始阈值与步长均按上述规则自动计算，
        #        前端传入的 initial_threshold / threshold_step 不再生效；
        #     2) 自动模式（auto）：初始阈值仍由采样分析（_calculate_optimal_threshold）得出
        #        （该函数本身返回整数），步长同样按 2.5% 规则自动计算；
        #     3) 初始阈值与步长均必须为整数（小数位向上进位，1.1→2、2.5→3）。
        if _file_count > 0:
            if threshold_mode != 'auto':
                initial_threshold = max(1, math.ceil(_file_count * 0.05))
            threshold_step = max(1, math.ceil(_file_count * 0.025))
            print(
                f"[Pipeline] ★ 阈值按影像文件数自动计算: 影像文件数={_file_count}, "
                f"初始阈值={initial_threshold} (5%×{_file_count} 向上取整), "
                f"步长={threshold_step} (2.5%×{_file_count} 向上取整)",
                flush=True,
            )
        else:
            # 防御性兜底：无影像文件时步骤1本会失败，此处仅保证阈值不为 0/负数
            if initial_threshold < 1:
                initial_threshold = 1
            threshold_step = max(1, math.ceil(threshold_step))

        for iteration in range(1, max_iterations + 1):
            round_label = f"第{iteration}轮"
            round_start = time.time()
            round_stage_times = {}

            # 每轮独立输出子目录
            round_dir = os.path.join(output_dir, f"round_{iteration}")
            step3_out = os.path.join(round_dir, "step3")
            step4_out = os.path.join(round_dir, "step4")
            step5_out = os.path.join(round_dir, "step5")
            step6_out = os.path.join(round_dir, "step6")
            for d in [step3_out, step4_out, step5_out, step6_out]:
                os.makedirs(d, exist_ok=True)

            # ── 进度基准：步骤1-2占15%，剩余85%平分给各轮次
            # 简化处理：每轮步骤3-6各占一段，随轮次动态分配
            # 使用固定"当前轮"视角：当前轮的步骤3/4/5/6对应UI显示
            def _iter_progress(step: int, msg: str, pct: float):
                """更新当前迭代的步骤进度（含ETA估算）"""
                # 计算整体进度和ETA
                elapsed = time.time() - timing["overall_start"]
                # 步骤1-2占15%，每轮步骤3-6占剩余85%按轮次平分
                done_rounds = len(all_rounds_results)
                total_estimated = max_iterations
                overall_pct = 15.0 + (done_rounds / total_estimated) * 85.0
                if step > 2 and step <= 6:
                    round_frac = (step - 2) / 4.0
                    overall_pct += (round_frac / total_estimated) * 85.0
                overall_pct = min(99.0, overall_pct)

                eta_sec = None
                if overall_pct > 2 and done_rounds > 0:
                    avg_round_time = sum(timing["round_times"]) / len(timing["round_times"])
                    remaining_rounds = max_iterations - done_rounds
                    eta_sec = avg_round_time * remaining_rounds + elapsed

                task_manager.update_task(
                    task_id,
                    status=TaskStatus.RUNNING,
                    message=f"[{round_label}·步骤{step}/6] {msg}",
                    progress=overall_pct,
                    result={
                        "current_step": step,
                        "step_message": f"{round_label}: {msg}",
                        "step_progress": pct,
                        "current_round": iteration,
                        "total_rounds_so_far": len(all_rounds_results),
                        "dashboard_metrics": dashboard_metrics,
                        "timing": {
                            "elapsed_seconds": round(elapsed, 1),
                            "eta_seconds": round(eta_sec, 1) if eta_sec else None,
                        },
                    },
                )

            # ── 步骤 3：汉明距离匹配 ────────────────────────
            _iter_progress(3, f"开始汉明距离匹配...", 0)

            # 生成替换值：
            #   - 活跃类别：每轮自动生成新替换值（用全部类别数计算偏移，防止冻结后值冲突）
            #   - 冻结类别：保持冻结时的替换值不变（仍需参与匹配，否则像素变"未分类"）
            active_replacements = generate_auto_replacements(active_classes, iteration, total_classes=num_landcover_types)

            # 为冻结类别补充固定的替换值（冻结类必须参与匹配，否则其像素无法被分类）
            frozen_replacements = {}
            for c in classes_data:
                csv_name = f"{c['name']}.csv"
                class_name = c["name"]
                if class_name in frozen_classes and c["values"]:
                    frozen_replacements[csv_name] = str(c["values"][0])

            # 合并：活跃类用新值，冻结类用固定值
            all_replacements = {**frozen_replacements, **active_replacements}

            # 每轮阈值自动递增：第1轮=initial_threshold，第2轮=+threshold_step，第3轮=+2*threshold_step，……
            max_threshold = initial_threshold + (iteration - 1) * threshold_step
            # 设置上限保护：阈值不超过实际序列长度（子波段总数），超过则几乎所有序列都会匹配
            # ★ 复用循环外已计算的 seq_len（子波段总数），不再按文件数重新统计
            # 欧氏距离（消融实验2）为连续距离值，量纲与波段数不同，不做波段数上限截断
            if distance_metric == 'euclidean':
                current_threshold = max_threshold
                # 输出文件名中的阈值格式化（避免浮点文件名过长）
                _thr_str = f"{current_threshold:g}"
            else:
                sequence_length_cap = seq_len if seq_len > 0 else 100
                current_threshold = min(max_threshold, sequence_length_cap)
                _thr_str = f"{current_threshold:g}"
            step3_output_file = os.path.join(step3_out, f"out1_{_thr_str}.csv")
            single_round_config = [{
                "threshold": current_threshold,
                "replacements": all_replacements,  # 包含活跃类别+冻结类别的替换值
            }]

            t3_start = time.time()
            result3 = await asyncio.to_thread(
                run_step3,
                task_manager=task_manager,
                task_id=task_id,
                images_folder=input_img_dir,
                samples_folder=step2_out,
                target_file=target_file,
                output_file=step3_output_file,
                match_order=match_order,  # 全部类别都参与匹配（含冻结类别）
                rounds=single_round_config,
                execution_mode=execution_mode,
                num_processes=num_processes,
                distance_metric=distance_metric,
                hamming_tolerance=hamming_tolerance,
            )
            round_stage_times["step3"] = round(time.time() - t3_start, 2)

            # ── 步骤3在多进程执行中途检测到暂停信号 ──
            if result3.get("paused"):
                stop_reason = f"用户暂停（第{iteration}轮步骤3执行中途）"
                overall_pct = 15.0 + (len(all_rounds_results) / max_iterations) * 85.0
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PAUSED,
                    message=f"[第{iteration}轮] 已暂停（步骤3执行中），可点击恢复继续",
                    progress=min(99.0, overall_pct),
                    result={
                        "current_step": 3,
                        "step_message": "已暂停（步骤3执行中）",
                        "step_progress": 0,
                        "current_round": iteration,
                        "total_rounds_so_far": len(all_rounds_results),
                        "all_rounds": all_rounds_results,
                        "dashboard_metrics": dashboard_metrics,
                        "paused_after_round": iteration - 1,
                        "timing": timing,
                    },
                )
                print(f"[Pipeline] 流程已暂停（第{iteration}轮步骤3中途），等待用户操作...", flush=True)
                pause_event = _create_pause_event(task_id)
                await pause_event.wait()
                with _lock:
                    action = _pause_actions.get(task_id, PauseAction.RESUME)
                _cleanup_pause_event(task_id)
                if action == PauseAction.RESTART:
                    task_manager.update_task(
                        task_id, status=TaskStatus.RUNNING,
                        message="正在重新开始全流程...", progress=0.0,
                        result={"current_step": 1, "step_message": "重新开始中...", "step_progress": 0, "current_round": 0},
                    )
                    print("[Pipeline] 用户选择重新开始，终止当前流程", flush=True)
                    raise RestartPipelineSignal()
                # 恢复执行：重新执行当前轮次（因步骤3未完成）
                task_manager.update_task(
                    task_id, status=TaskStatus.RUNNING,
                    message=f"[第{iteration}轮] 恢复执行，重新运行步骤3...",
                    progress=min(99.0, overall_pct),
                )
                print(f"[Pipeline] 恢复执行，重新执行第{iteration}轮步骤3", flush=True)
                # 重新执行步骤3
                t3_start = time.time()
                result3 = await asyncio.to_thread(
                    run_step3,
                    task_manager=task_manager,
                    task_id=task_id,
                    images_folder=input_img_dir,
                    samples_folder=step2_out,
                    target_file=target_file,
                    output_file=step3_output_file,
                    match_order=match_order,
                    rounds=single_round_config,
                    execution_mode=execution_mode,
                    num_processes=num_processes,
                )
                round_stage_times["step3"] = round(time.time() - t3_start, 2)

            if not result3.get("success"):
                raise RuntimeError(f"{round_label}步骤3失败: {result3.get('error', '未知错误')}")

            step3_csv = result3.get("output_file") or step3_output_file
            _iter_progress(3, f"完成，输出: {Path(step3_csv).name}", 100)

            # 更新 classes_data 的 values：
            #   - 活跃类别：用本轮新替换值
            #   - 冻结类别：all_replacements中是其固定值，更新后仍相同（保持一致性）
            for c in classes_data:
                csv_name = f"{c['name']}.csv"
                if csv_name in all_replacements:
                    c["values"] = [int(all_replacements[csv_name])]

            # ── 步骤 4：CSV → ENVI 转换 ─────────────────────
            _iter_progress(4, "开始ENVI格式转换...", 0)

            t4_start = time.time()
            result4 = await asyncio.to_thread(
                run_step4,
                task_manager=task_manager,
                task_id=task_id,
                input_csv=step3_csv,
                images_folder=input_img_dir,
                output_dir=step4_out,
                add_geo=add_geo,
                max_sequence_length=max_sequence_length,  # 步骤4：序列长度过滤，用户设置的值
            )
            round_stage_times["step4"] = round(time.time() - t4_start, 2)
            if not result4.get("success"):
                raise RuntimeError(f"{round_label}步骤4失败: {result4.get('error', '未知错误')}")

            dat_file = result4.get("output_envi") or result4.get("dat_file") or result4.get("output_file")
            if not dat_file:
                dat_files = list(Path(step4_out).glob("*.dat"))
                if not dat_files:
                    raise RuntimeError(f"{round_label}步骤4未生成.dat文件")
                dat_file = str(dat_files[0])
            _iter_progress(4, f"完成，输出: {Path(dat_file).name}", 100)

            # ── 步骤 5：ENVI 重分类 ─────────────────────────
            _iter_progress(5, "开始重分类优化...", 0)

            # 重分类需要传入所有类别（包括冻结的），否则冻结类别的像素值不会被映射到类名
            # 冻结类别的"values"保持上一轮结果不变，仍需正确映射
            t5_start = time.time()
            result5 = await asyncio.to_thread(
                run_step5,
                task_manager=task_manager,
                task_id=task_id,
                dat_path=dat_file,
                classes=classes_data,
            )
            round_stage_times["step5"] = round(time.time() - t5_start, 2)
            if not result5.get("success"):
                raise RuntimeError(f"{round_label}步骤5失败: {result5.get('error', '未知错误')}")

            # 提取未分类像素比例
            stats5 = result5.get("stats", {})
            total_pixels = stats5.get("total_pixels", 1) or 1
            unclassified_pixels = stats5.get("unclassified_pixels", 0)
            unclassified_ratio = unclassified_pixels / total_pixels * 100  # 百分比

            reclass_dat = result5.get("new_dat_path") or result5.get("reclass_dat") or result5.get("output_file")
            if not reclass_dat:
                stem = Path(dat_file).stem
                reclass_dat = str(Path(dat_file).parent / f"{stem}_reclass.dat")

            _iter_progress(
                5,
                f"完成，未分类像素: {unclassified_ratio:.1f}%（{unclassified_pixels}/{total_pixels}）",
                100,
            )

            # ── 步骤 6：精度评价 ────────────────────────────
            _iter_progress(6, "开始精度评价...", 0)

            t6_start = time.time()
            result6 = await asyncio.to_thread(
                run_step6,
                task_manager=task_manager,
                task_id=task_id,
                reclass_dat=reclass_dat,
                shp_folder=shp_folder,
                output_dir=step6_out,
            )
            round_stage_times["step6"] = round(time.time() - t6_start, 2)
            if not result6.get("success"):
                raise RuntimeError(f"{round_label}步骤6失败: {result6.get('error', '未知错误')}")

            metrics6 = result6.get("metrics", {})
            kappa = metrics6.get("kappa", 0.0)
            oa = metrics6.get("overall_accuracy", 0.0)

            _iter_progress(
                6,
                f"完成，OA={oa:.2f}%，Kappa={kappa:.4f}",
                100,
            )

            # 更新仪表板动态指标
            dashboard_metrics["overall_accuracy"] = round(oa, 2)
            dashboard_metrics["kappa"] = round(kappa, 4)

            # ── 记录本轮耗时 ─────────────────────────────────
            round_elapsed = round(time.time() - round_start, 2)
            timing["round_times"].append(round_elapsed)
            timing["round_stage_times"].append(round_stage_times)

            # ── 记录本轮结果 ─────────────────────────────────
            round_record = {
                "round": iteration,
                "hamming_threshold": current_threshold,   # 本轮实际使用的距离阈值（兼容旧字段名）
                "distance_metric": distance_metric,        # 本轮使用的距离度量（hamming/euclidean，消融实验2）
                "hamming_tolerance": hamming_tolerance,    # 本轮使用的汉明容差 ε（消融实验1配套）
                "kappa": kappa,
                "overall_accuracy": oa,
                "unclassified_ratio": round(unclassified_ratio, 2),
                "unclassified_pixels": int(unclassified_pixels),
                "total_pixels": int(total_pixels),
                "auto_replacements": all_replacements,    # 本轮使用的全部替换值（含冻结类别）
                "outputs": {
                    "step3_csv": step3_csv,
                    "step4_dat": dat_file,
                    "step5_reclass": reclass_dat,
                    "step6_report": step6_out,
                },
                "class_accuracies": result6.get("class_accuracies", []),
                "active_classes": list(active_classes),   # 本轮活跃类别（未冻结）
                "frozen_classes": list(frozen_classes),    # 已冻结类别
                "stop_triggered": False,
                "elapsed_seconds": round_elapsed,
                "stage_times": round_stage_times,
            }
            all_rounds_results.append(round_record)
            final_round = iteration
            final_result6 = result6
            final_step5_stats = stats5

            # ── 暂停检查（asyncio.Event 机制，不退出后台线程）──
            if _check_paused(task_id):
                stop_reason = f"用户暂停（第{iteration}轮完成后）"
                round_record["stop_triggered"] = True
                round_record["stop_reason"] = stop_reason
                # 计算当前整体进度
                overall_pct = 15.0 + (len(all_rounds_results) / max_iterations) * 85.0
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PAUSED,
                    message=f"[第{iteration}轮] 已暂停，可点击恢复继续执行",
                    progress=min(99.0, overall_pct),
                    result={
                        "current_step": 3,
                        "step_message": f"已暂停（第{iteration}轮完成）",
                        "step_progress": 0,
                        "current_round": iteration,
                        "total_rounds_so_far": len(all_rounds_results),
                        "all_rounds": all_rounds_results,
                        "dashboard_metrics": dashboard_metrics,
                        "paused_after_round": iteration,
                        "timing": timing,
                    },
                )

                # ── 创建 Event 并阻塞等待用户操作 ──
                pause_event = _create_pause_event(task_id)
                print(f"[Pipeline] 流程已暂停（第{iteration}轮完成），等待用户操作...", flush=True)
                await pause_event.wait()  # 阻塞直到 resume/restart API 被调用

                # ── 检查用户选择的动作 ──
                with _lock:
                    action = _pause_actions.get(task_id, PauseAction.RESUME)
                _cleanup_pause_event(task_id)

                if action == PauseAction.RESTART:
                    # 用户选择重新开始 → 抛内部信号
                    task_manager.update_task(
                        task_id,
                        status=TaskStatus.RUNNING,
                        message="正在重新开始全流程...",
                        progress=0.0,
                        result={"current_step": 1, "step_message": "重新开始中...", "step_progress": 0, "current_round": 0},
                    )
                    print("[Pipeline] 用户选择重新开始，终止当前流程", flush=True)
                    raise RestartPipelineSignal()

                # action == RESUME：更新状态为 RUNNING，继续迭代
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.RUNNING,
                    message=f"[第{iteration + 1}轮] 已恢复执行，准备下一轮...",
                    progress=overall_pct,
                    result={
                        "current_step": 3,
                        "step_message": f"已恢复，准备第{iteration + 1}轮匹配...",
                        "step_progress": 0,
                        "current_round": iteration + 1,
                        "total_rounds_so_far": len(all_rounds_results),
                        "all_rounds": all_rounds_results,
                        "dashboard_metrics": dashboard_metrics,
                        "timing": timing,
                    },
                )
                print(f"[Pipeline] 已恢复执行，进入第{iteration + 1}轮迭代", flush=True)
                continue  # 进入下一轮迭代

            # ── 类别冻结判断（固定轮次模式下禁用，保证跑满设定轮数）──
            if not fixed_rounds:
                # 提取本轮各类别的 PA
                class_acc_map = {
                    ca["class_name"]: ca["producer_accuracy"]
                    for ca in result6.get("class_accuracies", [])
                }
                newly_frozen = set()
                for c in classes_data:
                    name = c["name"]
                    if name in frozen_classes:
                        continue  # 已冻结，跳过
                    pa = class_acc_map.get(name)
                    if pa is None:
                        continue
                    # 记录 PA 历史
                    class_pa_history[name].append(pa)

                    # 冻结策略：连续N轮PA增幅 < 0.5% 则冻结（N由用户设置）
                    # ★ 关键修复：PA=0%的类别不应被冻结——这说明该类别尚未被有效匹配，
                    # 冻结它只会让阈值停止增长，永远无法达到有效匹配。
                    # 只有当PA > 0时才考虑冻结。
                    stop_gap      = 0.5                    # 固定增幅阈值（%）
                    consecutive_r = consecutive_rounds  # 用户可配置，默认3

                    history = class_pa_history[name]
                    # 过滤掉 None 值（步骤6未能计算该类的PA时）
                    valid_history = [h for h in history if h is not None]
                    # ★ PA=0时不冻结：需要最近一次PA > 0才允许冻结判断
                    if pa > 0 and len(valid_history) >= consecutive_r + 1:
                        # 取最近 consecutive_r 次增幅：用索引 -consecutive_r 到 -1
                        recent_improvements = [
                            valid_history[-(consecutive_r - i)] - valid_history[-(consecutive_r - i) - 1]
                            for i in range(consecutive_r)
                        ]
                        if all(imp < stop_gap for imp in recent_improvements):
                            newly_frozen.add(name)

                if newly_frozen:
                    frozen_classes.update(newly_frozen)
                    # 更新活跃类别列表（从 match_order 中移除冻结的）
                    active_classes = [cn for cn in match_order if cn.replace('.csv', '') not in frozen_classes]
                    round_record["newly_frozen"] = list(newly_frozen)
                    round_record["active_classes"] = list(active_classes)

                # 如果所有类别都冻结，提前退出
                if not active_classes:
                    stop_reason = f"所有类别均已收敛，全流程结束"
                    round_record["stop_triggered"] = True
                    round_record["stop_reason"] = stop_reason
                    break

                # ── 判断终止条件（级联判断：先Kappa，再未分类）──
                if kappa >= kappa_threshold:
                    # 第一条件满足：Kappa达标 → 直接终止
                    stop_reason = f"Kappa={kappa:.4f} ≥ {kappa_threshold}，实验结束"
                    round_record["stop_triggered"] = True
                    round_record["stop_reason"] = stop_reason
                    break
                elif unclassified_ratio < unclassified_threshold:
                    # 第一条件不满足，但第二条件满足：未分类比例达标 → 终止
                    stop_reason = f"Kappa={kappa:.4f}<{kappa_threshold}，但未分类={unclassified_ratio:.1f}% < {unclassified_threshold}%，实验结束"
                    round_record["stop_triggered"] = True
                    round_record["stop_reason"] = stop_reason
                    break

                # 两个条件均不满足，通知继续
                condition_msg = f"Kappa={kappa:.4f}<{kappa_threshold}，未分类={unclassified_ratio:.1f}%≥{unclassified_threshold}%"
            else:
                # 固定轮次模式：不冻结、不提前终止，仅记录指标后继续下一轮
                condition_msg = f"固定轮次模式（共{max_iterations}轮）：Kappa={kappa:.4f}，未分类={unclassified_ratio:.1f}%"

            task_manager.update_task(
                task_id,
                status=TaskStatus.RUNNING,
                message=f"[{round_label}] {condition_msg}，继续下一轮...",
                result={
                    "current_step": 3,
                    "step_message": f"准备第{iteration+1}轮匹配...",
                    "step_progress": 0,
                    "current_round": iteration + 1,
                    "total_rounds_so_far": len(all_rounds_results),
                    "all_rounds": all_rounds_results,
                    "dashboard_metrics": dashboard_metrics,
                },
            )

        else:
            # for-else：达到最大轮次退出
            stop_reason = (
                f"已跑满固定轮次 {max_iterations} 轮，实验结束"
                if fixed_rounds
                else f"已达最大迭代轮次 {max_iterations}，强制终止"
            )
            if all_rounds_results:
                all_rounds_results[-1]["stop_triggered"] = True
                all_rounds_results[-1]["stop_reason"] = stop_reason

        # ────────────────────────────────────────────────────
        # 全流程完成
        # ────────────────────────────────────────────────────
        timing["overall_end"] = time.time()
        timing["overall_seconds"] = round(timing["overall_end"] - timing["overall_start"], 2)

        # 汇总步骤3-6各阶段总耗时（跨所有轮次累加）
        for key in ["step3", "step4", "step5", "step6"]:
            total = sum(rst.get(key, 0) for rst in timing["round_stage_times"])
            if total > 0:
                timing["stage_times"][key] = round(total, 2)

        report_path = _save_timing_report(output_dir, timing, stop_reason or "完成", final_round,
                                          distance_metric=distance_metric,
                                          hamming_tolerance=hamming_tolerance)
        timing["report_file"] = report_path

        final_metrics = final_result6.get("metrics") if final_result6 else None
        final_class_acc = final_result6.get("class_accuracies") if final_result6 else None

        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            message=f"全流程完成（共{final_round}轮）: {stop_reason}",
            progress=100.0,
            result={
                "current_step": 6,
                "stop_reason": stop_reason,
                "total_rounds": final_round,
                "all_rounds": all_rounds_results,
                "final_frozen_classes": list(frozen_classes),
                "final_active_classes": list(active_classes),
                "metrics": final_metrics,
                "class_accuracies": final_class_acc,
                "dashboard_metrics": dashboard_metrics,
                "timing": {
                    "overall_seconds": timing["overall_seconds"],
                    "stage_times": timing["stage_times"],
                    "round_times": timing["round_times"],
                    "report_file": report_path,
                },
                "outputs": {
                    "step1_csv": step1_csv,
                    "step2_dir": step2_out,
                    "final_round": final_round,
                    "final_round_dir": os.path.join(output_dir, f"round_{final_round}") if final_round > 0 else output_dir,
                },
            },
        )

    except RestartPipelineSignal:
        # ── 用户选择重新开始 → 递归重启 ──
        print(f"[Pipeline] 用户选择重新开始，重置状态后递归调用...", flush=True)
        task_manager.update_task(
            task_id,
            status=TaskStatus.RUNNING,
            message="全流程重新开始，正在初始化...",
            progress=0.0,
            result={"current_step": 1, "step_message": "重新开始中...", "step_progress": 0, "current_round": 0},
        )
        # 递归调用自身（会覆盖旧状态）
        await _run_pipeline_background(
            task_id=task_id,
            input_img_dir=input_img_dir,
            output_dir=output_dir,
            filename=filename,
            num_classes=num_classes,
            delete_ranges=delete_ranges,
            adjust_sizes=adjust_sizes,
            shp_input=shp_input,
            tif_folder=tif_folder,
            target_file=target_file,
            threshold_mode=threshold_mode,
            initial_threshold=initial_threshold,
            threshold_step=threshold_step,
            auto_threshold_samples=auto_threshold_samples,
            add_geo=add_geo,
            max_sequence_length=max_sequence_length,
            classes_data=classes_data,
            shp_folder=shp_folder,
            kappa_threshold=kappa_threshold,
            unclassified_threshold=unclassified_threshold,
            max_iterations=max_iterations,
            consecutive_rounds=consecutive_rounds,
            existing_step1_csv=existing_step1_csv,
            execution_mode=execution_mode,
            num_processes=num_processes,
            skip_discretization=skip_discretization,
            normalize_continuous=normalize_continuous,
            distance_metric=distance_metric,
            zero_policy=zero_policy,
            hamming_tolerance=hamming_tolerance,
        )
        return  # 递归完成后退出

    except Exception as e:
        error_msg = str(e)
        print(f"[Pipeline Error] {error_msg}")
        traceback.print_exc()

        # 失败时也记录已运行时间（安全访问，防止变量未初始化）
        try:
            if 'timing' in dir() and timing:
                timing["overall_end"] = time.time()
                timing["overall_seconds"] = round(timing["overall_end"] - timing.get("overall_start", time.time()), 2)
                for key in ["step3", "step4", "step5", "step6"]:
                    total = sum(rst.get(key, 0) for rst in timing.get("round_stage_times", []))
                    if total > 0:
                        timing["stage_times"][key] = round(total, 2)
                result_timing = {
                    "overall_seconds": timing["overall_seconds"],
                    "stage_times": timing.get("stage_times", {}),
                    "round_times": timing.get("round_times", []),
                    "report_file": timing.get("report_file"),
                }
            else:
                result_timing = None
        except Exception:
            result_timing = None

        rounds_list = all_rounds_results if 'all_rounds_results' in dir() else []
        metrics = dashboard_metrics if 'dashboard_metrics' in dir() else {}

        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=f"全流程失败: {error_msg}",
            error=error_msg,
            result={
                "all_rounds": rounds_list,
                "current_round": len(rounds_list),
                "dashboard_metrics": metrics,
                "timing": result_timing,
            },
        )
