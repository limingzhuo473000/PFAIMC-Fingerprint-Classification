"""
GPU加速优化模块 v2.0 - 针对RTX 4050 Laptop GPU优化
===============================================

核心优化：
1. 自动创建ASCII路径junction（解决中文路径下NVRTC编译失败问题）
2. 预编译GPU内核（避免首次运行时17秒编译开销）
3. 批量汉明距离计算（矩阵化，比逐元素快5-8倍）
4. 显存管理（自动分批，防止OOM）
5. CPU/GPU自动切换（GPU不可用时无缝回退）

依赖: cupy-cuda12x[ctk]
"""

import os
import sys
import time
import shutil
import numpy as np

# ============================================================
# 全局状态
# ============================================================
USE_GPU = False
cp = None
_gpu_info = {
    'name': 'N/A',
    'vram_total_mb': 0,
    'vram_free_mb': 0,
    'cuda_version': 'N/A',
    'cupy_version': 'N/A',
    'junction_path': None,
    'init_time': 0,
}

# ASCII junction路径（解决中文路径下NVRTC编译失败）
_JUNCTION_TARGET = None
_JUNCTION_PATH = None


def _get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_junction():
    """
    创建ASCII路径junction，解决中文路径下NVRTC无法找到头文件的问题。
    
    CuPy使用NVRTC在运行时编译CUDA内核，NVRTC的文件API在Windows上
    无法处理非ASCII路径。通过创建一个ASCII路径的junction链接，
    让CuPy通过ASCII路径访问头文件。
    """
    global _JUNCTION_TARGET, _JUNCTION_PATH
    
    _JUNCTION_TARGET = _get_project_root()
    
    # 检查路径是否已经是纯ASCII
    try:
        _JUNCTION_TARGET.encode('ascii')
        # 纯ASCII路径，不需要junction
        _JUNCTION_PATH = _JUNCTION_TARGET
        return _JUNCTION_PATH
    except UnicodeEncodeError:
        pass
    
    # 需要创建junction
    junction_name = 'rs_gpu'
    _JUNCTION_PATH = os.path.join('C:\\', junction_name)
    
    # 检查junction是否已存在且有效
    if os.path.exists(_JUNCTION_PATH):
        # 验证junction指向正确的目标
        test_file = os.path.join(_JUNCTION_PATH, 'backend', 'scripts', 'gpu_optimize.py')
        if os.path.exists(test_file):
            return _JUNCTION_PATH
        else:
            # junction无效，删除重建
            try:
                os.rmdir(_JUNCTION_PATH)
            except Exception:
                pass
    
    # 创建junction
    try:
        import subprocess
        result = subprocess.run(
            ['cmd', '/c', 'mklink', '/J', _JUNCTION_PATH, _JUNCTION_TARGET],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"[GPU加速] ASCII junction已创建: {_JUNCTION_PATH} -> {_JUNCTION_TARGET}")
            return _JUNCTION_PATH
        else:
            print(f"[GPU加速] junction创建失败: {result.stderr}")
    except Exception as e:
        print(f"[GPU加速] junction创建异常: {e}")
    
    # fallback: 使用原始路径
    _JUNCTION_PATH = _JUNCTION_TARGET
    return _JUNCTION_PATH


def _setup_cache_dirs():
    """设置CuPy和CUDA缓存目录
    
    注意: 必须在 import cupy 之前调用，因为CuPy在导入时会读取这些环境变量。
    使用强制赋值（非setdefault），确保覆盖bash等外部环境可能设置的无效路径。
    """
    project_root = _get_project_root()
    
    # CuPy内核缓存目录（NVRTC编译缓存）
    cupy_cache = os.path.join(project_root, '.cupy_cache')
    os.makedirs(cupy_cache, exist_ok=True)
    os.environ['CUPY_CACHE_DIR'] = cupy_cache  # 强制赋值，覆盖可能的无效bash路径
    
    # CUDA编译缓存目录
    cuda_cache = os.path.join(project_root, '.cuda_cache')
    os.makedirs(cuda_cache, exist_ok=True)
    os.environ['CUDA_CACHE_PATH'] = cuda_cache
    
    # NVIDIA ComputeCache重定向
    nv_cache = os.path.join(project_root, '.nv_cache')
    os.makedirs(nv_cache, exist_ok=True)
    os.environ['CUDA_DEVICE_CACHE_PATH'] = nv_cache
    
    return cupy_cache


def _precompile_kernels():
    """
    预编译GPU内核，避免首次使用时的NVRTC编译开销。
    
    CuPy使用NVRTC在运行时编译CUDA代码，首次编译需要数秒。
    预编译后，后续调用直接使用缓存的内核。
    """
    if not USE_GPU or cp is None:
        return
    
    try:
        print("[GPU加速] 预编译GPU内核...", flush=True)
        start = time.time()
        
        # 预编译int16数组的not_equal + sum内核（汉明距离核心）
        ref_test = cp.array([[1, 2, 3], [4, 5, 6]], dtype=cp.int16)
        target_test = cp.array([1, 2, 3], dtype=cp.int16)
        
        # 触发not_equal内核编译
        _ = cp.not_equal(ref_test, target_test)
        
        # 触发sum内核编译
        _ = cp.not_equal(ref_test, target_test).sum(axis=1)
        
        # 触发min内核编译
        _ = cp.min(cp.array([1, 2, 3], dtype=cp.int16))
        
        # 触发argmin内核编译
        _ = cp.argmin(cp.array([1, 2, 3], dtype=cp.int16))
        
        # 触发unique内核编译
        _ = cp.unique(cp.array([1, 2, 3], dtype=cp.int32))
        
        # 触发3D广播运算内核（批量汉明距离）
        cells_test = cp.array([[1, 2, 3], [4, 5, 6]], dtype=cp.int16)
        _ = cp.not_equal(cells_test[:, None, :], ref_test[None, :, :]).sum(axis=2)
        
        # 触发矩阵乘法内核
        mat_test = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        _ = cp.dot(mat_test, mat_test)
        
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start
        print(f"[GPU加速] 内核预编译完成 ({elapsed:.1f}s)", flush=True)
        
    except Exception as e:
        print(f"[GPU加速] 内核预编译警告: {e}", flush=True)


def init_gpu():
    """
    初始化GPU支持。
    
    流程:
    1. 创建ASCII junction（解决中文路径问题）
    2. 设置缓存目录
    3. 导入CuPy并测试GPU
    4. 预编译内核
    5. 记录GPU信息
    
    返回: True if GPU available, False otherwise
    """
    global USE_GPU, cp
    import threading
    
    # 防止多线程重复初始化
    if not hasattr(init_gpu, '_lock'):
        init_gpu._lock = threading.Lock()
    
    with init_gpu._lock:
        if USE_GPU:
            return True
        
        start = time.time()
        
        try:
            # Step 1: 创建junction
            _ensure_junction()
            
            # Step 2: 设置缓存目录
            _setup_cache_dirs()
            
            # Step 3: 导入CuPy
            import cupy as _cp
            cp = _cp
            
            # Step 4: 测试GPU基本功能
            test_arr = cp.array([1, 2, 3])
            _ = test_arr + 1
            cp.cuda.Stream.null.synchronize()
            
            # Step 5: 收集GPU信息
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props.get('name', b'Unknown')
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='replace')
            
            free, total = cp.cuda.runtime.memGetInfo()
            cuda_ver = cp.cuda.runtime.runtimeGetVersion()
            
            _gpu_info['name'] = name
            _gpu_info['vram_total_mb'] = int(total / 1024 / 1024)
            _gpu_info['vram_free_mb'] = int(free / 1024 / 1024)
            _gpu_info['cuda_version'] = f"{cuda_ver // 1000}.{(cuda_ver % 1000) // 10}"
            _gpu_info['cupy_version'] = cp.__version__
            _gpu_info['junction_path'] = _JUNCTION_PATH
            _gpu_info['init_time'] = time.time() - start
            
            USE_GPU = True
            
            print(f"[GPU加速] GPU初始化成功!", flush=True)
            print(f"  设备: {name}", flush=True)
            print(f"  显存: {_gpu_info['vram_free_mb']} MB / {_gpu_info['vram_total_mb']} MB", flush=True)
            print(f"  CUDA: {_gpu_info['cuda_version']}, CuPy: {_gpu_info['cupy_version']}", flush=True)
            print(f"  Junction: {_JUNCTION_PATH}", flush=True)
            
            # Step 6: 预编译内核
            _precompile_kernels()
            
            return True
            
        except ImportError:
            print("[GPU加速] CuPy未安装，使用CPU模式", flush=True)
            print("[GPU加速] 安装命令: pip install cupy-cuda12x[ctk]", flush=True)
            USE_GPU = False
            return False
        except Exception as e:
            print(f"[GPU加速] GPU初始化失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            USE_GPU = False
            return False


# ============================================================
# 基础工具函数
# ============================================================

def to_gpu(array_cpu):
    """将CPU数组转移到GPU"""
    if USE_GPU and cp is not None:
        return cp.asarray(array_cpu)
    return array_cpu


def to_cpu(array):
    """将数组转移回CPU"""
    if USE_GPU and cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array


def is_gpu_array(array):
    """检查数组是否在GPU上"""
    if USE_GPU and cp is not None:
        return isinstance(array, cp.ndarray)
    return False


def get_gpu_info():
    """获取GPU信息字典"""
    if USE_GPU and cp is not None:
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            _gpu_info['vram_free_mb'] = int(free / 1024 / 1024)
            _gpu_info['vram_total_mb'] = int(total / 1024 / 1024)
        except Exception:
            pass
    return dict(_gpu_info)


def free_gpu_memory():
    """释放GPU缓存显存"""
    if USE_GPU and cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


# ============================================================
# 核心: 批量汉明距离计算 (GPU矩阵化)
# ============================================================

def batch_hamming_distance_gpu(
    cells_matrix,     # (n_cells, seq_len) np.int16/np.float32 - 待匹配像元序列
    ref_matrix,       # (n_refs, seq_len) cp.ndarray or np.int16/np.float32 - 参考序列矩阵
    repl_ids=None,    # (n_refs,) cp.ndarray or np.int32 - 每个参考序列的类别ID
    batch_size=5000,  # 每批处理的像元数
    threshold=None,   # 距离阈值，超过则不匹配
    progress_callback=None,  # 进度回调 fn(current, total)
    distance_metric='hamming'  # 距离度量: 'hamming' / 'euclidean'（消融实验2）
):
    """
    GPU批量距离计算 - 核心加速函数（支持汉明/欧氏距离）

    算法:
        对每个像元序列，计算到所有参考序列的距离，
        找到最小距离和对应的类别。

    矩阵化:
        hamming:
            diff = (n_cells, 1, seq_len) != (1, n_refs, seq_len)  →  (n_cells, n_refs, seq_len)
            dists = diff.sum(axis=2)  →  (n_cells, n_refs)
        euclidean:
            diff = (n_cells, 1, seq_len) - (1, n_refs, seq_len)
            dists = sqrt((diff*diff).sum(axis=2))
        min_dists = dists.min(axis=1)  →  (n_cells,)
        min_indices = dists.argmin(axis=1)  →  (n_cells,)

    参数:
        cells_matrix: 待匹配像元序列矩阵 (n_cells, seq_len)
        ref_matrix: 参考序列矩阵 (n_refs, seq_len)
        repl_ids: 每个参考序列对应的类别ID (n_refs,)
        batch_size: 每批处理的像元数 (控制显存使用)
        threshold: 距离阈值，超过则标记为不匹配（汉明:整数位差数 / 欧氏:连续距离值）
        progress_callback: 进度回调函数
        distance_metric: 'hamming'（汉明距离）或 'euclidean'（欧氏距离，消融实验2）

    返回:
        dict with keys:
            'min_distances': (n_cells,) np.int32/np.float32 - 每个像元的最小距离
            'min_indices': (n_cells,) np.int32 - 最小距离对应的参考序列索引
            'matched_repl_ids': (n_cells,) np.int32 - 匹配的类别ID (-1表示未匹配)
            'ambiguous': (n_cells,) np.bool_ - 是否歧义(多个类别有相同最小距离)
            'gpu_time': float - GPU计算耗时(秒)
    """
    if not USE_GPU or cp is None:
        raise RuntimeError("GPU not available")

    n_cells = cells_matrix.shape[0]
    n_refs = ref_matrix.shape[0]
    seq_len = ref_matrix.shape[1]

    is_euclidean = (distance_metric == 'euclidean')

    start = time.time()

    # 将参考矩阵放到GPU（整个计算过程中保持不变）
    if not isinstance(ref_matrix, cp.ndarray):
        ref_gpu = cp.asarray(ref_matrix)
    else:
        ref_gpu = ref_matrix

    if repl_ids is not None and not isinstance(repl_ids, cp.ndarray):
        repl_ids_gpu = cp.asarray(repl_ids)
    else:
        repl_ids_gpu = repl_ids

    # 预分配结果数组（欧氏距离为连续值，用float32存储）
    dist_dtype = cp.float32 if is_euclidean else cp.int32
    all_min_dists = cp.empty(n_cells, dtype=dist_dtype)
    all_min_indices = cp.empty(n_cells, dtype=cp.int32)
    matched_repl_ids = cp.full(n_cells, -1, dtype=cp.int32)
    ambiguous = cp.zeros(n_cells, dtype=cp.bool_)

    if threshold is not None:
        all_within = cp.empty(n_cells, dtype=cp.bool_)

    # 向量化歧义检查常量
    INT_MAX = cp.iinfo(cp.int32).max
    INT_MIN = cp.iinfo(cp.int32).min

    # 单次遍历: 同时计算最小距离 + 歧义检查（消除冗余距离矩阵重算）
    # 每批的3D张量大小: batch_size * n_refs * seq_len * (bool 1字节 / float32 4字节)
    # 对于 batch=5000, n_refs=1000, seq_len=20: 5000*1000*20 = 100MB (可接受)
    for i in range(0, n_cells, batch_size):
        end = min(i + batch_size, n_cells)
        batch = cp.asarray(cells_matrix[i:end])
        batch_len = end - i

        # 核心: 批量距离矩阵计算
        if is_euclidean:
            # 欧氏距离（消融实验2）
            # (batch_len, 1, seq_len) - (1, n_refs, seq_len) → (batch_len, n_refs, seq_len)
            # (diff*diff).sum(axis=2) → sqrt → (batch_len, n_refs)
            diff = batch[:, None, :] - ref_gpu[None, :, :]
            dists = cp.sqrt((diff * diff).sum(axis=2, dtype=cp.float32))
        else:
            # 汉明距离
            # (batch_len, 1, seq_len) != (1, n_refs, seq_len) → (batch_len, n_refs, seq_len)
            # sum(axis=2) → (batch_len, n_refs)
            diff = cp.not_equal(batch[:, None, :], ref_gpu[None, :, :])
            dists = diff.sum(axis=2, dtype=cp.int32)

        # 释放中间变量
        del diff

        # 最小距离和索引
        min_d = cp.min(dists, axis=1)           # (batch_len,)
        min_idx = cp.argmin(dists, axis=1)       # (batch_len,)
        all_min_dists[i:end] = min_d
        all_min_indices[i:end] = min_idx

        # 阈值检查
        if threshold is not None:
            within = min_d <= threshold          # (batch_len,) bool
            all_within[i:end] = within
        else:
            within = cp.ones(batch_len, dtype=cp.bool_)

        # 向量化歧义检查 (替代Python for循环)
        # 找到所有等于最小距离的参考序列: eq_mask (batch_len, n_refs)
        eq_mask = (dists == min_d[:, None])

        if repl_ids_gpu is not None:
            # 广播repl_ids: (1, n_refs) → (batch_len, n_refs)
            repl_broadcast = repl_ids_gpu[cp.newaxis, :]

            # 在匹配位置取repl_id，非匹配位置取极值
            # min_id: 匹配参考序列中的最小类别ID
            # max_id: 匹配参考序列中的最大类别ID
            # 若 min_id != max_id → 歧义
            masked_for_min = cp.where(eq_mask, repl_broadcast, INT_MAX)
            masked_for_max = cp.where(eq_mask, repl_broadcast, INT_MIN)
            min_ids = masked_for_min.min(axis=1)    # (batch_len,)
            max_ids = masked_for_max.max(axis=1)    # (batch_len,)

            amb_batch = (min_ids != max_ids) & within  # 在阈值内且有歧义

            # 非歧义且在阈值内 → 设置matched_repl_id
            matched_mask = within & (~amb_batch)
            matched_repl_ids[i:end] = cp.where(matched_mask, min_ids, -1)
            ambiguous[i:end] = amb_batch

            del masked_for_min, masked_for_max, min_ids, max_ids, amb_batch, matched_mask

        del batch, dists, min_d, min_idx, eq_mask

        if progress_callback:
            progress_callback(end, n_cells)

    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start
    
    # 传回CPU
    result = {
        'min_distances': cp.asnumpy(all_min_dists),
        'min_indices': cp.asnumpy(all_min_indices),
        'matched_repl_ids': cp.asnumpy(matched_repl_ids),
        'ambiguous': cp.asnumpy(ambiguous),
        'gpu_time': gpu_time,
    }
    
    # 清理GPU内存
    del all_min_dists, all_min_indices, matched_repl_ids, ambiguous
    if repl_ids_gpu is not None:
        del repl_ids_gpu
    # ref_gpu 由调用者管理
    free_gpu_memory()
    
    return result


def hamming_distance_single_gpu(
    target_seq,    # (seq_len,) - 单个像元序列
    ref_matrix,    # (n_refs, seq_len) - 参考矩阵 (已在GPU上)
    repl_ids=None  # (n_refs,) - 类别ID
):
    """
    GPU计算单个像元到所有参考序列的汉明距离。
    
    注意: 对于批量计算，请使用 batch_hamming_distance_gpu() 效率高得多。
    此函数仅供小规模或调试使用。
    
    返回:
        min_dist: 最小汉明距离
        matched_repl_id: 匹配的类别ID (-1 if ambiguous or no match)
        is_ambiguous: 是否歧义
    """
    if not USE_GPU or cp is None:
        raise RuntimeError("GPU not available")
    
    if not isinstance(ref_matrix, cp.ndarray):
        ref_matrix = cp.asarray(ref_matrix)
    
    target_gpu = cp.asarray(target_seq) if not isinstance(target_seq, cp.ndarray) else target_seq
    
    # 汉明距离
    dists = cp.not_equal(ref_matrix, target_gpu).sum(axis=1)
    min_dist = int(cp.min(dists).get())
    
    # 检查歧义
    min_mask = dists == min_dist
    if repl_ids is not None:
        if not isinstance(repl_ids, cp.ndarray):
            repl_ids = cp.asarray(repl_ids)
        matched = cp.unique(repl_ids[min_mask]).get()
    else:
        matched = cp.unique(min_mask.nonzero()[0]).get()
    
    is_ambiguous = len(matched) > 1
    matched_id = int(matched[0]) if len(matched) == 1 else -1
    
    return min_dist, matched_id, is_ambiguous


# ============================================================
# 自动批量大小估算
# ============================================================

def estimate_batch_size(n_refs, seq_len, safety_factor=0.7, distance_metric='hamming'):
    """
    根据可用显存估算最优批量大小。

    3D张量内存 = batch_size * n_refs * seq_len * diff_bytes
                 (hamming: bool 1字节 / euclidean: float32 4字节)
    加上距离矩阵 = batch_size * n_refs * 4 bytes (int32/float32)

    参数:
        n_refs: 参考序列数量
        seq_len: 序列长度
        safety_factor: 安全系数 (0-1)
        distance_metric: 'hamming' / 'euclidean'（欧氏距离中间张量为float32，显存占用更大）

    返回: 建议的batch_size
    """
    if not USE_GPU or cp is None:
        return 5000  # CPU默认

    # 汉明: bool 1字节；欧氏: float32 4字节
    diff_bytes = 4 if distance_metric == 'euclidean' else 1

    try:
        free, _ = cp.cuda.runtime.memGetInfo()
        free_bytes = free * safety_factor

        # 每个像元需要的显存:
        # 1. 3D diff张量: n_refs * seq_len * diff_bytes
        # 2. 距离矩阵: n_refs * 4 bytes
        # 3. 输入数据: seq_len * 4 bytes (float32/int32)
        per_cell_bytes = n_refs * seq_len * diff_bytes + n_refs * 4 + seq_len * 4

        batch_size = int(free_bytes / per_cell_bytes)
        batch_size = max(100, min(batch_size, 50000))  # 限制在100-50000

        return batch_size
    except Exception:
        return 5000


# ============================================================
# 初始化
# ============================================================

# 模块加载时不自动初始化（延迟到首次使用时）
# 避免导入模块时的副作用


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GPU加速模块 v2.0 测试")
    print("=" * 60)
    
    init_gpu()
    
    print(f"\nGPU加速: {'已启用' if USE_GPU else '未启用'}")
    info = get_gpu_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    if USE_GPU:
        print("\n性能测试: 批量汉明距离")
        
        n_refs, seq_len = 500, 20
        ref = np.random.randint(0, 10, size=(n_refs, seq_len), dtype=np.int16)
        repl = np.random.randint(0, 5, size=(n_refs,), dtype=np.int32)
        
        for n_cells in [1000, 5000, 10000, 50000]:
            cells = np.random.randint(0, 10, size=(n_cells, seq_len), dtype=np.int16)
            
            # GPU
            result = batch_hamming_distance_gpu(
                cells, ref, repl, batch_size=5000, threshold=10
            )
            
            # CPU对比
            start = time.time()
            for i in range(n_cells):
                d = np.not_equal(ref, cells[i]).sum(axis=1)
            cpu_time = time.time() - start
            
            print(f"  {n_cells:>6d} cells: GPU={result['gpu_time']:.4f}s "
                  f"CPU={cpu_time:.4f}s Speedup={cpu_time/result['gpu_time']:.1f}x")
    
    print("=" * 60)
