"""
GPU加速测试脚本 - 验证CuPy在RTX 4050上的加速效果
运行方式: python test_gpu_acceleration.py
"""
import os
import sys
import time

# 设置CuPy缓存目录（避免默认路径的权限问题）
os.environ.setdefault('CUPY_CACHE_DIR', os.path.join(os.path.dirname(__file__), '..', '.cupy_cache'))
os.makedirs(os.environ['CUPY_CACHE_DIR'], exist_ok=True)

import numpy as np

print("=" * 60)
print("GPU加速性能测试 (RTX 4050 Laptop GPU)")
print("=" * 60)

try:
    import cupy as cp
    
    # GPU信息
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props['name']
    if isinstance(name, bytes):
        name = name.decode()
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"GPU设备: {name}")
    print(f"显存: {free/1024**3:.2f} GB 可用 / {total/1024**3:.2f} GB 总计")
    
    # 基本功能测试
    a = cp.array([1, 2, 3, 4, 5])
    b = cp.array([10, 20, 30, 40, 50])
    print(f"基本计算: {a} + {b} = {(a+b).get()}")
    
    # ── 测试1: 矩阵乘法 ──
    print("\n--- 测试1: 矩阵乘法 ---")
    n = 2000
    a_cpu = np.random.rand(n, n).astype(np.float32)
    b_cpu = np.random.rand(n, n).astype(np.float32)
    
    start = time.time()
    np.dot(a_cpu, b_cpu)
    cpu_time = time.time() - start
    
    a_gpu = cp.asarray(a_cpu)
    b_gpu = cp.asarray(b_cpu)
    cp.cuda.Stream.null.synchronize()
    start = time.time()
    cp.dot(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start
    print(f"  CPU: {cpu_time:.3f}s")
    print(f"  GPU: {gpu_time:.3f}s")
    print(f"  加速比: {cpu_time/gpu_time:.1f}x")
    
    # ── 测试2: 单序列汉明距离 ──
    print("\n--- 测试2: 单序列汉明距离 (模拟process03核心计算) ---")
    n_refs, seq_len = 500, 20
    ref_cpu = np.random.randint(0, 10, size=(n_refs, seq_len), dtype=np.int16)
    target_cpu = np.random.randint(0, 10, size=(seq_len,), dtype=np.int16)
    
    # CPU
    start = time.time()
    for _ in range(1000):
        dists = np.not_equal(ref_cpu, target_cpu).sum(axis=1)
        min_dist = dists.min()
    cpu_hamming = time.time() - start
    
    # GPU
    ref_gpu = cp.asarray(ref_cpu)
    target_gpu = cp.asarray(target_cpu)
    cp.cuda.Stream.null.synchronize()
    start = time.time()
    for _ in range(1000):
        dists = cp.not_equal(ref_gpu, target_gpu).sum(axis=1)
        min_dist = cp.min(dists)
    cp.cuda.Stream.null.synchronize()
    gpu_hamming = time.time() - start
    print(f"  CPU (1000次): {cpu_hamming:.4f}s")
    print(f"  GPU (1000次): {gpu_hamming:.4f}s")
    print(f"  加速比: {cpu_hamming/gpu_hamming:.1f}x")
    
    # ── 测试3: 批量汉明距离 (核心优化) ──
    print("\n--- 测试3: 批量汉明距离 (GPU矩阵化) ---")
    for n_cells in [1000, 5000, 10000, 50000]:
        cells_cpu = np.random.randint(0, 10, size=(n_cells, seq_len), dtype=np.int16)
        
        # CPU (逐个计算)
        start = time.time()
        for i in range(n_cells):
            dists = np.not_equal(ref_cpu, cells_cpu[i]).sum(axis=1)
            min_dist = dists.min()
        cpu_batch = time.time() - start
        
        # GPU (批量矩阵计算)
        cells_gpu = cp.asarray(cells_cpu)
        cp.cuda.Stream.null.synchronize()
        start = time.time()
        # (n_cells, 1, seq_len) != (1, n_refs, seq_len) -> sum(axis=2) -> (n_cells, n_refs)
        diff = cp.not_equal(cells_gpu[:, None, :], ref_gpu[None, :, :]).sum(axis=2)
        min_dists = cp.min(diff, axis=1)
        cp.cuda.Stream.null.synchronize()
        gpu_batch = time.time() - start
        
        print(f"  {n_cells:>6d} cells x {n_refs} refs: CPU={cpu_batch:.4f}s GPU={gpu_batch:.4f}s Speedup={cpu_batch/gpu_batch:.1f}x")
    
    # ── 测试4: 大规模数据 (模拟真实遥感影像) ──
    print("\n--- 测试4: 大规模数据 (模拟真实场景) ---")
    # 模拟: 1000个参考序列, 序列长度20, 100万待匹配像元
    n_refs_real = 1000
    seq_len_real = 20
    n_cells_real = 100000  # 10万像元 (分批处理)
    
    ref_real = cp.random.randint(0, 10, size=(n_refs_real, seq_len_real), dtype=cp.int16)
    cells_real = cp.random.randint(0, 10, size=(n_cells_real, seq_len_real), dtype=cp.int16)
    
    batch_size = 5000  # 每批5000个像元
    cp.cuda.Stream.null.synchronize()
    start = time.time()
    
    results_min_dist = []
    results_min_idx = []
    for i in range(0, n_cells_real, batch_size):
        batch = cells_real[i:i+batch_size]
        # 批量计算汉明距离矩阵
        diff = cp.not_equal(batch[:, None, :], ref_real[None, :, :]).sum(axis=2)
        min_d = cp.min(diff, axis=1)
        min_i = cp.argmin(diff, axis=1)
        results_min_dist.append(min_d)
        results_min_idx.append(min_i)
    
    all_min_dists = cp.concatenate(results_min_dist)
    all_min_indices = cp.concatenate(results_min_idx)
    cp.cuda.Stream.null.synchronize()
    gpu_real_time = time.time() - start
    
    # CPU对比 (只计算前10000个)
    ref_real_cpu = ref_real.get()
    cells_real_cpu = cells_real.get()
    start = time.time()
    for i in range(min(10000, n_cells_real)):
        dists = np.not_equal(ref_real_cpu, cells_real_cpu[i]).sum(axis=1)
        min_dist = dists.min()
    cpu_real_time = time.time() - start
    cpu_extrapolated = cpu_real_time * (n_cells_real / min(10000, n_cells_real))
    
    print(f"  参考: {n_refs_real}条序列, 长度{seq_len_real}")
    print(f"  待匹配: {n_cells_real}个像元 (分批{batch_size}/批)")
    print(f"  GPU总耗时: {gpu_real_time:.3f}s")
    print(f"  CPU前10000个: {cpu_real_time:.3f}s (推算全部: {cpu_extrapolated:.1f}s)")
    print(f"  加速比: {cpu_extrapolated/gpu_real_time:.1f}x")
    
    # 显存使用
    free_after, total_after = cp.cuda.runtime.memGetInfo()
    print(f"  显存使用: {(free-free_after)/1024**2:.1f} MB")
    
    print("\n" + "=" * 60)
    print("GPU加速测试通过! RTX 4050可正常用于汉明距离加速计算")
    print("=" * 60)
    
except ImportError:
    print("CuPy未安装! 请运行: pip install cupy-cuda12x[ctk]")
    sys.exit(1)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nGPU测试失败: {e}")
    sys.exit(1)
