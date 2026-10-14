"""
GPU加速测试脚本 - 检测GPU可用性并测试性能提升
"""
import time
import numpy as np

print("=" * 60)
print("GPU加速性能测试")
print("=" * 60)

# 1. 检测CuPy (NVIDIA GPU)
print("\n1. 检测NVIDIA GPU (CuPy)...")
try:
    import cupy as cp
    print(f"   [OK] CuPy已安装！")
    print(f"   GPU设备: {cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['name'].decode()}")
    print(f"   CUDA版本: {cp.cuda.runtime.runtimeGetVersion()}")
    HAS_GPU = True
except ImportError:
    print("   [FAIL] CuPy未安装")
    HAS_GPU = False
except Exception as e:
    print(f"   [FAIL] GPU检测失败: {e}")
    HAS_GPU = False

# 2. 性能对比测试
if HAS_GPU:
    print("\n2. 性能对比测试（矩阵运算）...")
    
    # 创建测试数据
    print("   生成测试数据...")
    size = 5000
    a_cpu = np.random.rand(size, size).astype(np.float32)
    b_cpu = np.random.rand(size, size).astype(np.float32)
    
    # CPU测试
    print("   CPU计算中...")
    start = time.time()
    c_cpu = np.dot(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f"   [OK] CPU耗时: {cpu_time:.2f}秒")
    
    # GPU测试
    print("   GPU计算中...")
    a_gpu = cp.asarray(a_cpu)
    b_gpu = cp.asarray(b_cpu)
    
    # 预热GPU
    _ = cp.dot(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()
    
    start = time.time()
    c_gpu = cp.dot(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start
    print(f"   [OK] GPU耗时: {gpu_time:.2f}秒")
    
    print(f"\n   [INFO] 加速比: {cpu_time/gpu_time:.1f}x 倍加速！")
    
    # 汉明距离测试（你的系统核心计算）
    print("\n3. 汉明距离计算测试（你的系统核心操作）...")
    
    # 模拟你的数据：1000条参考序列，每条长度16
    n_refs = 1000
    seq_len = 16
    
    ref_matrix = np.random.randint(0, 5, (n_refs, seq_len))
    current = np.random.randint(0, 5, seq_len)
    
    # CPU版本
    print("   CPU计算汉明距离...")
    start = time.time()
    for _ in range(100):  # 重复100次
        dists = np.not_equal(ref_matrix, current).sum(axis=1)
        min_dist = int(dists.min())
    cpu_time2 = time.time() - start
    print(f"   [OK] CPU耗时: {cpu_time2:.2f}秒 (100次)")
    
    # GPU版本
    print("   GPU计算汉明距离...")
    ref_matrix_gpu = cp.asarray(ref_matrix)
    current_gpu = cp.asarray(current)
    
    start = time.time()
    for _ in range(100):
        dists = cp.not_equal(ref_matrix_gpu, current_gpu).sum(axis=1)
        min_dist = int(cp.min(dists).get())
    gpu_time2 = time.time() - start
    print(f"   [OK] GPU耗时: {gpu_time2:.2f}秒 (100次)")
    
    print(f"\n   [INFO] 汉明距离加速比: {cpu_time2/gpu_time2:.1f}x 倍加速！")
    
else:
    print("\n[FAIL] 未检测到GPU，将优化CPU版本")
    print("\n优化建议：")
    print("1. 安装CuPy: pip install cupy-cuda12x")
    print("2. 或优化多进程参数（你的CPU有16核）")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
