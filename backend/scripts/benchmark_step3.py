"""
性能测试脚本 - 分析步骤3的瓶颈
运行此脚本可以了解当前算法的性能瓶颈
"""

import time
import numpy as np
import sys
from pathlib import Path

# 添加backend到路径
backend_dir = str(Path(__file__).parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

def test_matching_performance():
    """测试不同批处理大小的性能"""
    
    print("=" * 60)
    print("步骤3 性能测试")
    print("=" * 60)
    
    # 模拟数据
    n_pixels = 10000  # 1万个像素
    n_refs = 500      # 500条参考序列
    seq_len = 7       # 序列长度
    
    print(f"\n测试配置:")
    print(f"  像素数: {n_pixels}")
    print(f"  参考序列数: {n_refs}")
    print(f"  序列长度: {seq_len}")
    
    # 生成随机测试数据
    print("\n生成测试数据...")
    target_data = np.random.randint(0, 100, size=(n_pixels, seq_len), dtype=np.int16)
    ref_matrix = np.random.randint(0, 100, size=(n_refs, seq_len), dtype=np.int16)
    repl_ids = np.random.randint(0, 10, size=n_refs, dtype=np.int32)
    threshold = 15
    
    # 测试1: 逐像素处理（当前方式）
    print("\n" + "=" * 60)
    print("测试1: 逐像素处理（当前方式）")
    print("=" * 60)
    
    start = time.time()
    results_slow = []
    for i in range(n_pixels):
        current = target_data[i]
        dists = np.sum(ref_matrix != current, axis=1)
        min_dist = np.min(dists)
        results_slow.append(min_dist)
    elapsed_slow = time.time() - start
    print(f"耗时: {elapsed_slow:.2f}s")
    print(f"速度: {n_pixels / elapsed_slow:.0f} 像素/秒")
    
    # 测试2: 批处理（优化方式）
    print("\n" + "=" * 60)
    print("测试2: 批处理（优化方式）")
    print("=" * 60)
    
    batch_size = 500
    start = time.time()
    results_fast = []
    
    for start_idx in range(0, n_pixels, batch_size):
        end_idx = min(start_idx + batch_size, n_pixels)
        batch = target_data[start_idx:end_idx]
        
        # 向量化计算: (batch_size, n_refs)
        dists = np.sum(batch[:, np.newaxis, :] != ref_matrix[np.newaxis, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        results_fast.extend(min_dists)
    
    elapsed_fast = time.time() - start
    print(f"耗时: {elapsed_fast:.2f}s")
    print(f"速度: {n_pixels / elapsed_fast:.0f} 像素/秒")
    
    # 性能对比
    print("\n" + "=" * 60)
    print("性能对比")
    print("=" * 60)
    speedup = elapsed_slow / elapsed_fast
    print(f"加速比: {speedup:.2f}x")
    print(f"时间节省: {elapsed_slow - elapsed_fast:.2f}s ({((elapsed_slow - elapsed_fast) / elapsed_slow * 100):.1f}%)")
    
    # 估算实际运行时间
    print("\n" + "=" * 60)
    print("实际运行时间估算")
    print("=" * 60)
    
    # 假设实际图像大小
    real_n_pixels = 1000 * 1000  # 1000x1000图像
    real_n_refs = 1000
    
    # 根据测试结果估算
    slow_time = (real_n_pixels / (n_pixels / elapsed_slow)) * (real_n_refs / n_refs)
    fast_time = (real_n_pixels / (n_pixels / elapsed_fast)) * (real_n_refs / n_refs)
    
    print(f"图像大小: {real_n_pixels} 像素")
    print(f"参考序列: {real_n_refs} 条")
    print(f"\n当前方式预计耗时: {slow_time:.1f}s ({slow_time / 60:.1f}分钟)")
    print(f"优化方式预计耗时: {fast_time:.1f}s ({fast_time / 60:.1f}分钟)")
    print(f"节省时间: {(slow_time - fast_time) / 60:.1f}分钟")
    
    return {
        "slow_time": elapsed_slow,
        "fast_time": elapsed_fast,
        "speedup": speedup,
        "real_slow": slow_time,
        "real_fast": fast_time,
    }


def test_preparse_performance():
    """测试预解析的性能提升"""
    
    print("\n" + "=" * 60)
    print("测试: 预解析优化")
    print("=" * 60)
    
    # 模拟CSV解析
    n_cells = 10000
    seq_len = 7
    
    # 生成模拟CSV数据
    print(f"\n生成 {n_cells} 个单元格的模拟数据...")
    csv_data = []
    for i in range(n_cells):
        seq = " ".join([str(np.random.randint(0, 100)) for _ in range(seq_len)])
        csv_data.append(seq)
    
    # 测试1: 重复解析（当前方式）
    print("\n测试1: 重复解析（当前方式）")
    start = time.time()
    for _ in range(10):  # 模拟10次重复解析
        for cell in csv_data:
            vals = cell.split()
            arr = np.array([int(v) for v in vals], dtype=np.int16)
    elapsed_repeat = time.time() - start
    print(f"耗时: {elapsed_repeat:.2f}s")
    
    # 测试2: 预解析一次（优化方式）
    print("\n测试2: 预解析一次（优化方式）")
    start = time.time()
    parsed = []
    for cell in csv_data:
        vals = cell.split()
        arr = np.array([int(v) for v in vals], dtype=np.int16)
        parsed.append(arr)
    elapsed_once = time.time() - start
    print(f"耗时: {elapsed_once:.2f}s")
    
    print(f"\n预解析节省时间: {elapsed_repeat - elapsed_once:.2f}s")
    print(f"性能提升: {elapsed_repeat / elapsed_once:.2f}x")


if __name__ == "__main__":
    print("开始性能测试...")
    
    # 测试1: 匹配性能
    results = test_matching_performance()
    
    # 测试2: 预解析性能
    test_preparse_performance()
    
    print("\n" + "=" * 60)
    print("优化建议")
    print("=" * 60)
    print("\n1. 使用批处理匹配（加速比: {:.1f}x）".format(results['speedup']))
    print("2. 预解析目标文件（避免重复解析）")
    print("3. 安装CuPy启用GPU加速")
    print("\n预计优化后，20分钟的任务可缩短至: {:.1f}分钟".format(results['real_fast'] / 60))
