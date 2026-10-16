"""
诊断全指数数据 Step1 处理流程
追踪从 TIF → CSV → 零值替换 → 分位数分类 → 合并 的每一步文件数
"""
import os, sys, glob, csv, shutil, tempfile, json
import numpy as np
import pandas as pd

# 使用项目 venv 的 rasterio
import rasterio

IMG_DIR = r'E:\指纹\129034英文测试\全指数图像'
NUM_CLASSES = 5

# 创建临时工作目录
work_dir = os.path.join(tempfile.gettempdir(), "diag_step1_full")
if os.path.exists(work_dir):
    shutil.rmtree(work_dir)
os.makedirs(work_dir)

step2_dir = os.path.join(work_dir, "step02_csv")
step3_dir = os.path.join(work_dir, "step03_csv")
step4_dir = os.path.join(work_dir, "step04_csv")
os.makedirs(step2_dir)
os.makedirs(step3_dir)
os.makedirs(step4_dir)

# ---- Step 2: 影像转CSV ----
print("=" * 60)
print("Step 2: 影像转CSV")
image_files = sorted(set(
    glob.glob(os.path.join(IMG_DIR, '*.tif')) +
    glob.glob(os.path.join(IMG_DIR, '*.TIF')) +
    glob.glob(os.path.join(IMG_DIR, '*.tiff')) +
    glob.glob(os.path.join(IMG_DIR, '*.TIFF'))
))
print(f"  输入TIF文件数: {len(image_files)}")

ok_count = 0
fail_list = []
csv_shapes = {}
for fp in image_files:
    fname = os.path.splitext(os.path.basename(fp))[0]
    out_path = os.path.join(step2_dir, f"{fname}.csv")
    try:
        with rasterio.open(fp) as src:
            data = src.read(1)
        pd.DataFrame(data).to_csv(out_path, index=False, header=False)
        # 验证写入的CSV
        with open(out_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        shape = (len(rows), len(rows[0]) if rows else 0)
        csv_shapes[shape] = csv_shapes.get(shape, 0) + 1
        ok_count += 1
    except Exception as e:
        fail_list.append((os.path.basename(fp), str(e)))

print(f"  成功转CSV: {ok_count}")
print(f"  失败: {len(fail_list)}")
if fail_list:
    for fn, err in fail_list[:10]:
        print(f"    {fn}: {err}")
print(f"  CSV维度分布: {csv_shapes}")

# ---- Step 3: 零值替换 ----
print("=" * 60)
print("Step 3: 零值替换")
csv_files_03 = sorted(glob.glob(os.path.join(step2_dir, "*.csv")))
print(f"  输入CSV文件数: {len(csv_files_03)}")
ok_count_03 = 0
for fp in csv_files_03:
    try:
        df = pd.read_csv(fp, header=None)
        df = df.replace(0, "")
        out_path = os.path.join(step3_dir, os.path.basename(fp))
        df.to_csv(out_path, index=False, header=False)
        ok_count_03 += 1
    except Exception as e:
        print(f"  零值替换失败 {os.path.basename(fp)}: {e}")
print(f"  成功零值替换: {ok_count_03}")

# ---- Step 4: 分位数分类 ----
print("=" * 60)
print("Step 4: 分位数分类")
csv_files_04 = sorted(glob.glob(os.path.join(step3_dir, "*.csv")))
print(f"  输入CSV文件数: {len(csv_files_04)}")

ok_count_04 = 0
fail_04 = []
for fp in csv_files_04:
    out_path = os.path.join(step4_dir, os.path.basename(fp))
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if len(rows) < 2 or len(rows[0]) < 2:
            fail_04.append((os.path.basename(fp), "rows<2 or cols<2"))
            continue
        
        data_values = []
        for row in rows:
            for cell in row:
                try:
                    val = float(cell)
                    if not np.isnan(val) and val > 0:
                        data_values.append(val)
                except (ValueError, TypeError):
                    pass
        
        if not data_values:
            fail_04.append((os.path.basename(fp), "no positive data_values"))
            continue
        
        if NUM_CLASSES > 1:
            quantiles = [(i + 1) / NUM_CLASSES for i in range(NUM_CLASSES - 1)]
            thresholds = sorted(np.quantile(data_values, quantiles).tolist())
        else:
            thresholds = []
        
        def classify_cell(cell):
            try:
                x = float(cell)
                if NUM_CLASSES == 1:
                    return '1'
                for i, threshold in enumerate(thresholds):
                    if x <= threshold:
                        return str(i + 1)
                return str(NUM_CLASSES)
            except Exception:
                return str(cell)
        
        new_rows = [[classify_cell(cell) for cell in row] for row in rows]
        
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)
        ok_count_04 += 1
    except Exception as e:
        fail_04.append((os.path.basename(fp), str(e)))

print(f"  成功分位数分类: {ok_count_04}")
print(f"  失败: {len(fail_04)}")
if fail_04:
    for fn, err in fail_04[:10]:
        print(f"    {fn}: {err}")

# ---- Step 5: 合并CSV ----
print("=" * 60)
print("Step 5: 合并CSV")
csv_files_05 = sorted(glob.glob(os.path.join(step4_dir, "*.csv")))
print(f"  输入CSV文件数(待合并): {len(csv_files_05)}")

# 检查每个CSV的维度
merge_shapes = {}
for fp in csv_files_05:
    with open(fp, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    shape = (len(rows), max(len(r) for r in rows) if rows else 0)
    merge_shapes[shape] = merge_shapes.get(shape, 0) + 1
print(f"  待合并CSV维度分布: {merge_shapes}")

# 执行合并
merged_data = None
target_rows = 0
target_cols = 0
valid_files = 0
failed_files = 0

for fp in csv_files_05:
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if not rows or not rows[0]:
            failed_files += 1
            continue
        
        current_rows = len(rows)
        current_cols = max(len(row) for row in rows) if rows else 0
        
        if target_rows == 0:
            target_rows = current_rows
            target_cols = current_cols
            merged_data = [[[] for _ in range(target_cols)] for _ in range(target_rows)]
        
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
    except Exception as e:
        failed_files += 1
        print(f"  合并失败 {os.path.basename(fp)}: {e}")

print(f"  有效合并文件数: {valid_files}")
print(f"  失败文件数: {failed_files}")

if merged_data:
    # 检查第一个非空cell的序列长度
    seq_len = 0
    for i in range(min(5, len(merged_data))):
        for j in range(min(5, len(merged_data[i]))):
            cell = merged_data[i][j]
            if isinstance(cell, list) and len(cell) > 0:
                seq_len = len(cell)
                print(f"\n  样本 cell[{i}][{j}] 序列长度: {len(cell)}")
                print(f"  样本值(前20): {' '.join(cell[:20])}")
                break
        if seq_len > 0:
            break
    
    # 统计所有cell的序列长度分布
    seq_lens = []
    for i in range(len(merged_data)):
        for j in range(len(merged_data[i])):
            cell = merged_data[i][j]
            if isinstance(cell, list) and len(cell) > 0:
                seq_lens.append(len(cell))
    
    if seq_lens:
        print(f"\n  序列长度统计:")
        print(f"    min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.1f}")
        from collections import Counter
        len_dist = Counter(seq_lens)
        for length, count in sorted(len_dist.items()):
            print(f"    长度={length}: {count} 个cell")

print("\n" + "=" * 60)
print(f"结论: merged.csv 每个cell的序列长度 = {seq_len if seq_lens else 'N/A'}")
print(f"期望值: ~{ok_count_04} (通过分位数分类的文件数)")
