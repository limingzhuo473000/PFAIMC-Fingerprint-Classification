"""
步骤6: 评价测试 - 核心处理逻辑

从 06评价测试.py 提取封装
"""

import os
import re
import sys as _sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory

import numpy as np
import pandas as pd

# 延迟导入重型依赖（避免服务器启动时因缺少可选包而崩溃）
def _lazy_import_geopandas():
    import geopandas as _gpd
    return _gpd

def _lazy_import_rasterio():
    import rasterio as _rio
    return _rio

def _lazy_import_sklearn_metrics():
    from sklearn.metrics import confusion_matrix as _cm, cohen_kappa_score as _ck, accuracy_score as _acc
    return _cm, _ck, _acc

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

def read_class_names_from_hdr(hdr_path: str) -> list[str]:
    """从 HDR 文件中解析 class names"""
    if not os.path.exists(hdr_path):
        raise FileNotFoundError(f"HDR文件不存在: {hdr_path}")

    content = None
    for encoding in ['gbk', 'utf-8']:
        try:
            with open(hdr_path, 'r', encoding=encoding) as f:
                content = f.read()
            if content.strip():
                break
        except UnicodeDecodeError:
            continue

    if not content:
        raise ValueError(f"无法读取HDR文件: {hdr_path}")

    match = re.search(r'class names\s*=\s*\{(.+?)\}', content, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"HDR文件中未找到class names字段")

    names_str = match.group(1).strip()
    return [name.strip() for name in names_str.split(',')]


def read_envi_classification(dat_path: str):
    """读取 ENVI 分类结果"""
    if not os.path.exists(dat_path):
        if not dat_path.endswith('.dat') and os.path.exists(dat_path + '.dat'):
            dat_path += '.dat'
        else:
            raise FileNotFoundError(f"分类文件不存在: {dat_path}")

    with _lazy_import_rasterio().open(dat_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

    return data, transform, crs


def load_shp_folder(shp_folder: str, class_name_to_code: dict):
    """读取 SHP 文件夹，按名称匹配类别

    匹配策略（按优先级）：
    1. 精确匹配：SHP 文件名（不含扩展名）精确等于 HDR 类名
    2. 包含匹配：SHP 文件名包含于 HDR 类名，或 HDR 类名包含于 SHP 文件名
    3. 前缀匹配：SHP 文件名与 HDR 类名共享最长前缀
    4. 顺序分配：以上都失败时，按文件排序顺序直接分配代码值
    """
    if not os.path.isdir(shp_folder):
        raise NotADirectoryError(f"SHP文件夹不存在: {shp_folder}")

    shp_files = sorted([f for f in os.listdir(shp_folder) if f.lower().endswith('.shp')])
    if not shp_files:
        raise FileNotFoundError(f"SHP文件夹中没有.shp文件: {shp_folder}")

    class_info = []
    all_gdfs = []
    matched_names = set()
    unmatched_files = []

    # 构建可用代码值列表（从 class_name_to_code 的 values 中收集）
    available_codes = sorted(set(class_name_to_code.values()))
    code_index = 0

    def _find_best_match(shp_name: str) -> tuple[str, int] | None:
        """为 SHP 文件名寻找最佳匹配的 HDR 类名和代码"""
        # 1. 精确匹配
        if shp_name in class_name_to_code:
            return shp_name, class_name_to_code[shp_name]

        # 2. 包含匹配（双向）
        for hdr_name, code in class_name_to_code.items():
            if shp_name in hdr_name or hdr_name in shp_name:
                return hdr_name, code

        # 3. 前缀匹配（找最长公共前缀）
        best_match = None
        best_prefix_len = 0
        for hdr_name, code in class_name_to_code.items():
            # 计算最长公共前缀长度
            min_len = min(len(shp_name), len(hdr_name))
            prefix_len = 0
            for i in range(min_len):
                if shp_name[i] == hdr_name[i]:
                    prefix_len += 1
                else:
                    break
            if prefix_len > best_prefix_len:
                best_prefix_len = prefix_len
                best_match = (hdr_name, code)

        if best_match and best_prefix_len >= 2:
            return best_match

        return None

    for shp_file in shp_files:
        class_name = os.path.splitext(shp_file)[0]
        shp_path = os.path.join(shp_folder, shp_file)

        # 尝试匹配
        match_result = _find_best_match(class_name)

        if match_result:
            matched_name, class_code = match_result
            matched_names.add(matched_name)
            print(f"[步骤6] SHP匹配成功: {class_name}.shp → HDR类名 '{matched_name}' (代码 {class_code})")
        else:
            # 模糊匹配失败，使用顺序分配
            if code_index < len(available_codes):
                class_code = available_codes[code_index]
                code_index += 1
                print(f"[步骤6] SHP模糊匹配失败，顺序分配: {class_name}.shp → 代码 {class_code}")
            else:
                unmatched_files.append(class_name)
                print(f"[步骤6] SHP跳过(无可用代码): {class_name}.shp")
                continue

        try:
            gdf = _lazy_import_geopandas().read_file(shp_path)
        except Exception as e:
            print(f"[步骤6] 读取SHP失败 {shp_file}: {e}")
            continue

        if len(gdf) == 0:
            print(f"[步骤6] SHP为空(0个要素): {shp_file}")
            continue

        # 记录原始要素数
        original_count = len(gdf)

        gdf['true_code'] = class_code
        gdf['class_name'] = class_name
        gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]

        if len(gdf) == 0:
            print(f"[步骤6] SHP过滤后为空(原始{original_count}个): {shp_file}")
            continue

        all_gdfs.append(gdf)
        class_info.append((class_code, class_name, len(gdf)))

    if not all_gdfs:
        # 构建详细的诊断信息
        hdr_names = list(class_name_to_code.keys())
        shp_names = [os.path.splitext(f)[0] for f in shp_files]
        diagnostic = (
            f"所有SHP文件均无有效多边形。\n"
            f"诊断信息:\n"
            f"  - SHP文件夹: {shp_folder}\n"
            f"  - SHP文件: {shp_names}\n"
            f"  - HDR类名: {hdr_names}\n"
            f"  - 匹配成功: {list(matched_names)}\n"
            f"  - 未匹配/跳过: {unmatched_files}\n"
            f"可能原因:\n"
            f"  1. SHP文件名与HDR中的class names不一致（如 '耕地少.shp' vs '耕地'）\n"
            f"  2. SHP文件几何图形全部无效或为空\n"
            f"  3. SHP坐标系与影像不匹配，导致所有要素被过滤\n"
            f"建议: 将SHP文件名改为与HDR中的class names一致，或检查SHP数据完整性。"
        )
        raise ValueError(diagnostic)

    combined_gdf = pd.concat(all_gdfs, ignore_index=True)
    return combined_gdf, class_info


def _extract_pixels_for_class(args):
    """提取单个类别的像元（顶层，可被多进程序列化）
    使用共享内存访问大影像数组，避免序列化拷贝。
    """
    rows_dict_list, shm_name, img_shape, transform_data, h, w, skip_nodata, is_point = args

    import numpy as np
    from multiprocessing import shared_memory as _shm
    from rasterio.transform import rowcol as _rowcol, Affine as _Affine
    from rasterio import features as _features

    transform = _Affine(*transform_data)

    # 从共享内存中以只读方式访问影像数组，无需拷贝
    existing_shm = _shm.SharedMemory(name=shm_name)
    class_img_int = np.ndarray(img_shape, dtype=np.int32, buffer=existing_shm.buf)

    real_codes = []
    pred_codes = []

    try:
        for row_dict in rows_dict_list:
            true_code = row_dict['true_code']
            geom = row_dict['geometry']

            if is_point:
                r, c = _rowcol(transform, geom.x, geom.y, op=np.round)
                r, c = int(r), int(c)
                if 0 <= r < h and 0 <= c < w:
                    val = int(class_img_int[r, c])
                    if val == 1:
                        continue
                    if skip_nodata and val == 0:
                        continue
                    real_codes.append(true_code)
                    pred_codes.append(val)
            else:
                try:
                    mask = _features.geometry_mask([geom], out_shape=(h, w), transform=transform, invert=True)
                except Exception:
                    continue
                pixels = class_img_int[mask]
                if skip_nodata:
                    valid_mask = (pixels != 0) & (pixels != 1)
                else:
                    valid_mask = pixels != 1
                valid_pixels = pixels[valid_mask]
                if len(valid_pixels) == 0:
                    continue
                real_codes.extend([true_code] * len(valid_pixels))
                pred_codes.extend(valid_pixels.tolist())
    finally:
        existing_shm.close()

    return real_codes, pred_codes


def extract_pixels(gdf, class_img, transform, skip_nodata: bool = False):
    """提取像元（多进程按类别并行，共享内存传递大数组）"""
    h, w = class_img.shape
    class_img_int = class_img.astype(np.int32)

    first_geom = gdf.iloc[0].geometry
    is_point = 'Point' in first_geom.geom_type

    transform_data = (transform.a, transform.b, transform.c,
                      transform.d, transform.e, transform.f)

    # 创建共享内存：所有子进程共享同一份影像数组，无序列化开销
    shm = shared_memory.SharedMemory(create=True, size=class_img_int.nbytes)
    shared_arr = np.ndarray(class_img_int.shape, dtype=np.int32, buffer=shm.buf)
    shared_arr[:] = class_img_int[:]

    # 按 true_code 分组
    groups = {}
    for idx, row in gdf.iterrows():
        tc = int(row['true_code'])
        if tc not in groups:
            groups[tc] = []
        groups[tc].append({'true_code': tc, 'geometry': row.geometry})

    args_list = [
        (rows, shm.name, class_img_int.shape, transform_data, h, w, skip_nodata, is_point)
        for rows in groups.values()
    ]

    all_real = []
    all_pred = []

    try:
        with ProcessPoolExecutor(max_workers=min(_MAX_WORKERS, len(args_list) or 1), initializer=_init_pool) as ex:
            futures = [ex.submit(_extract_pixels_for_class, args) for args in args_list]
            for fut in as_completed(futures):
                try:
                    real, pred = fut.result()
                    all_real.extend(real)
                    all_pred.extend(pred)
                except Exception as e:
                    print(f"像元提取任务异常: {e}")
    finally:
        # 确保共享内存被释放
        shm.close()
        shm.unlink()

    return all_real, all_pred


def compute_metrics(real_codes: list, pred_codes: list, class_info: list):
    """计算精度指标"""
    confusion_matrix, cohen_kappa_score, accuracy_score = _lazy_import_sklearn_metrics()

    unique_codes = [info[0] for info in class_info]
    class_names_list = [info[1] for info in class_info]

    code_to_idx = {code: i for i, code in enumerate(unique_codes)}
    n_classes = len(unique_codes)

    real_idx = [code_to_idx[c] for c in real_codes]
    pred_idx = []
    for p in pred_codes:
        if p in code_to_idx:
            pred_idx.append(code_to_idx[p])
        else:
            nearest = min(unique_codes, key=lambda x: abs(x - p))
            pred_idx.append(code_to_idx[nearest])

    labels = list(range(n_classes))
    cm = confusion_matrix(real_idx, pred_idx, labels=labels)
    oa = accuracy_score(real_idx, pred_idx) * 100
    kappa = cohen_kappa_score(real_idx, pred_idx)

    producer = []
    user = []
    for i in range(n_classes):
        total_actual = cm[i, :].sum()
        total_pred = cm[:, i].sum()
        producer.append(cm[i, i] / total_actual * 100 if total_actual > 0 else 0)
        user.append(cm[i, i] / total_pred * 100 if total_pred > 0 else 0)

    return cm, oa, kappa, producer, user, class_names_list


def export_report(
    cm,
    oa,
    kappa,
    producer,
    user,
    class_names_list,
    class_info,
    output_txt,
    output_csv,
):
    """导出精度评价报告"""
    n = len(class_names_list)
    display_names = [f"{name}({code})" for code, name, _ in class_info]

    cm_display = cm
    ref_totals = cm_display.sum(axis=1)
    pred_totals = cm_display.sum(axis=0)
    total_pixels = cm.sum()

    # 百分比混淆矩阵
    cm_pct = np.zeros_like(cm_display, dtype=float)
    for i in range(n):
        if ref_totals[i] > 0:
            cm_pct[i, :] = cm_display[i, :] / ref_totals[i] * 100

    # Commission / Omission
    commission_pct = []
    omission_pct = []
    for i in range(n):
        comm_pix = pred_totals[i] - cm_display[i, i]
        commission_pct.append(comm_pix / pred_totals[i] * 100 if pred_totals[i] > 0 else 0)
        om_pix = ref_totals[i] - cm_display[i, i]
        omission_pct.append(om_pix / ref_totals[i] * 100 if ref_totals[i] > 0 else 0)

    # 写入TXT报告
    with open(output_txt, 'w', encoding='utf-8') as f:
        sep = "=" * 80
        f.write(f"{sep}\n")
        f.write("ENVI 分类精度评价报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{sep}\n\n")

        # 像素数混淆矩阵
        f.write("Confusion Matrix (Pixels)\n")
        f.write("                " + "".join(f"{dn:<14}" for dn in display_names) + f"{'Total':<8}\n")
        for i in range(n):
            f.write(f"{display_names[i]:<16}")
            for j in range(n):
                f.write(f"{int(cm_display[i, j]):<14}")
            f.write(f"{int(ref_totals[i]):<8}\n")
        f.write(f"{'Total':<16}")
        for j in range(n):
            f.write(f"{int(pred_totals[j]):<14}")
        f.write(f"{int(total_pixels):<8}\n")

        # 百分比混淆矩阵
        f.write("\nConfusion Matrix (Percent)\n")
        f.write("                " + "".join(f"{dn:<14}" for dn in display_names) + f"{'Total':<8}\n")
        for i in range(n):
            f.write(f"{display_names[i]:<16}")
            for j in range(n):
                f.write(f"{cm_pct[i, j]:<14.2f}")
            f.write(f"{100.0:<8.2f}\n")

        # 精度
        f.write("\nClass    Producer(%)    User(%)        Commission(%)    Omission(%)\n")
        f.write("-" * 80 + "\n")
        for i in range(n):
            f.write(f"{display_names[i]:<10} "
                    f"{producer[i]:<16.2f} "
                    f"{user[i]:<14.2f} "
                    f"{commission_pct[i]:<16.2f} "
                    f"{omission_pct[i]:<14.2f}\n")

        # 汇总
        f.write(f"\n{'Overall Accuracy:':<20} {oa:.2f}%\n")
        f.write(f"{'Kappa Coefficient:':<20} {kappa:.4f}\n")

    # CSV混淆矩阵
    cm_df = pd.DataFrame(cm, index=display_names, columns=display_names)
    cm_df.to_csv(output_csv, encoding='utf-8-sig')

    return cm_pct.tolist()


# ============================================================
# 主函数
# ============================================================

def run_step6(
    task_manager,
    task_id: str,
    reclass_dat: str,
    shp_folder: str,
    output_dir: str,
) -> dict:
    """
    执行步骤6处理
    """
    def update_progress(progress: float, message: str):
        task_manager.update_task(task_id, progress=progress, message=message)

    try:
        os.makedirs(output_dir, exist_ok=True)

        # 1. 读取分类影像
        update_progress(10, "读取分类影像...")
        class_img, transform, crs = read_envi_classification(reclass_dat)

        # 2. 读取类别名称
        update_progress(20, "读取类别名称...")
        base = os.path.splitext(reclass_dat)[0]
        hdr_path = base + '.hdr'
        all_class_names = read_class_names_from_hdr(hdr_path)

        class_name_to_code = {
            name: idx
            for idx, name in enumerate(all_class_names)
            if idx >= 2
        }

        # 3. 读取SHP文件夹
        update_progress(35, "读取验证样本...")
        gdf, class_info = load_shp_folder(shp_folder, class_name_to_code)

        # 注入未分类
        class_info.insert(0, (0, '未分类', 0))

        # 4. 坐标系转换
        if crs and gdf.crs and gdf.crs != crs:
            update_progress(45, "坐标系转换...")
            gdf = gdf.to_crs(crs)

        # 5. 提取像元
        update_progress(55, "提取验证像元...")
        real_codes, pred_codes = extract_pixels(gdf, class_img, transform, skip_nodata=False)

        if len(real_codes) == 0:
            return {"success": False, "error": "无有效像素"}

        # 6. 计算精度
        update_progress(70, "计算精度指标...")
        cm, oa, kappa, producer, user, class_names_list = compute_metrics(
            real_codes, pred_codes, class_info
        )

        # 7. 导出报告
        update_progress(85, "导出报告...")
        base_name = os.path.splitext(os.path.basename(reclass_dat))[0]
        txt_out = os.path.join(output_dir, f"{base_name}_accuracy_report.txt")
        csv_out = os.path.join(output_dir, f"{base_name}_confusion_matrix.csv")

        cm_pct = export_report(
            cm, oa, kappa, producer, user,
            class_names_list, class_info,
            txt_out, csv_out
        )

        update_progress(100, "评价完成!")

        return {
            "success": True,
            "metrics": {
                "overall_accuracy": round(oa, 2),
                "kappa": round(kappa, 4),
                "total_pixels": len(real_codes),
            },
            "class_accuracies": [
                {
                    "class_name": class_names_list[i],
                    "class_code": class_info[i][0],
                    "producer_accuracy": round(producer[i], 2),
                    "user_accuracy": round(user[i], 2),
                }
                for i in range(len(class_names_list))
            ],
            "confusion_matrix": cm.tolist(),
            "class_names": class_names_list,
            "report_file": txt_out,
            "csv_file": csv_out,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
