"""路径规范化工具：修复 Windows 路径中的双斜杠/混合斜杠问题。

★ 背景（2026-09-18）：形如 "E://指纹//样本" 的路径会被 pandas 误判为 fsspec
网络协议 URL（"E:" 被当成协议名），read_csv 时报
"ImportError: `Import fsspec` failed"。
本模块把所有路径字段统一收敛为单斜杠（保留 UNC 前缀 //），后端与 Windows
API 均可正确处理正斜杠路径。
"""


def normalize_path(p):
    """规范化单个路径字符串：反斜杠→正斜杠，连续斜杠收敛为一个（UNC 前缀除外）。"""
    if not isinstance(p, str) or not p:
        return p
    p = p.replace("\\", "/")
    prefix = ""
    if p.startswith("//"):
        prefix = "//"  # 保留 UNC 路径前缀
        p = p[2:]
    while "//" in p:
        p = p.replace("//", "/")
    return prefix + p


# 请求模型中被视为"路径"的字段名（其余字段如类别名、模式串不做处理）
_PATH_KEYS = {
    "input_img_dir", "output_dir", "existing_step1_csv", "shp_input",
    "tif_folder", "target_file", "shp_folder", "experiment_data_path",
    "csv_path", "img_dir", "sample_dir", "envi_dir",
}


def normalize_request_paths(data):
    """递归遍历请求数据（dict/list），规范化所有路径字段的斜杠。"""
    if isinstance(data, dict):
        return {
            k: normalize_path(v) if (k in _PATH_KEYS and isinstance(v, str)) else normalize_request_paths(v)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [normalize_request_paths(x) for x in data]
    return data
