# -*- coding: utf-8 -*-
"""
消融实验1「严格控制变量」版启动脚本
=====================================
唯一变量 = 是否做 5 级分位数离散化。
与基线（E:\\指纹\\124036全时相\\离散等级5）逐项对齐：
    输入影像   同目录 E:\\指纹\\124036中\\图像（200 个 tif）
    训练样本   同 E:\\指纹\\124036中\\样本\\实验样本
    验证样本   同 E:\\指纹\\124036中\\样本\\评价样本
    类别与编码 同 人工表面(3)/水体(4)/河漫滩(2)/耕地(5)
    距离度量   同 汉明 hamming（严格逐位相等，不加欧氏）
    迭代       同 fixed_rounds=True，max_iterations=25
    零值策略   同 exclude（只取 >0 参与统计）

差异只有一处：skip_discretization=True（不做 5 级离散化）。
且 normalize_continuous=False —— 不引入 z-score 这第二个变量。

用法： python _run_ablation1_strict.py
"""
import json
import urllib.request

URL = "http://127.0.0.1:8001/api/pipeline/run"

payload = {
    # ── 步骤1 ──
    "input_img_dir": r"E:\指纹\124036中\图像",
    "output_dir": r"E:\指纹\消融实验\124036无离散化汉明",
    "filename": "merged",
    "num_classes": 5,
    "skip_discretization": True,      # ★ 唯一变量：取消离散化
    "normalize_continuous": False,    # ★ 严格单变量：不做 z-score
    "zero_policy": "exclude",
    "adjust_sizes": True,
    # ── 步骤2 ──
    "shp_input": r"E:\指纹\124036中\样本\实验样本",
    # ── 步骤3 ──
    "distance_metric": "hamming",     # ★ 与基线一致，不加欧氏
    "execution_mode": "multiprocess",
    "num_processes": 0,
    "threshold_mode": "manual",
    "auto_threshold_samples": 2000,
    # ── 步骤4 ──
    "add_geo": True,
    # ── 步骤5：类别编码与基线完全一致 ──
    "classes": [
        {"name": "人工表面", "values": [3]},
        {"name": "水体", "values": [4]},
        {"name": "河漫滩", "values": [2]},
        {"name": "耕地", "values": [5]},
    ],
    # ── 步骤6 ──
    "shp_folder": r"E:\指纹\124036中\样本\评价样本",
    # ── 迭代控制 ──
    "fixed_rounds": True,
    "max_iterations": 25,
    "kappa_threshold": 0.8,
    "unclassified_threshold": 10.0,
    "consecutive_rounds": 3,
}

if __name__ == "__main__":
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = resp.read().decode("utf-8", "ignore")
    print("HTTP", resp.status)
    print(body[:3000])
