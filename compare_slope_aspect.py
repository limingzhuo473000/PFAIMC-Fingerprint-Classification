# -*- coding: utf-8 -*-
"""对比：基线(无坡度坡向) vs 新实验(加坡度坡向) 的逐轮精度"""
import re, os, glob

BASE_OLD = r"E:/指纹/124036中/数据"      # 基线
BASE_NEW = r"E:/高程和坡向/数据"          # 加坡度坡向
CLASSES = ["人工表面", "水体", "河漫滩", "耕地"]


def parse_report(txt):
    """解析 accuracy_report.txt → OA, Kappa, {类: (PA, UA)}"""
    oa = re.search(r"Overall Accuracy:\s+([\d.]+)%", txt)
    ka = re.search(r"Kappa Coefficient:\s+([\d.]+)", txt)
    # Class  Producer(%)    User(%) 表格
    cls = {}
    block = txt.split("Class    Producer")[1] if "Class    Producer" in txt else ""
    for line in block.splitlines()[2:]:  # 跳过表头和分隔线
        m = re.match(r"(\S+)\s+([\d.]+)\s+([\d.]+)", line.strip())
        if m and m.group(1) not in ("未分类(0)",):
            cls[m.group(1)] = (m.group(2), m.group(3))
    return (oa.group(1) if oa else None,
            ka.group(1) if ka else None, cls)


print(f"{'轮次':>6} | {'新实验 OA%':>9} {'Kappa':>6} || {'基线 OA%':>8} {'Kappa':>6} || 各类 PA/UA (新|基线)")
print("-" * 110)

# 基线汇总 CSV
import csv
old_rows = {}
with open(os.path.join(BASE_OLD, "精度评价汇总.csv"), encoding="utf-8-sig") as f:
    for r in csv.reader(f):
        if r and r[1].isdigit() and len(r) >= 4:
            old_rows[int(r[1])] = {"oa": r[2], "kappa": r[3],
                                   "cls": dict(zip(CLASSES, [(r[4+2*i], r[5+2*i]) for i in range(4)]))}

# 新实验每轮报告
rounds = sorted([d for d in os.listdir(BASE_NEW) if d.startswith("round_")],
                key=lambda d: int(d.split("_")[1]))
new_data = {}
for rd in rounds:
    rep = glob.glob(os.path.join(BASE_NEW, rd, "step6", "*accuracy_report.txt"))
    if not rep:
        continue
    txt = open(rep[0], encoding="utf-8").read()
    oa, kappa, cls = parse_report(txt)
    if oa:
        new_data[int(rd.split("_")[1])] = {"oa": oa, "kappa": kappa, "cls": cls}

max_r = max(max(new_data), max(old_rows))
for i in range(1, max_r + 1):
    n = new_data.get(i)
    o = old_rows.get(i)
    def fmt(side, which):
        d = side.get(i)
        if not d:
            return "-"
        c = d["cls"].get(which, ("-", "-"))
        return f"{c[0]:>6}/{c[1]:<6}"
    n_oa = f"{n['oa']:>7}" if n else "   跑未到"
    n_k = f"{n['kappa']:>6}" if n else "     -"
    o_oa = f"{o['oa']:>7}" if o else "   跑未到"
    o_k = f"{o['kappa']:>6}" if o else "     -"
    detail = ""
    if n and o:
        detail = " || ".join(
            f"{c} {n['cls'].get(c, ('-','-'))[0]:>6}/{n['cls'].get(c, ('-','-'))[1]:<6}|{o['cls'].get(c, ('-','-'))[0]:>6}/{o['cls'].get(c, ('-','-'))[1]:<6}"
            for c in CLASSES)
    print(f"{i:>6} | {n_oa:>9} {n_k:>6} || {o_oa:>8} {o_k:>6} || {detail}")
