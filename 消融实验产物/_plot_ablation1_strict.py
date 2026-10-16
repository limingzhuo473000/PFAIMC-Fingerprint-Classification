# -*- coding: utf-8 -*-
"""
消融实验1「严格控制变量」结果出图 + 汇总表更新
对照：
  BASE = E:\指纹\124036全时相\离散等级5   （5级离散 + 严格汉明，OA 89.91%）
  STRICT = E:\指纹\消融实验\124036无离散化汉明 （无离散化 + 严格汉明，不做 z-score）
唯一变量 = 是否离散化。
"""
import os, re, glob, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = r"E:\指纹\124036全时相\离散等级5"
STRICT = r"E:\指纹\消融实验\124036无离散化汉明"
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "124036消融1_严格单变量_无离散化汉明.png")


def scan(base):
    """[(轮次, 阈值T, OA, Kappa, 未分类数)]"""
    out = []
    for d in glob.glob(os.path.join(base, "round_*")):
        m = re.search(r"round_(\d+)", os.path.basename(d))
        if not m:
            continue
        r = int(m.group(1))
        f3 = glob.glob(os.path.join(d, "step3", "*.csv"))
        T = int(re.search(r"out1_(\d+)", os.path.basename(f3[0])).group(1)) if f3 else None
        f6 = glob.glob(os.path.join(d, "step6", "*accuracy_report.txt"))
        oa = kap = None
        unc = None
        if f6:
            t = open(f6[0], encoding="utf-8", errors="ignore").read()
            m = re.search(r"Overall Accuracy[^0-9]*([0-9.]+)", t)
            oa = float(m.group(1)) if m else None
            m = re.search(r"Kappa Coefficient[^0-9]*([0-9.]+)", t)
            kap = float(m.group(1)) if m else None
            m = re.search(r"\nTotal\s+(\d+)", t)
            unc = int(m.group(1)) if m else None
        out.append((r, T, oa, kap, unc))
    out.sort(key=lambda x: x[0])
    return out


b = scan(BASE)
s = scan(STRICT)
if not s:
    raise SystemExit("未找到 STRICT 结果，实验可能仍在运行")

bT = [x[1] for x in b]; bOA = [x[2] for x in b]
sT = [x[1] for x in s]; sOA = [x[2] for x in s]
sUNC = [x[4] for x in s]

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
fig.suptitle("消融实验1（严格控制变量）：只取消离散化，其余与基线完全一致",
             fontsize=15, fontweight="bold", y=0.99)

# ① OA 逐轮对比
ax = axes[0]
ax.plot(bT, bOA, "o-", color="#2c3e50", lw=2.4, ms=7, label="基线：5级离散 + 汉明")
ax.plot(sT, sOA, "s--", color="#c0392b", lw=2.4, ms=7, label="消融1：无离散化 + 汉明")
ax.set_xlabel("匹配阈值 T"); ax.set_ylabel("总体精度 OA (%)")
ax.set_title("① 逐轮 OA（唯一变量=离散化）", fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=.3)
best = max(x for x in sOA if x is not None)
ax.annotate(f"消融1峰值 {best}%", (sT[sOA.index(best)], best),
            textcoords="offset points", xytext=(-10, -22), color="#c0392b", fontweight="bold")

# ② 未分类像素数
ax = axes[1]
ax.bar(sT, sUNC, width=3.2, color="#7f8c8d", alpha=.75)
ax.axhline(109, ls="--", color="#c0392b", lw=1.8, label="验证样本总数 109")
ax.set_xlabel("匹配阈值 T"); ax.set_ylabel("未分类样本数（共 109）")
ax.set_title("② 消融1：几乎全像元未分类", fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=.3, axis="y")

# ③ 组件贡献条形
ax = axes[2]
labels = ["基线\n5级+汉明", "消融1-严格\n无离散化+汉明"]
vals = [max(bOA), best]
colors = ["#2c3e50", "#c0392b"]
bars = ax.bar(labels, vals, color=colors, width=.55)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width()/2, v + 1.5, f"{v:.2f}%",
            ha="center", fontsize=13, fontweight="bold")
ax.set_ylabel("总体精度 OA (%)"); ax.set_ylim(0, 105)
ax.set_title("③ 单变量对照结论", fontsize=12, fontweight="bold")
ax.grid(alpha=.3, axis="y")
ax.annotate(f"Δ = {best - max(bOA):+.2f} pp", xy=(0.5, 55), ha="center",
            fontsize=13, color="#c0392b", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print("saved:", OUT_PNG)

# ── 输出 CSV ──
csv_path = os.path.join(os.path.dirname(OUT_PNG), "124036消融1_严格单变量_逐轮结果.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["轮次", "阈值T", "OA(%)", "Kappa", "未分类数"])
    for row in s:
        w.writerow(row)
print("saved:", csv_path)
print("\n=== 关键结论 ===")
print(f"基线(5级+汉明) 峰值 OA = {max(bOA)}%")
print(f"消融1-严格(无离散化+汉明) 峰值 OA = {best}%  未分类 {max(x for x in sUNC if x is not None)}/109")
print(f"Δ = {best - max(bOA):+.2f} pp")
