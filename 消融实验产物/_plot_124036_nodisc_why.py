# -*- coding: utf-8 -*-
"""
124036 去离散化（连续值+z-score+欧氏）OA=93.58% 的成因拆解图
数据源（纯读磁盘）：
  - E:\指纹\消融实验\124036去离散化            （本次 run，T=10..30，5 轮）
  - E:\指纹\124036全时相\离散等级5              （基线，5级离散+严格汉明，89.91%）
  - 消融实验汇总.md                             （消融2：5级+欧氏 = 92.66%）
"""
import re, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

NEW = r"E:\指纹\消融实验\124036去离散化"
BASE = r"E:\指纹\124036全时相\离散等级5"

def scan(base):
    """返回 [(轮次, 阈值T, OA, 未分类数)]"""
    out = []
    for d in sorted(glob.glob(os.path.join(base, "round_*")),
                    key=lambda p: int(re.search(r"round_(\d+)", p).group(1))):
        r = int(re.search(r"round_(\d+)", os.path.basename(d)).group(1))
        f3 = glob.glob(os.path.join(d, "step3", "*.csv"))
        T = int(re.search(r"out1_(\d+)", os.path.basename(f3[0])).group(1)) if f3 else None
        f6 = glob.glob(os.path.join(d, "step6", "*accuracy_report.txt"))
        oa = unc = None
        if f6:
            t = open(f6[0], encoding="utf-8", errors="ignore").read()
            m = re.search(r"Overall Accuracy[^0-9]*([0-9.]+)", t)
            oa = float(m.group(1)) if m else None
            m = re.search(r"\nTotal\s+(\d+)", t)          # 列合计行：第1个是“未分类”预测总数
            unc = int(m.group(1)) if m else None
        out.append((r, T, oa, unc))
    return out

new_rows = scan(NEW)
Ts = [x[1] for x in new_rows]
OAs = [x[2] for x in new_rows]
UNCs = [x[3] for x in new_rows]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("124036 去离散化 OA=93.58% —— 精度从哪来？",
             fontsize=17, fontweight="bold", y=0.975)

# ── ① 阈值扫描曲线 ────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(Ts, OAs, "o-", color="#c0392b", lw=2.4, ms=9, label="OA (%)")
ax.axhline(89.91, ls="--", color="#555", lw=1.8, label="基线 89.91%（5级+汉明）")
ax.set_xlabel("欧氏距离阈值 T", fontsize=11)
ax.set_ylabel("总体精度 OA (%)", fontsize=11)
ax.set_title("① 阈值曲线：T=15 即到顶，只跑 5 轮", fontsize=13, fontweight="bold")
ax2 = ax.twinx()
ax2.bar(Ts, UNCs, width=2.2, color="#7f8c8d", alpha=0.35, label="未分类样本数")
ax2.set_ylabel("未分类样本数（共109）", fontsize=11)
for t, o, u in zip(Ts, OAs, UNCs):
    ax.annotate(f"{o}%", (t, o), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=10, color="#c0392b", fontweight="bold")
    ax2.annotate(f"未分{u}", (t, u), textcoords="offset points", xytext=(0, 4),
                 ha="center", fontsize=9, color="#555")
ax.set_ylim(84, 100)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)

# ── ② 类别级正确数 ────────────────────────────────────────────
ax = axes[0, 1]
cls = ["人工表面", "水体", "河漫滩", "耕地"]
ref = [52, 17, 7, 33]
base_ok = [45, 17, 6, 30]
new_ok = [49, 17, 6, 30]
x = np.arange(len(cls)); w = 0.36
ax.bar(x - w/2, base_ok, w, color="#95a5a6", label="基线（5级+汉明）98/109")
ax.bar(x + w/2, new_ok, w, color="#c0392b", label="去离散化（连续+欧氏）102/109")
for i in range(len(cls)):
    ax.text(i - w/2, base_ok[i] + 0.8, str(base_ok[i]), ha="center", fontsize=10)
    ax.text(i + w/2, new_ok[i] + 0.8, str(new_ok[i]), ha="center",
            fontsize=10, fontweight="bold", color="#c0392b")
    ax.text(i, ref[i] + 3.2, f"参考{ref[i]}", ha="center", fontsize=9, color="#666")
ax.set_xticks(x); ax.set_xticklabels(cls, fontsize=11)
ax.set_ylabel("正确分类样本数", fontsize=11)
ax.set_title("② 增益只有 4 个像素，全在人工表面", fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 60)

# ── ③ step2 类间区分度评分（硬证据）────────────────────────────
ax = axes[1, 0]
score_cls = ["河漫滩", "人工表面", "水体", "耕地"]
base_score = [117.67, 117.67, 135.67, 175.00]
new_score = [200.0, 200.0, 200.0, 200.0]
y = np.arange(len(score_cls)); h = 0.36
ax.barh(y + h/2, base_score, h, color="#95a5a6", label="基线：5级量化后（满分200）")
ax.barh(y - h/2, new_score, h, color="#27ae60", label="去离散化：连续值")
for i in range(len(score_cls)):
    ax.text(base_score[i] + 3, i + h/2, f"{base_score[i]:.1f}", va="center", fontsize=10)
    ax.text(new_score[i] - 12, i - h/2, "200.0", va="center", fontsize=10,
            color="white", fontweight="bold")
ax.axvline(200, ls="--", color="#27ae60", lw=1.2)
ax.set_yticks(y); ax.set_yticklabels(score_cls, fontsize=11)
ax.set_xlabel("step2 类间区分度评分", fontsize=11)
ax.set_xlim(0, 225)
ax.set_title("③ 量化证据：离散后类别编码发生碰撞", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="x", alpha=0.3)

# ── ④ 增益归因 ────────────────────────────────────────────────
ax = axes[1, 1]
labels = ["基线\n5级+汉明", "换欧氏\n(5级+欧氏)", "再去离散化\n(连续+欧氏)", "（参考）\n连续+ε=0.5汉明"]
vals = [88.07, 92.66, 93.58, 95.41]
colors = ["#95a5a6", "#2980b9", "#c0392b", "#8e44ad"]
bars = ax.bar(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v}%", ha="center",
            fontsize=12, fontweight="bold")
ax.text(1, 90.4, "换度量\n+4.59pp", ha="center", va="center",
        fontsize=11, color="white", fontweight="bold")
ax.text(2, 90.4, "再去离散化\n+0.92pp", ha="center", va="center",
        fontsize=11, color="white", fontweight="bold")
ax.set_ylabel("总体精度 OA (%)", fontsize=11)
ax.set_ylim(80, 100)
ax.set_title("④ 归因：主要是欧氏距离的功劳", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.text(0.5, -0.16, "注：前三根来自 09-22 消融批（25 轮、阈值10→130）；紫柱为该批消融1，功能已移除",
        transform=ax.transAxes, ha="center", fontsize=9, color="#777")

plt.tight_layout(rect=[0, 0, 1, 0.955])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "124036去离散化_精度成因拆解.png")
plt.savefig(out, dpi=130)
print("saved:", out)
print("轮次数据:", new_rows)
