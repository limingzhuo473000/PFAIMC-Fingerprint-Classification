# -*- coding: utf-8 -*-
"""对比 129034 九指数两次运行：9指数离散5(2026-09-23, seq=320) vs 9指数数据/离散5(2026-08-24, seq=318)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 实测数据 ----
new_rounds = list(range(1, 30))
new_thr = [16 + 8 * (i - 1) for i in new_rounds]
new_oa = [6.48, 7.41, 8.33, 8.33, 11.11, 14.81, 16.67, 20.37, 25.0, 27.78, 31.48, 35.19, 35.19,
          37.04, 37.04, 38.89, 46.3, 50.0, 53.7, 62.96, 67.59, 70.37, 73.15, 75.0, 75.93,
          78.7, 78.7, 78.7, 78.7]
old_rounds = list(range(1, 8))
old_thr = [96 + 25 * (i - 1) for i in old_rounds]
old_oa = [48.15, 62.04, 83.33, 91.67, 91.67, 91.67, 91.67]

# 全局等级分布（merged.csv 采样 3000 行）
g_new = [20.01, 20.00, 20.00, 20.00, 19.99]
g_old = [36.02, 16.00, 16.00, 15.99, 15.99]
# 验证样本序列等级分布（72 条）
s_new = [37.23, 12.52, 10.60, 11.08, 28.58]
s_old = [52.84, 10.37, 9.25, 9.48, 18.06]

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)
fig.suptitle("129034 九指数 · 两次运行差异溯源（离散等级5，同一批 108 个验证样本）",
             fontsize=16, fontweight="bold", y=0.975)

# ===== 左上：阈值 - OA 曲线 =====
ax = fig.add_subplot(gs[0, 0])
ax.plot(new_thr, new_oa, "o-", color="#2E7DD1", lw=2, ms=5, label="新 run 2026-09-23（序列长 320，初值16/步长8，29轮）")
ax.plot(old_thr, old_oa, "s-", color="#E4572E", lw=2, ms=6, label="旧 run 2026-08-24（序列长 318，初值96/步长25，7轮）")
ax.axhline(78.70, color="#2E7DD1", ls="--", lw=1, alpha=0.5)
ax.axhline(91.67, color="#E4572E", ls="--", lw=1, alpha=0.5)
ax.annotate("78.70%", xy=(240, 78.70), xytext=(150, 70), color="#2E7DD1",
            fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2E7DD1", lw=1.2))
ax.annotate("91.67%", xy=(171, 91.67), xytext=(185, 84), color="#E4572E",
            fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#E4572E", lw=1.2))
ax.set_xlabel("汉明距离阈值 T", fontsize=11)
ax.set_ylabel("总体精度 OA (%)", fontsize=11)
ax.set_title("① 阈值扫描曲线：新 run 全程被压制", fontsize=13, fontweight="bold")
ax.grid(alpha=0.3, ls=":")
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(0, 100)

# ===== 右上：等级分布 =====
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(5)
w = 0.35
ax2.bar(x - w / 2, g_old, w, color="#E4572E", alpha=0.85, label="旧（0 值被排除出分位数统计）")
ax2.bar(x + w / 2, g_new, w, color="#2E7DD1", alpha=0.85, label="新（0 值参与分位数统计）")
for i, (a, b) in enumerate(zip(g_old, g_new)):
    ax2.text(i - w / 2, a + 0.7, f"{a:.1f}", ha="center", fontsize=9, color="#E4572E")
    ax2.text(i + w / 2, b + 0.7, f"{b:.1f}", ha="center", fontsize=9, color="#2E7DD1")
ax2.set_xticks(x)
ax2.set_xticklabels([f"等级{i}" for i in range(1, 6)])
ax2.set_ylabel("占全影像像元的比例 (%)", fontsize=11)
ax2.set_title("② 离散化等级分布：旧 run 等级1 膨胀到 36%", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3, ls=":")
ax2.set_ylim(0, 42)
ax2.text(0.02, 0.93,
         "旧：36 / 16 / 16 / 16 / 16  ← 排除 0 后的数学特征\n"
         "新：20 / 20 / 20 / 20 / 20  ← 严格五等分",
         transform=ax2.transAxes, fontsize=9, va="top",
         bbox=dict(boxstyle="round", fc="#FFF8E1", ec="#C9A227", alpha=0.9))

# ===== 左下：样本序列等级分布 =====
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(x - w / 2, s_old, w, color="#E4572E", alpha=0.85, label="旧 run")
ax3.bar(x + w / 2, s_new, w, color="#2E7DD1", alpha=0.85, label="新 run")
for i, (a, b) in enumerate(zip(s_old, s_new)):
    ax3.text(i - w / 2, a + 0.8, f"{a:.1f}", ha="center", fontsize=9, color="#E4572E")
    ax3.text(i + w / 2, b + 0.8, f"{b:.1f}", ha="center", fontsize=9, color="#2E7DD1")
ax3.set_xticks(x)
ax3.set_xticklabels([f"等级{i}" for i in range(1, 6)])
ax3.set_ylabel("占样本序列单元的比例 (%)", fontsize=11)
ax3.set_title("③ 验证样本的指纹编码分布", fontsize=13, fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3, ls=":")
ax3.set_ylim(0, 60)

# ===== 右下：城镇用地去向（差异来源）=====
ax4 = fig.add_subplot(gs[1, 1])
labels = ["城镇用地\n(40个)", "草地\n(44个)", "水体\n(14个)", "其他建设用地\n(10个)"]
old_ok = [34, 43, 12, 10]
new_ok = [22, 41, 12, 10]
x4 = np.arange(4)
ax4.bar(x4 - w / 2, old_ok, w, color="#E4572E", alpha=0.85, label="旧 run 正确数")
ax4.bar(x4 + w / 2, new_ok, w, color="#2E7DD1", alpha=0.85, label="新 run 正确数")
for i, (a, b) in enumerate(zip(old_ok, new_ok)):
    ax4.text(i - w / 2, a + 0.6, str(a), ha="center", fontsize=10, fontweight="bold", color="#E4572E")
    ax4.text(i + w / 2, b + 0.6, str(b), ha="center", fontsize=10, fontweight="bold", color="#2E7DD1")
    if a != b:
        ax4.annotate("", xy=(i + w / 2, b), xytext=(i + w / 2, a),
                     arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.6))
        ax4.text(i + w / 2 + 0.06, (a + b) / 2, f"-{a-b}", color="#C0392B",
                 fontsize=10, fontweight="bold", va="center")
ax4.set_xticks(x4)
ax4.set_xticklabels(labels, fontsize=10)
ax4.set_ylabel("分类正确的样本数", fontsize=11)
ax4.set_title("④ 精度落差来源：城镇用地 34 → 22", fontsize=13, fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(axis="y", alpha=0.3, ls=":")
ax4.set_ylim(0, 50)
ax4.text(0.03, 0.90,
         "新 run 中 10 个城镇用地被误判为水体、6 个判为草地\n"
         "（水体用户精度 100% → 52.17%）",
         transform=ax4.transAxes, fontsize=9, va="top",
         bbox=dict(boxstyle="round", fc="#FDECEA", ec="#C0392B", alpha=0.9))

out = r"E:\指纹\消融实验\129034九指数_两次运行差异溯源.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("saved:", out)
