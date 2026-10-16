"""
离散化算法示意图生成脚本
使用 matplotlib 绘制连续 DN 值 -> 阈值划分 -> 离散类别的流程
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
ax.set_xlim(0, 680)
ax.set_ylim(0, 330)
ax.axis("off")

# ---- 颜色定义 ----
C_TITLE = "#444441"
C_LABEL = "#888780"
C_THRESHOLD = "#534AB7"
C_AXIS = "#888780"

# 四个类别的颜色 (fill, edge, text)
CAT_COLORS = {
    "水体": ("#E6F1FB", "#185FA5", "#0C447C"),
    "林地": ("#EAF3DE", "#3B6D11", "#27500A"),
    "建筑": ("#FAEEDA", "#854F0B", "#633806"),
    "农田": ("#E1F5EE", "#0F6E56", "#085041"),
}

# 阈值位置 (x 坐标)
T1, T2, T3 = 225, 340, 465

# ==================== 标题 ====================
ax.text(340, 312, "离散化算法流程", ha="center", va="center",
        fontsize=13, fontweight="bold", color=C_TITLE)

# ==================== 阈值标签 ====================
for tx, label in [(T1, "$T_1$"), (T2, "$T_2$"), (T3, "$T_3$")]:
    ax.text(tx, 290, label, ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_THRESHOLD)
    ax.plot([tx, tx], [286, 280], color=C_THRESHOLD, linewidth=0.6)

# ==================== 连续 DN 值色带 ====================
# 用渐变色带模拟连续光谱
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("dn", ["#E6F1FB", "#85B7EB", "#378ADD", "#042C53"])
gradient = np.linspace(0, 1, 256).reshape(1, -1)
ax.imshow(gradient, aspect="auto", cmap=cmap,
          extent=[120, 560, 252, 280], zorder=2)

# 色带边框
band_rect = mpatches.FancyBboxPatch((120, 252), 440, 28,
    boxstyle="round,pad=0,rounding_size=3",
    linewidth=0.5, edgecolor="#B4B2A9", facecolor="none", zorder=3)
ax.add_patch(band_rect)

# 色带标签
ax.text(112, 266, "连续 DN 值", ha="right", va="center",
        fontsize=10, color=C_LABEL)

# 数值刻度
for x, val in [(120, "0"), (340, "128"), (560, "255")]:
    ax.plot([x, x], [252, 246], color=C_AXIS, linewidth=0.5)
    ax.text(x, 240, val, ha="center", va="center", fontsize=9, color=C_LABEL)

# ==================== 虚线: 色带 -> 处理框 ====================
for tx in [T1, T2, T3]:
    ax.plot([tx, tx], [252, 215], color=C_AXIS, linewidth=0.5,
            linestyle=(0, (4, 3)), zorder=1)

# ==================== 离散化处理框 ====================
proc_rect = FancyBboxPatch((140, 165), 400, 50,
    boxstyle="round,pad=0,rounding_size=8",
    linewidth=0.5, edgecolor="#5F5E5A", facecolor="#F1EFE8", zorder=2)
ax.add_patch(proc_rect)
ax.text(340, 200, "离散化处理", ha="center", va="center",
        fontsize=11, fontweight="bold", color=C_TITLE)
ax.text(340, 180, "按阈值 $T_1$/$T_2$/$T_3$ 分段映射为类别",
        ha="center", va="center", fontsize=10, color=C_LABEL)

# 虚线穿过处理框
for tx in [T1, T2, T3]:
    ax.plot([tx, tx], [165, 215], color="#B4B2A9", linewidth=0.5,
            linestyle=(0, (3, 2)), zorder=1)

# ==================== 虚线: 处理框 -> 离散色块 ====================
for tx in [T1, T2, T3]:
    ax.plot([tx, tx], [165, 135], color=C_AXIS, linewidth=0.5,
            linestyle=(0, (4, 3)), zorder=1)

# ==================== 离散类别色块 ====================
ax.text(112, 115, "离散类别", ha="right", va="center",
        fontsize=10, color=C_LABEL)

categories = [
    ("水体", 120, 105, "$C_1$"),
    ("林地", 225, 115, "$C_2$"),
    ("建筑", 340, 125, "$C_3$"),
    ("农田", 465,  95, "$C_4$"),
]

for name, x, w, clabel in categories:
    fill, edge, txt = CAT_COLORS[name]
    rect = FancyBboxPatch((x, 95), w, 40,
        boxstyle="round,pad=0,rounding_size=3",
        linewidth=0.5, edgecolor=edge, facecolor=fill, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w / 2, 115, name, ha="center", va="center",
            fontsize=11, color=txt, fontweight="bold")
    ax.text(x + w / 2, 80, clabel, ha="center", va="center",
            fontsize=9, color=C_LABEL)

# ==================== 底部公式 ====================
formula_rect = FancyBboxPatch((180, 24), 320, 36,
    boxstyle="round,pad=0,rounding_size=6",
    linewidth=0.5, edgecolor="#B4B2A9", facecolor="none",
    linestyle="--", zorder=2)
ax.add_patch(formula_rect)
ax.text(340, 42, r"$C(x) = C_k$,  若 $T_k \leq x < T_{k+1}$,  $k = 1 \ldots 4$",
        ha="center", va="center", fontsize=11, color="#5F5E5A")

plt.tight_layout(pad=0.5)

# ==================== 保存 ====================
output_path = r"F:\预处理\预处理\网页端 - 副本(全多进程+计时)\网页端 - 副本(全多进程+计时)\discretization_diagram.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()

print(f"示意图已保存至: {output_path}")
