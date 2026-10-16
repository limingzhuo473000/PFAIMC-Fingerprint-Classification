# -*- coding: utf-8 -*-
"""
为什么「跳过离散化 + 严格汉明」OA 恒为 0？
用真实样本序列说话（纯读磁盘，不跑服务）：
  基线 step2（5级离散编码） vs 本次 step2（原始连续值）
  同类地物样本两两之间的汉明距离分布对比。
"""
import os, csv, glob, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = r"E:\指纹\124036全时相\离散等级5\step2"      # 5级离散
STRICT = r"E:\指纹\消融实验\124036无离散化汉明\step2"  # 原始连续值
N = 200                                            # 序列长度（= 时相文件数）
T_MAX = 130                                        # 25 轮阈值上限（10 起、步长5）

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "为什么无离散化加汉明等于0.png")


def load(folder):
    """返回 {类别: [序列(list[str]), ...]}"""
    d = {}
    for f in glob.glob(os.path.join(folder, "*.csv")):
        cls = os.path.basename(f).replace(".csv", "")
        seqs = []
        with open(f, encoding="utf-8-sig", errors="ignore") as fh:
            for row in csv.DictReader(fh):
                v = (row.get("提取值") or "").split()
                if len(v) == N:
                    seqs.append(v)
        if seqs:
            d[cls] = seqs
    return d


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def pair_dist(d):
    out = []
    for cls, seqs in d.items():
        for a, b in itertools.combinations(seqs[:12], 2):   # 每类取前12个样本两两比较
            out.append(hamming(a, b))
    return np.array(out)


b = pair_dist(load(BASE))
s = pair_dist(load(STRICT))

print("=== 同类地物样本两两汉明距离（序列长度 N=200）===")
print(f"5级离散编码： 均值 {b.mean():.1f}  最小 {b.min()}  最大 {b.max()}  样本对 {len(b)}")
print(f"原始连续值： 均值 {s.mean():.1f}  最小 {s.min()}  最大 {s.max()}  样本对 {len(s)}")

fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))
fig.suptitle("为什么「跳过离散化 + 严格汉明」精度必然为 0 —— 序列长度 200 vs 阈值上限 130",
             fontsize=15, fontweight="bold", y=0.99)

# ── ① 指纹序列示意 ──────────────────────────────────
ax = axes[0]
ax.set_title("① 一个像元的「指纹序列」= 200 个时相值排成一行", fontsize=12, fontweight="bold")
np.random.seed(0)
disc = np.random.choice([1, 2, 3, 4, 5], 24)
cont = np.array([9641, 9851, 10325, 10712, 11808, 12707, 11807, 11520, 10990, 10233,
                 9960, 9812, 10045, 10678, 11234, 11567, 11201, 10789, 10256, 9876,
                 9654, 9543, 9612, 9780])
for yi, (name, vals, cmap) in enumerate([
        ("5级离散化后\n取值∈{1..5}", disc, plt.cm.tab10),
        ("不离散化（原始DN）\n取值∈[0,65535]", cont, plt.cm.viridis)]):
    y0 = 1 - yi
    for i, v in enumerate(vals):
        ax.add_patch(plt.Rectangle((i, y0 - 0.42), 0.92, 0.84,
                                   color=plt.cm.tab10(v / 5) if yi == 0
                                   else plt.cm.viridis((v - cont.min()) / (cont.max() - cont.min())),
                                   ec="white", lw=.6))
    ax.text(-1.2, y0, name, ha="right", va="center", fontsize=10)
ax.set_xlim(-1.2, 25.5); ax.set_ylim(-0.6, 1.6); ax.axis("off")
ax.text(11, -0.55, "…… 共 200 个位置（200 个时相/波段文件）", ha="center", fontsize=10, color="#555")

# ── ② 同类地物两两汉明距离分布 ────────────────────────
ax = axes[1]
ax.set_title("② 同类地物样本两两距离（越低越容易匹配）", fontsize=12, fontweight="bold")
bins = np.arange(0, 205, 5)
ax.hist(b, bins=bins, alpha=.75, color="#2c3e50", label=f"5级离散（均值 {b.mean():.0f}）")
ax.hist(s, bins=bins, alpha=.75, color="#c0392b", label=f"原始连续值（均值 {s.mean():.0f}）")
ax.axvline(30, ls="--", color="#2c3e50", lw=1.6)
ax.text(32, ax.get_ylim()[1] * .82, "首轮阈值 T=10\n（离散时大量命中）", fontsize=9, color="#2c3e50")
ax.axvline(T_MAX, ls="--", color="#c0392b", lw=1.6)
ax.text(T_MAX - 6, ax.get_ylim()[1] * .55, f"25轮阈值上限 T={T_MAX}", fontsize=9,
        color="#c0392b", ha="right")
ax.axvline(200, color="black", lw=2)
ax.text(198, ax.get_ylim()[1] * .3, "序列长度 200\n（连续值恒等于此）", fontsize=9, ha="right")
ax.set_xlabel("汉明距离（200 位中不相等的个数）"); ax.set_ylabel("样本对数量")
ax.legend(fontsize=9); ax.grid(alpha=.25, axis="y")

# ── ③ 阈值永远追不上 ────────────────────────────────
ax = axes[2]
ax.set_title("③ 阈值扫描 vs 满值距离 200", fontsize=12, fontweight="bold")
Ts = np.arange(10, 131, 5)
ax.plot(Ts, np.full_like(Ts, 200), "o-", color="#c0392b", lw=2.4, ms=5,
        label="连续值实际距离 = 200（恒值）")
ax.plot(Ts, Ts, lw=2.2, color="#2c3e50", label="阈值 T（10 → 130）")
ax.fill_between(Ts, Ts, 200, color="#c0392b", alpha=.12)
ax.text(70, 165, "距离恒 > 阈值 → 判定「未分类」", ha="center", fontsize=11,
        color="#c0392b", fontweight="bold")
ax.set_xlabel("轮次阈值 T"); ax.set_ylabel("距离 / 阈值")
ax.set_ylim(0, 220); ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=.25)
ax.annotate("109 个验证样本\n全部落入未分类 → OA = 0.00%",
            xy=(130, 200), xytext=(62, 92), fontsize=11, color="#c0392b", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved:", OUT)
