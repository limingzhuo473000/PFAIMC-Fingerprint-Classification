# -*- coding: utf-8 -*-
"""验证 _decide_zero_policy 的自动判据在两个真实数据集上是否命中各自的最优口径。

判据期望：
  129034 九指数  p≈20.0%  → k_incl=4 < k_excl=5 → exclude（实测 OA 91.67% > 78.70%）
  124036 全时相  p≈15.7%  → k_incl=5 = k_excl=5 → include（实测 OA 89.91% > 88.07%）
"""
import os, sys, csv, tempfile
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
DST = r"E:\预处理\预处理\网页端 - 副本(全多进程+计时)\网页端 - 副本(全多进程+计时)"
sys.path.insert(0, DST)

from backend.scripts.process01 import _decide_zero_policy, _process_single_quantile_file

rng = np.random.default_rng(20260923)


def make_case(zero_ratio, n=400000, n_classes=5, tag=""):
    """构造一个波段 CSV：zero_ratio 的值为 0（背景/NoData），其余为有效地物连续值。"""
    d = tempfile.mkdtemp(prefix="zp_")
    nz = int(n * (1 - zero_ratio))
    vals = np.concatenate([
        np.zeros(n - nz, dtype=np.float64),
        # 有效地物：对数正态，模拟反射率/指数的右偏分布
        rng.lognormal(mean=6.0, sigma=1.1, size=nz),
    ])
    rng.shuffle(vals)
    rows = vals.reshape(500, -1)
    with open(os.path.join(d, "band_0001.csv"), "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows.tolist())
    return d


print("=" * 78)
print("自动判据验证（模拟两组真实数据集的 0 值占比）")
print("=" * 78)
expect = [
    ("129034 九指数  (p≈20.0%)", 0.200, True,  "排除 0 → 实测 OA 91.67%（包含 0 仅 78.70%）"),
    ("124036 全时相  (p≈15.7%)", 0.157, False, "包含 0 → 实测 OA 89.91%（排除 0 仅 88.07%）"),
]
ok_all = True
for tag, p, want_exclude, note in expect:
    d = make_case(p, tag=tag)
    info = _decide_zero_policy(d, 5, "auto")
    got = bool(info["exclude_zero"])
    ok = (got == want_exclude)
    ok_all &= ok
    print(f"\n【{tag}】")
    print(f"  采样值数量      : {info['sampled_values']}")
    print(f"  实测 0 值占比   : {info['zero_ratio']*100:.2f}%  (设定 {p*100:.1f}%)")
    print(f"  有效地物占用等级: include={info['levels_incl']}/5   exclude={info['levels_excl']}/5")
    print(f"  自动判定        : {'排除 0' if got else '包含 0'}   期望: {'排除 0' if want_exclude else '包含 0'}  → {'✓ 命中' if ok else '✗ 不符'}")
    print(f"  判定理由        : {info['reason']}")
    print(f"  实测对照        : {note}")

print("\n" + "=" * 78)
print("端到端：_process_single_quantile_file 在两种口径下的等级分布")
print("=" * 78)
for tag, p in [("129034 (p=20.0%)", 0.200), ("124036 (p=15.7%)", 0.157)]:
    d = make_case(p)
    src = os.path.join(d, "band_0001.csv")
    for excl in (False, True):
        out = os.path.join(d, f"out_{'excl' if excl else 'incl'}.csv")
        _process_single_quantile_file((src, out, 5, excl))
        with open(out, encoding="utf-8") as f:
            lv = np.array([int(x) for row in csv.reader(f) for x in row])
        dist = [float((lv == g).mean() * 100) for g in range(1, 6)]
        print(f"  {tag}  exclude_zero={str(excl):5s} → " +
              "  ".join(f"L{g}:{dist[g-1]:5.2f}%" for g in range(1, 6)))

print("\n结论:", "自动判据在两个数据集上均命中各自最优口径 ✓" if ok_all else "存在不符，需调整判据 ✗")
