# -*- coding: utf-8 -*-
"""移除 ε-容差汉明（后端部分）
- 删除 hamming_tolerance 参数/字段/传参/计算分支
- 「跳过离散化 + 汉明」强制切换为欧氏距离
对 SRC 与 DST 两个项目同时执行；每条规则统计命中数，0 命中会 WARN。
"""
import re, os, sys

sys.stdout.reconfigure(encoding="utf-8")

PROJ = {
    "SRC": r"E:\网页端 - 副本(全多进程+计时)01 - 副本\网页端 - 副本(全多进程+计时)01 - 副本\开发项目",
    "DST": r"E:\预处理\预处理\网页端 - 副本(全多进程+计时)\网页端 - 副本(全多进程+计时)",
}

TOL_BLOCK_NEW = (
    "    # 跳过离散化后指纹为连续浮点值，严格汉明（逐位相等判定）下任意两条序列的距离\n"
    "    # 恒等于序列长度，任何阈值都不会命中。因此该组合强制改用欧氏距离。\n"
    "    if request.skip_discretization and request.distance_metric == 'hamming':\n"
    "        request.distance_metric = 'euclidean'\n"
    "        print(\n"
    "            \"[消融实验1] 已勾选「跳过离散化」：连续浮点序列不适用逐位相等判定，\"\n"
    "            \"距离度量已自动切换为欧氏距离。\",\n"
    "            flush=True,\n"
    "        )\n"
)

RE_TOL_BLOCK = re.compile(
    r"    if request\.skip_discretization and request\.distance_metric == 'hamming':\n"
    r".*?\n    elif request\.hamming_tolerance > 0.*?\n        \)\n",
    re.S,
)

BLOCK_RULES = {
    "backend/scripts/gpu_optimize.py": [
        ("gpu_sig",
         "    distance_metric='hamming',  # 距离度量: 'hamming' / 'euclidean'（消融实验2）\n"
         "    hamming_tolerance=0.0,   # 汉明距离容差 ε（>0 时对连续值启用容差判定，消融实验1）\n",
         "    distance_metric='hamming',  # 距离度量: 'hamming' / 'euclidean'（消融实验2）\n"),
        ("gpu_doc_title",
         "    GPU批量距离计算 - 核心加速函数（支持汉明/欧氏距离，汉明支持 ε 容差）",
         "    GPU批量距离计算 - 核心加速函数（支持汉明/欧氏距离）"),
        ("gpu_doc_alg",
         "        hamming (ε>0，消融实验1配套):\n"
         "            diff = |cells - refs| > ε  →  bool，再 sum(axis=2)\n"
         "            距离语义仍是「不一致波段数」，量纲与经典汉明一致\n",
         ""),
        ("gpu_doc_param",
         "        hamming_tolerance: 汉明距离容差 ε，>0 时 |ref-current| <= ε 视为该波段相等\n",
         ""),
        ("gpu_use_tol",
         "    is_euclidean = (distance_metric == 'euclidean')\n"
         "    use_tol = (not is_euclidean) and (float(hamming_tolerance) > 0)\n",
         "    is_euclidean = (distance_metric == 'euclidean')\n"),
        ("gpu_branch",
         "        elif use_tol:\n"
         "            # ε-容差汉明（消融实验1）\n"
         "            # |diff| > ε 的波段才计 1，其余计 0\n"
         "            # (batch_len, n_refs, seq_len) → sum(axis=2) → (batch_len, n_refs)\n"
         "            diff = cp.abs(batch[:, None, :] - ref_gpu[None, :, :]) > float(hamming_tolerance)\n"
         "            dists = diff.sum(axis=2, dtype=cp.int32)\n",
         ""),
    ],
    "backend/scripts/process03.py": [
        ("p03_build_sig",
         "                      distance_metric='hamming', hamming_tolerance=0.0):",
         "                      distance_metric='hamming'):"),
        ("p03_build_doc_a",
         "        hamming_tolerance: 汉明距离的容差 ε（仅在 distance_metric='hamming' 时生效）。\n"
         "            0 = 严格逐个数值相等判定（经典汉明距离）；\n"
         "            >0 = ε-容差汉明：|ref - current| <= ε 即视为该波段\"相等\"，不计入距离。\n"
         "            用于消融实验1（跳过离散化）：连续浮点序列用严格相等判定会导致\n"
         "            距离恒等于序列长度、任何阈值下都无法匹配（OA=0），必须放宽为容差判定。\n"
         "            建议先对连续值做逐波段标准化，此时 ε 的单位为标准差。\n",
         ""),
        ("p03_build_doc_b",
         "        hamming_tolerance: 汉明距离的容差 ε（仅在 distance_metric='hamming' 时生效）。\n"
         "            0 = 严格逐位相等判定（经典汉明距离）；\n"
         "            >0 = ε-容差汉明：|ref - current| <= ε 即视为该波段\"相等\"，不计入距离。\n"
         "            用于消融实验1（跳过离散化）：连续浮点序列用严格相等判定会导致\n"
         "            距离恒等于序列长度、任何阈值下都无法匹配（OA=0），必须放宽为容差判定。\n"
         "            建议先对连续值做逐波段标准化，此时 ε 的单位为标准差。\n",
         ""),
        ("p03_ret_doc",
         "        ref_matrix:  (total_refs, seq_len)  hamming→np.int16 / euclidean 或 ε>0→np.float32",
         "        ref_matrix:  (total_refs, seq_len)  hamming→np.int16 / euclidean→np.float32"),
        ("p03_use_tol",
         "    _use_tol = (distance_metric == 'hamming' and hamming_tolerance > 0)\n",
         ""),
        ("p03_ref_branch",
         "    elif _use_tol:\n"
         "        # ε-容差汉明（消融实验1的配套判定）：需要做减法比较，必须用 float32\n"
         "        print(\n"
         "            f\"[步骤3] ε-容差汉明距离: ε={hamming_tolerance}，参考矩阵改用 float32\",\n"
         "            flush=True\n"
         "        )\n"
         "        ref_matrix = np.array(ref_list, dtype=np.float32)   # (total_refs, seq_len)\n",
         ""),
        ("p03_match_sig",
         "                          use_gpu=False, distance_metric='hamming',\n"
         "                          hamming_tolerance=0.0):",
         "                          use_gpu=False, distance_metric='hamming'):"),
        ("p03_match_doc",
         "        hamming_tolerance: 汉明距离容差 ε，>0 时逐波段判定放宽为 |ref-current| <= ε 即相等\n",
         ""),
        ("p03_tol_var",
         "    _tol = float(hamming_tolerance) if hamming_tolerance else 0.0\n",
         ""),
        ("p03_gpu_branch",
         "                    elif _tol > 0:\n"
         "                        # ε-容差汉明（消融实验1）：|ref-current| > ε 才计入不一致波段数\n"
         "                        # 距离仍为「不一致波段数」，量纲与经典汉明一致（整数计数）\n"
         "                        dists = (cp.abs(ref_matrix_gpu - current_arr) > _tol).sum(axis=1)\n",
         ""),
        ("p03_cpu_branch",
         "            elif _tol > 0:\n"
         "                # ε-容差汉明（消融实验1）：|ref-current| > ε 才计入不一致波段数\n"
         "                # 距离仍为「不一致波段数」，量纲与经典汉明一致（整数计数）\n"
         "                dists = (np.abs(ref_matrix - current_arr) > _tol).sum(axis=1)\n",
         ""),
        ("p03_run_doc",
         "        hamming_tolerance: 汉明距离容差 ε（仅 hamming 模式生效，默认 0）。\n"
         "            消融实验1（跳过离散化）必须配合 ε>0：连续浮点序列用严格相等判定时\n"
         "            距离恒等于序列长度，任何阈值下都不会命中。\n"
         "            ε>0 时逐波段判定为 |ref - current| <= ε 即相等，距离仍是不一致波段数，\n"
         "            阈值语义与经典汉明一致。建议输入序列先做逐波段标准化，此时 ε 单位为标准差。\n",
         ""),
    ],
    "backend/routers/pipeline.py": [
        ("pipe_field",
         "    hamming_tolerance: float = Field(default=0.0, ge=0.0, le=1000.0, "
         "description=\"消融实验1配套：汉明距离容差 ε。0=严格逐位相等判定（经典汉明）；"
         ">0 时 |ref-current|<=ε 视为该波段相等。跳过离散化后序列为连续浮点值，"
         "必须设 ε>0 否则距离恒=序列长度、必然零匹配。"
         "输入做了逐波段标准化时，ε 的单位为标准差\")\n",
         ""),
        ("pipe_autothr_doc",
         "    自动计算最优初始距离阈值（支持汉明/欧氏距离，汉明支持 ε 容差）。",
         "    自动计算最优初始距离阈值（支持汉明/欧氏距离）。"),
        # SRC 版（use_tol）
        ("pipe_use_tol",
         "    is_euclidean = (distance_metric == 'euclidean')\n"
         "    # ε-容差汉明（消融实验1配套）：|ref-current| <= ε 视为该波段相等\n"
         "    use_tol = (not is_euclidean) and (float(hamming_tolerance) > 0)\n",
         "    is_euclidean = (distance_metric == 'euclidean')\n"),
        ("pipe_dtype_if",
         "    if is_euclidean or use_tol:\n"
         "        # 欧氏距离 / ε-容差汉明：都需要做减法比较，用 float32\n",
         "    if is_euclidean:\n"
         "        # 欧氏距离：需要做减法比较，用 float32\n"),
        ("pipe_cpu_branch",
         "        if use_tol:\n"
         "            # ε-容差汉明：超过 ε 的波段才计 1，距离仍为「不一致波段数」\n"
         "            return (np.abs(matrix - arr) > float(hamming_tolerance)).sum(axis=1)\n",
         ""),
        # DST 版（_tol）
        ("pipe_tol_var",
         "    _tol = float(hamming_tolerance) if hamming_tolerance else 0.0\n"
         "    # 欧氏 / ε-容差模式下必须保留原始浮点值用 float32 参与减法比较\n"
         "    _need_float = _is_euclid or _tol > 0\n",
         "    # 欧氏模式下必须保留原始浮点值用 float32 参与减法比较\n"
         "    _need_float = _is_euclid\n"),
        ("pipe_gpu_log",
         "参考{', ε=%g' % _tol if _tol > 0 else ''})",
         "参考)"),
        ("pipe_gpu_branch",
         "                elif _tol > 0:\n"
         "                    diff = cp.abs(batch[:, None, :] - ref_gpu[None, :, :]) > _tol\n"
         "                    dists = diff.sum(axis=2, dtype=cp.int32)\n",
         ""),
        ("pipe_cpu_comment",
         "        # CPU计算（ε=0 → 严格逐位不等计数；ε>0 → |ref-current|>ε 的波段计数）",
         "        # CPU计算（严格逐位不等计数）"),
        ("pipe_cpu_branch2",
         "            elif _tol > 0:\n"
         "                dists = (np.abs(ref_matrix - current_arr) > _tol).sum(axis=1)\n",
         ""),
        # 全流程 docstring（两项目缩进不同）
        ("pipe_run_doc_a",
         "    hamming_tolerance:   消融实验1配套 — 汉明距离容差 ε。0=严格逐位相等（经典汉明）；\n"
         "                         >0 时 |ref-current|<=ε 视为该波段相等，不计入距离。\n"
         "                         跳过离散化后序列为连续浮点值，必须设 ε>0。\n",
         ""),
        ("pipe_run_doc_b",
         "        hamming_tolerance:   消融实验1配套 — 汉明距离容差 ε。0=严格逐位相等（经典汉明）；\n"
         "                             >0 时 |ref-current|<=ε 视为该波段相等，不计入距离。\n"
         "                             跳过离散化后序列为连续浮点值，必须设 ε>0。\n",
         ""),
        ("pipe_cfg_dict",
         '                "hamming_tolerance": hamming_tolerance,    # 本轮使用的汉明容差 ε（消融实验1配套）\n',
         ""),
    ],
    "backend/routers/step3.py": [
        ("s3_field",
         '    hamming_tolerance: float = Field(default=0.0, ge=0.0, le=1000.0, '
         'description="消融实验1配套：汉明距离容差 ε。0=严格逐位相等（经典汉明）；'
         '>0 时 |ref-current|<=ε 视为该波段相等")\n',
         ""),
    ],
}

# 行级清理（对所有 py 文件）
LINE_RULES = [
    ("line_param_decl", re.compile(r"^[ \t]*hamming_tolerance[ \t]*[:=].*\n", re.M)),
    ("line_request_arg", re.compile(r"^[ \t]*request\.hamming_tolerance,.*\n", re.M)),
]


def patch_file(path, rel):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    orig = src
    report = []

    for tag, old, new in BLOCK_RULES.get(rel, []):
        n = src.count(old)
        if n:
            src = src.replace(old, new)
        report.append((tag, n))

    # 校验块正则替换
    if rel.endswith("pipeline.py"):
        n = len(RE_TOL_BLOCK.findall(src))
        if n:
            src = RE_TOL_BLOCK.sub(lambda m: TOL_BLOCK_NEW, src, count=1)
        report.append(("pipe_tol_validate_block", n))

    for tag, rx in LINE_RULES:
        n = len(rx.findall(src))
        if n:
            src = rx.sub("", src)
        report.append((tag, n))

    changed = (src != orig)
    if changed:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)
    return report, changed


for pname, root in PROJ.items():
    print(f"\n{'='*70}\n[{pname}] {root}\n{'='*70}")
    for rel in BLOCK_RULES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"  MISSING {rel}")
            continue
        report, changed = patch_file(path, rel)
        warns = [t for t, n in report if n == 0]
        print(f"  {rel}: changed={changed}")
        print("    " + ", ".join(f"{t}={n}" for t, n in report))
        if warns:
            print(f"    [WARN 0命中] {warns}")
