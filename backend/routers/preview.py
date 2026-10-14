"""
预览与格式转换 API

提供：
- 预览：读取步骤2/步骤3输出CSV的内容，返回JSON供前端展示
- 格式转换：CSV → JSON / TXT，支持下载
"""

import os
import csv
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/preview", tags=["preview"])

# 允许预览的目录白名单（安全限制）
ALLOWED_BASE_DIRS = [
    "E:/output",
    "F:/预处理",
    "D:/output",
    "C:/output",
]


def _is_safe_path(file_path: str) -> bool:
    """安全检查：文件路径必须在白名单目录内"""
    abs_path = os.path.abspath(file_path)
    for base in ALLOWED_BASE_DIRS:
        base_abs = os.path.abspath(base)
        if abs_path.startswith(base_abs):
            return True
    return False


def _read_csv_preview(file_path: str, max_rows: int = 100) -> dict:
    """读取CSV文件的前N行"""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    rows = []
    headers = []
    total_rows = 0
    file_size = os.path.getsize(file_path)

    # 尝试多种编码
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin-1']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    if idx == 0:
                        headers = row
                        continue
                    total_rows += 1
                    if len(rows) < max_rows:
                        rows.append(row)
            break
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    else:
        raise ValueError(f"无法读取文件，所有编码尝试失败")

    return {
        "file": os.path.basename(file_path),
        "path": file_path,
        "size": file_size,
        "encoding": encoding if 'encoding' in dir() else 'unknown',
        "headers": headers,
        "total_rows": total_rows,
        "preview_rows": len(rows),
        "max_rows_limit": max_rows,
        "rows": rows,
    }


@router.get("/csv")
async def preview_csv(
    path: str = Query(..., description="CSV文件绝对路径"),
    rows: int = Query(default=100, ge=1, le=1000, description="预览行数（1-1000）"),
):
    """预览CSV文件内容"""
    if not _is_safe_path(path):
        raise HTTPException(status_code=403, detail="不允许访问该路径")

    try:
        result = _read_csv_preview(path, max_rows=rows)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_files(
    dir_path: str = Query(..., description="目录绝对路径"),
):
    """列出目录下所有文件"""
    if not _is_safe_path(dir_path):
        raise HTTPException(status_code=403, detail="不允许访问该路径")

    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail=f"目录不存在: {dir_path}")

    files = []
    try:
        for name in sorted(os.listdir(dir_path)):
            full_path = os.path.join(dir_path, name)
            is_dir = os.path.isdir(full_path)
            size = os.path.getsize(full_path) if not is_dir else 0
            files.append({
                "name": name,
                "path": full_path,
                "is_dir": is_dir,
                "size": size,
                "ext": os.path.splitext(name)[1].lower() if not is_dir else "",
            })
    except PermissionError:
        raise HTTPException(status_code=403, detail="权限不足，无法读取该目录")

    return {"dir": dir_path, "files": files}


# ================================================================
# 格式转换
# ================================================================

@router.get("/convert")
async def convert_csv(
    path: str = Query(..., description="CSV文件绝对路径"),
    fmt: str = Query(default="json", description="目标格式: json | txt"),
    delimiter: str = Query(default="\t", description="TXT格式的分隔符"),
):
    """将CSV文件转换为 JSON 或 TXT 格式"""
    if not _is_safe_path(path):
        raise HTTPException(status_code=403, detail="不允许访问该路径")

    if fmt not in ("json", "txt"):
        raise HTTPException(status_code=400, detail="支持的格式: json, txt")

    try:
        # 读取CSV
        for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    reader = csv.reader(f)
                    all_rows = list(reader)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("无法读取文件编码")

        headers = all_rows[0] if all_rows else []
        data_rows = all_rows[1:] if len(all_rows) > 1 else []

        if fmt == "json":
            # 转换为JSON对象数组
            json_data = []
            for row in data_rows:
                obj = {}
                for i, header in enumerate(headers):
                    obj[header] = row[i] if i < len(row) else ""
                json_data.append(obj)

            result = json.dumps(json_data, ensure_ascii=False, indent=2)
            base_name = os.path.splitext(os.path.basename(path))[0]
            return {
                "format": "json",
                "file": f"{base_name}.json",
                "row_count": len(data_rows),
                "data": json_data,
            }

        elif fmt == "txt":
            # 转换为TXT（制表符分隔）
            lines = [delimiter.join(headers)]
            for row in data_rows:
                lines.append(delimiter.join(row))
            content = "\n".join(lines)

            base_name = os.path.splitext(os.path.basename(path))[0]
            return {
                "format": "txt",
                "file": f"{base_name}.txt",
                "row_count": len(data_rows),
                "content": content,
            }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"文件不存在: {path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"转换失败: {type(e).__name__}: {e}")


@router.get("/download")
async def download_converted(
    path: str = Query(..., description="CSV文件绝对路径"),
    fmt: str = Query(default="txt", description="目标格式: txt | csv"),
    delimiter: str = Query(default="\t", description="分隔符"),
):
    """下载转换后的文件"""
    if not _is_safe_path(path):
        raise HTTPException(status_code=403, detail="不允许访问该路径")

    try:
        for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    reader = csv.reader(f)
                    all_rows = list(reader)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("无法读取")

        headers = all_rows[0] if all_rows else []
        data_rows = all_rows[1:] if len(all_rows) > 1 else []
        base_name = os.path.splitext(os.path.basename(path))[0]

        if fmt == "csv":
            # 原样返回（其实还是原始文件）
            content = "\n".join(",".join(r) for r in all_rows)
            media_type = "text/csv"
            filename = f"{base_name}.csv"
        else:
            # txt
            lines = [delimiter.join(headers)]
            for row in data_rows:
                lines.append(delimiter.join(row))
            content = "\n".join(lines)
            media_type = "text/plain"
            filename = f"{base_name}.{fmt}"

        return PlainTextResponse(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
