"""
步骤1: 影像填充合并 - API 路由

功能：影像文件 → CSV → 零值处理 → 分位数分类 → 按像元位置合并
"""

import os
import sys
import asyncio
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# 添加 backend 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.core.task_manager import task_manager, TaskStatus
from scripts.process01 import run_step1

router = APIRouter()


# ============================================================
# Request/Response Models
# ============================================================

class BandRange(BaseModel):
    """波段范围定义"""
    start: int = Field(ge=1, description="起始波段")
    end: int = Field(ge=1, description="结束波段")


class Step1Request(BaseModel):
    """步骤1请求参数"""
    input_img_dir: str = Field(description="影像文件夹路径")
    output_dir: str = Field(description="输出目录路径")
    filename: str = Field(description="输出文件名（不含扩展名）")
    num_classes: int = Field(default=5, ge=2, le=20, description="分位数分类数量")
    delete_ranges: Optional[list[BandRange]] = Field(
        default=None,
        description="要删除的波段范围列表"
    )
    adjust_sizes: bool = Field(default=True, description="是否自动调整不同尺寸的CSV文件")
    skip_discretization: bool = Field(default=False, description="消融实验1：跳过分位数离散化，直接合并零值处理后的数值文件")


class Step1Response(BaseModel):
    """步骤1响应"""
    task_id: str
    message: str


# ============================================================
# API Endpoints
# ============================================================

@router.post("/run", response_model=Step1Response)
async def run_step1_api(request: Step1Request, background_tasks: BackgroundTasks):
    """
    运行步骤1：影像填充合并处理

    参数：
    - input_img_dir: 影像文件夹路径（包含 .tif/.tiff/.img 文件）
    - output_dir: 输出目录路径
    - filename: 输出文件名（不含扩展名）
    - num_classes: 分位数分类数量（默认5）
    - delete_ranges: 要删除的波段范围（可选）
    - adjust_sizes: 是否调整不同尺寸的CSV文件（默认True）
    """
    # 验证路径
    if not os.path.exists(request.input_img_dir):
        raise HTTPException(status_code=400, detail=f"影像文件夹不存在: {request.input_img_dir}")

    # 创建输出目录
    os.makedirs(request.output_dir, exist_ok=True)

    # 创建任务
    task_id = task_manager.create_task()

    # 处理 delete_ranges
    delete_ranges = None
    if request.delete_ranges:
        delete_ranges = [(r.start, r.end) for r in request.delete_ranges]

    # 后台执行
    background_tasks.add_task(
        _run_step1_background,
        task_id,
        request.input_img_dir,
        request.output_dir,
        request.filename,
        request.num_classes,
        delete_ranges,
        request.adjust_sizes,
        request.skip_discretization,
    )

    return Step1Response(
        task_id=task_id,
        message="步骤1任务已创建，正在后台执行"
    )


async def _run_step1_background(
    task_id: str,
    input_img_dir: str,
    output_dir: str,
    filename: str,
    num_classes: int,
    delete_ranges: list | None,
    adjust_sizes: bool,
    skip_discretization: bool = False,
):
    """后台执行步骤1的实际逻辑"""
    try:
        task_manager.update_task(
            task_id,
            status=TaskStatus.RUNNING,
            message="开始处理影像数据...",
            progress=0.0,
        )

        # 执行处理
        result = await asyncio.to_thread(
            run_step1,
            task_manager=task_manager,
            task_id=task_id,
            input_img_dir=input_img_dir,
            output_dir=output_dir,
            filename=filename,
            num_classes=num_classes,
            delete_ranges=delete_ranges,
            adjust_sizes=adjust_sizes,
            skip_discretization=skip_discretization,
        )

        if result["success"]:
            task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                message="步骤1处理完成",
                progress=100.0,
                result={
                    "output_file": result.get("output_file", ""),
                    "processed_files": result.get("processed_files", 0),
                    "total_pixels": result.get("total_pixels", 0),
                },
            )
        else:
            task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                message=f"处理失败: {result.get('error', '未知错误')}",
                error=result.get("error", "未知错误"),
            )

    except Exception as e:
        error_msg = f"执行出错: {str(e)}"
        print(f"[Step1 Error] {error_msg}")
        traceback.print_exc()
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=error_msg,
            error=error_msg,
        )


@router.get("/info")
async def get_step1_info():
    """
    获取步骤1的功能说明和参数说明
    """
    return {
        "name": "影像填充合并",
        "description": "将遥感影像文件转换为指纹光谱序列并合并",
        "function": "影像文件(.tif/.tiff/.img) → CSV → 零值处理 → 分位数分类 → 按像元位置合并",
        "inputs": {
            "input_img_dir": "影像文件夹路径（包含 .tif/.tiff/.img 文件）",
            "output_dir": "输出目录路径",
            "filename": "输出文件名（不含扩展名）",
            "num_classes": "分位数分类数量（2-20，默认5）",
            "delete_ranges": "要删除的波段范围列表，如 [(1,7), (58,76)]",
            "adjust_sizes": "是否自动调整不同尺寸的CSV文件（默认True）",
        },
        "outputs": {
            "csv": "合并后的CSV文件",
            "temp": "临时处理文件（自动清理）",
        },
        "notes": [
            "支持批量处理文件夹中的所有影像文件",
            "会自动跳过空文件和非CSV文件",
            "分位数分类将数据按等分位数划分为1-N类",
            "合并时会将同一像元位置的所有波段值用空格连接",
        ],
    }
