"""
步骤4: ENVI进阶 - API 路由实现
"""

import os
import sys
import asyncio
import traceback
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.core.task_manager import task_manager, TaskStatus
from scripts.process04 import run_step4

router = APIRouter()


class Step4Request(BaseModel):
    input_csv: str
    images_folder: str
    output_dir: str
    add_geo: bool = True
    max_sequence_length: Optional[int] = None


@router.post("/run")
async def run_step4_api(request: Step4Request, background_tasks: BackgroundTasks):
    """运行步骤4：CSV转ENVI格式"""
    if not os.path.exists(request.input_csv):
        raise HTTPException(status_code=400, detail=f"CSV文件不存在: {request.input_csv}")
    if not os.path.exists(request.images_folder):
        raise HTTPException(status_code=400, detail=f"影像文件夹不存在: {request.images_folder}")

    os.makedirs(request.output_dir, exist_ok=True)

    task_id = task_manager.create_task()

    background_tasks.add_task(
        _run_step4_background,
        task_id,
        request.input_csv,
        request.images_folder,
        request.output_dir,
        request.add_geo,
        request.max_sequence_length,
    )

    return {"task_id": task_id, "message": "步骤4任务已创建，正在后台执行"}


async def _run_step4_background(
    task_id: str,
    input_csv: str,
    images_folder: str,
    output_dir: str,
    add_geo: bool,
    max_sequence_length: Optional[int],
):
    try:
        task_manager.update_task(task_id, status=TaskStatus.RUNNING, message="开始转换...", progress=5)

        result = await asyncio.to_thread(
            run_step4,
            task_manager=task_manager,
            task_id=task_id,
            input_csv=input_csv,
            images_folder=images_folder,
            output_dir=output_dir,
            add_geo=add_geo,
            max_sequence_length=max_sequence_length,
        )

        if result["success"]:
            task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                message="转换完成",
                progress=100.0,
                result=result,
            )
        else:
            task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                message=f"转换失败: {result.get('error')}",
                error=result.get("error"),
            )

    except Exception as e:
        error_msg = f"执行出错: {str(e)}"
        traceback.print_exc()
        task_manager.update_task(task_id, status=TaskStatus.FAILED, message=error_msg, error=error_msg)


@router.get("/info")
async def get_step4_info():
    return {
        "name": "ENVI格式转换",
        "description": "CSV → TXT → ENVI 三步自动化",
        "function": "CSV文件 → TXT格式 → ENVI栅格文件（含地理坐标）",
        "inputs": {
            "input_csv": "输入CSV文件路径",
            "images_folder": "影像图文件夹（确定序列长度）",
            "output_dir": "输出目录路径",
            "add_geo": "是否添加地理坐标",
            "max_sequence_length": "最大序列长度过滤",
        },
        "outputs": {
            "dat": "ENVI数据文件",
            "hdr": "ENVI头文件",
            "txt": "中间TXT文件",
        },
    }
