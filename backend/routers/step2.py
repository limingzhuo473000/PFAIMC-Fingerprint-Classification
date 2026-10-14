"""
步骤2: 样本进阶 - API 路由实现
"""

import os
import sys
import asyncio
import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.core.task_manager import task_manager, TaskStatus
from scripts.process02 import run_step2

router = APIRouter()


class Step2Request(BaseModel):
    shp_input: str
    tif_folder: str
    experiment_data_path: str
    output_dir: str


@router.post("/run")
async def run_step2_api(request: Step2Request, background_tasks: BackgroundTasks):
    """运行步骤2：SHP样本提取与差异性分析"""
    if not os.path.exists(request.shp_input):
        raise HTTPException(status_code=400, detail=f"SHP路径不存在: {request.shp_input}")
    if not os.path.exists(request.tif_folder):
        raise HTTPException(status_code=400, detail=f"TIF文件夹不存在: {request.tif_folder}")
    if not os.path.exists(request.experiment_data_path):
        raise HTTPException(status_code=400, detail=f"实验数据文件不存在: {request.experiment_data_path}")

    os.makedirs(request.output_dir, exist_ok=True)

    task_id = task_manager.create_task()

    background_tasks.add_task(
        _run_step2_background,
        task_id,
        request.shp_input,
        request.tif_folder,
        request.experiment_data_path,
        request.output_dir,
    )

    return {"task_id": task_id, "message": "步骤2任务已创建，正在后台执行"}


async def _run_step2_background(
    task_id: str,
    shp_input: str,
    tif_folder: str,
    experiment_data_path: str,
    output_dir: str,
):
    try:
        task_manager.update_task(task_id, status=TaskStatus.RUNNING, message="开始处理...", progress=5)

        result = await asyncio.to_thread(
            run_step2,
            task_manager=task_manager,
            task_id=task_id,
            shp_input=shp_input,
            tif_folder=tif_folder,
            experiment_data_path=experiment_data_path,
            output_dir=output_dir,
        )

        if result["success"]:
            task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                message="步骤2处理完成",
                progress=100.0,
                result=result,
            )
        else:
            task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                message=f"处理失败: {result.get('error', '未知错误')}",
                error=result.get("error"),
            )

    except Exception as e:
        error_msg = f"执行出错: {str(e)}"
        traceback.print_exc()
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=error_msg,
            error=error_msg,
        )


@router.get("/info")
async def get_step2_info():
    return {
        "name": "样本进阶分析",
        "description": "SHP样本点像素提取与汉明差异性分析",
        "function": "SHP样本点 → 像素值提取 → 组内/组间差异分析 → 推荐匹配顺序",
        "inputs": {
            "shp_input": "SHP文件或文件夹路径",
            "tif_folder": "TIF影像文件夹路径",
            "experiment_data_path": "实验数据CSV文件路径（步骤1输出）",
            "output_dir": "输出目录路径",
        },
        "outputs": {
            "csv": "每个类别的样本CSV文件",
            "txt": "匹配顺序建议文件",
        },
    }
