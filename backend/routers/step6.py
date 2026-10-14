"""
步骤6: 评价测试 - API 路由实现
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
from scripts.process06 import run_step6

router = APIRouter()


class Step6Request(BaseModel):
    reclass_dat: str
    shp_folder: str
    output_dir: str


@router.post("/run")
async def run_step6_api(request: Step6Request, background_tasks: BackgroundTasks):
    """运行步骤6：精度评价"""
    dat_file = request.reclass_dat
    if not os.path.exists(dat_file):
        if not dat_file.endswith('.dat'):
            dat_file = dat_file + '.dat'
        if not os.path.exists(dat_file):
            raise HTTPException(status_code=400, detail=f"重分类文件不存在: {request.reclass_dat}")

    if not os.path.exists(request.shp_folder):
        raise HTTPException(status_code=400, detail=f"SHP文件夹不存在: {request.shp_folder}")

    os.makedirs(request.output_dir, exist_ok=True)

    task_id = task_manager.create_task()

    background_tasks.add_task(
        _run_step6_background,
        task_id,
        dat_file,
        request.shp_folder,
        request.output_dir,
    )

    return {"task_id": task_id, "message": "步骤6任务已创建，正在后台执行"}


async def _run_step6_background(
    task_id: str,
    reclass_dat: str,
    shp_folder: str,
    output_dir: str,
):
    try:
        task_manager.update_task(task_id, status=TaskStatus.RUNNING, message="开始评价...", progress=5)

        result = await asyncio.to_thread(
            run_step6,
            task_manager=task_manager,
            task_id=task_id,
            reclass_dat=reclass_dat,
            shp_folder=shp_folder,
            output_dir=output_dir,
        )

        if result["success"]:
            task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                message="评价完成",
                progress=100.0,
                result=result,
            )
        else:
            task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                message=f"评价失败: {result.get('error')}",
                error=result.get("error"),
            )

    except Exception as e:
        error_msg = f"执行出错: {str(e)}"
        traceback.print_exc()
        task_manager.update_task(task_id, status=TaskStatus.FAILED, message=error_msg, error=error_msg)


@router.get("/info")
async def get_step6_info():
    return {
        "name": "精度评价",
        "description": "混淆矩阵、OA、Kappa、用户/生产者精度",
        "function": "重分类结果 + 验证样本 → 混淆矩阵 → 精度指标",
        "inputs": {
            "reclass_dat": "重分类结果文件（_reclass.dat）",
            "shp_folder": "SHP验证文件文件夹",
            "output_dir": "输出目录路径",
        },
        "outputs": {
            "txt": "精度评价报告",
            "csv": "混淆矩阵CSV文件",
        },
        "metrics": [
            "Overall Accuracy (OA) - 总体精度",
            "Kappa Coefficient - Kappa系数",
            "Producer Accuracy - 生产者精度",
            "User Accuracy - 用户精度",
            "Confusion Matrix - 混淆矩阵",
        ],
    }
