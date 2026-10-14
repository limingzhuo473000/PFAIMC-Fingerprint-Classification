"""
步骤5: 重分类优化 - API 路由实现
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
from scripts.process05 import run_step5

router = APIRouter()


class LandCoverClass(BaseModel):
    name: str
    values: list[int]


class Step5Request(BaseModel):
    dat_path: str
    classes: list[LandCoverClass]


@router.post("/run")
async def run_step5_api(request: Step5Request, background_tasks: BackgroundTasks):
    """运行步骤5：ENVI重分类"""
    dat_file = request.dat_path
    if not os.path.exists(dat_file):
        if not dat_file.endswith('.dat'):
            dat_file = dat_file + '.dat'
        if not os.path.exists(dat_file):
            raise HTTPException(status_code=400, detail=f"DAT文件不存在: {request.dat_path}")

    task_id = task_manager.create_task()

    classes_data = [{"name": c.name, "values": c.values} for c in request.classes]

    background_tasks.add_task(
        _run_step5_background,
        task_id,
        dat_file,
        classes_data,
    )

    return {"task_id": task_id, "message": "步骤5任务已创建，正在后台执行"}


async def _run_step5_background(
    task_id: str,
    dat_path: str,
    classes: list[dict],
):
    try:
        task_manager.update_task(task_id, status=TaskStatus.RUNNING, message="开始重分类...", progress=5)

        result = await asyncio.to_thread(
            run_step5,
            task_manager=task_manager,
            task_id=task_id,
            dat_path=dat_path,
            classes=classes,
        )

        if result["success"]:
            task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                message="重分类完成",
                progress=100.0,
                result=result,
            )
        else:
            task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                message=f"重分类失败: {result.get('error')}",
                error=result.get("error"),
            )

    except Exception as e:
        error_msg = f"执行出错: {str(e)}"
        traceback.print_exc()
        task_manager.update_task(task_id, status=TaskStatus.FAILED, message=error_msg, error=error_msg)


@router.get("/info")
async def get_step5_info():
    return {
        "name": "ENVI重分类",
        "description": "将数值映射为地物类别",
        "function": "ENVI分类结果 → 数值重映射 → 类别赋值（保留地理坐标）",
        "inputs": {
            "dat_path": "ENVI .dat 文件路径",
            "classes": "地物类别定义列表",
        },
        "outputs": {
            "dat": "重分类后的DAT文件",
            "hdr": "重分类后的HDR文件（含类别名称和颜色）",
        },
        "notes": [
            "0值 → 未分类（灰色）",
            "-1值 → 边框（黑色）",
            "其他值 → 用户定义的类别",
        ],
    }
