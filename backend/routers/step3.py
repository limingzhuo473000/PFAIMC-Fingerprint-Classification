"""
步骤3: 匹配测试 - API 路由实现
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
from scripts.process03 import run_step3

router = APIRouter()


class RoundConfig(BaseModel):
    threshold: float = Field(ge=0)
    replacements: dict[str, str]


class Step3Request(BaseModel):
    images_folder: str
    samples_folder: str
    target_file: str
    output_file: str
    match_order: list[str]
    rounds: list[RoundConfig]
    distance_metric: str = Field(default='hamming', description="消融实验2：距离度量，'hamming'=汉明距离（默认），'euclidean'=欧氏距离匹配")


@router.post("/run")
async def run_step3_api(request: Step3Request, background_tasks: BackgroundTasks):
    """运行步骤3：多轮距离匹配（汉明/欧氏，消融实验2可选）"""
    if not os.path.exists(request.images_folder):
        raise HTTPException(status_code=400, detail=f"影像文件夹不存在: {request.images_folder}")
    if not os.path.exists(request.samples_folder):
        raise HTTPException(status_code=400, detail=f"样本文件夹不存在: {request.samples_folder}")
    if not os.path.exists(request.target_file):
        raise HTTPException(status_code=400, detail=f"目标文件不存在: {request.target_file}")
    if request.distance_metric not in ('hamming', 'euclidean'):
        raise HTTPException(status_code=400, detail=f"不支持的距离度量: {request.distance_metric}（可选 'hamming' 或 'euclidean'）")

    task_id = task_manager.create_task()

    # 转换 rounds 为 dict 格式
    rounds_config = [
        {"threshold": r.threshold, "replacements": r.replacements}
        for r in request.rounds
    ]

    background_tasks.add_task(
        _run_step3_background,
        task_id,
        request.images_folder,
        request.samples_folder,
        request.target_file,
        request.output_file,
        request.match_order,
        rounds_config,
        request.distance_metric,
    )

    return {"task_id": task_id, "message": "步骤3任务已创建，正在后台执行"}


async def _run_step3_background(
    task_id: str,
    images_folder: str,
    samples_folder: str,
    target_file: str,
    output_file: str,
    match_order: list[str],
    rounds: list[dict],
    distance_metric: str = 'hamming',
):
    try:
        task_manager.update_task(task_id, status=TaskStatus.RUNNING, message="开始匹配...", progress=5)

        result = await asyncio.to_thread(
            run_step3,
            task_manager=task_manager,
            task_id=task_id,
            images_folder=images_folder,
            samples_folder=samples_folder,
            target_file=target_file,
            output_file=output_file,
            match_order=match_order,
            rounds=rounds,
            distance_metric=distance_metric,
        )

        if result["success"]:
            task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                message="匹配完成",
                progress=100.0,
                result=result,
            )
        else:
            task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                message=f"匹配失败: {result.get('error')}",
                error=result.get("error"),
            )

    except Exception as e:
        error_msg = f"执行出错: {str(e)}"
        traceback.print_exc()
        task_manager.update_task(task_id, status=TaskStatus.FAILED, message=error_msg, error=error_msg)


@router.get("/info")
async def get_step3_info():
    return {
        "name": "汉明距离匹配",
        "description": "多轮汉明距离最近邻匹配",
        "function": "样本序列 → 多轮距离匹配 → 类别赋值",
        "inputs": {
            "images_folder": "影像图文件夹（确定序列长度）",
            "samples_folder": "样本文件夹",
            "target_file": "待匹配的CSV文件",
            "output_file": "输出CSV文件路径",
            "match_order": "匹配顺序（样本文件名列表）",
            "rounds": "匹配轮次配置",
        },
        "outputs": {"csv": "匹配后的CSV文件"},
    }
