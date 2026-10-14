"""
遥感影像分类系统 - FastAPI 后端入口

启动命令:
    开发环境: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
    生产环境: gunicorn backend.main:app --workers 4 --bind 0.0.0.0:8000
"""

import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core.config import settings
from backend.core.task_manager import task_manager, TaskStatus
from pydantic import BaseModel, Field

# ============================================================
# Lifespan Events
# ============================================================

# 心跳超时后台检测任务句柄
_heartbeat_checker_task: asyncio.Task | None = None

HEARTBEAT_TIMEOUT_SECONDS = 30      # 30秒无心跳自动暂停
HEARTBEAT_CHECK_INTERVAL = 5        # 每5秒检查一次


async def _heartbeat_checker():
    """后台任务：定期检查心跳超时，自动暂停失去前端的任务。"""
    while True:
        try:
            await asyncio.sleep(HEARTBEAT_CHECK_INTERVAL)
            timed_out = task_manager.check_heartbeat_timeouts(HEARTBEAT_TIMEOUT_SECONDS)
            for task_id, task_info in timed_out:
                print(f"[Heartbeat] 任务 {task_id} 心跳超时（{HEARTBEAT_TIMEOUT_SECONDS}s无心跳），自动暂停", flush=True)
                # 标记暂停（step3 多进程循环每 0.25s 轮询此状态并终止子进程；
                # 轮次间的 _check_paused() 也会检测到，无需额外 signal）
                task_manager.pause_task(task_id)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Heartbeat] 检查异常: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    global _heartbeat_checker_task

    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")

    # Ensure temp directory exists
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Temp directory: {settings.temp_dir}")

    # Ensure backend directory exists for static files
    static_dir = Path(__file__).parent.parent / "templates"
    static_dir.mkdir(parents=True, exist_ok=True)

    # 启动心跳超时检测后台任务
    _heartbeat_checker_task = asyncio.create_task(_heartbeat_checker())
    print("[Heartbeat] 心跳超时检测已启动（超时阈值: 30s）")

    yield

    # Shutdown
    print("Shutting down...")
    if _heartbeat_checker_task:
        _heartbeat_checker_task.cancel()


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="遥感影像指纹光谱分类系统 API",
    lifespan=lifespan,
)

# Mount static files for CSS, JS, etc.
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ============================================================
# 开发模式：禁用静态文件缓存（防止 JS/CSS 更新后浏览器还用旧版本）
# ============================================================
class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith('/static/') and path.endswith(('.js', '.css')):
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response

app.add_middleware(NoCacheStaticMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API Routes - Import after app creation to avoid circular imports
# ============================================================

@app.get("/")
async def root():
    """Serve the main page."""
    index_path = Path(__file__).parent.parent / "templates" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "index.html not found", "path": str(index_path)}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }


# ============================================================
# GPU Status API
# ============================================================

@app.get("/gpu/status")
async def gpu_status():
    """GPU状态查询接口 - 返回GPU可用性和设备信息"""
    try:
        from backend.scripts.gpu_optimize import init_gpu, get_gpu_info, USE_GPU
        init_gpu()
        info = get_gpu_info()
        return {
            "gpu_available": USE_GPU,
            "gpu_name": info.get('name', 'N/A'),
            "vram_total_mb": info.get('vram_total_mb', 0),
            "vram_free_mb": info.get('vram_free_mb', 0),
            "cuda_version": info.get('cuda_version', 'N/A'),
            "cupy_version": info.get('cupy_version', 'N/A'),
            "junction_path": info.get('junction_path', None),
            "init_time": info.get('init_time', 0),
            "reason": "" if USE_GPU else "CuPy未安装或GPU初始化失败",
        }
    except Exception as e:
        return {
            "gpu_available": False,
            "gpu_name": "N/A",
            "reason": str(e),
        }


# ============================================================
# Task Management APIs
# ============================================================

class TaskStatusResponse(BaseModel):
    """Task status response model."""
    task_id: str
    status: str
    progress: float = Field(ge=0, le=100)
    message: str
    result: dict | None = None
    error: str | None = None


@app.post("/api/tasks", response_model=dict)
async def create_task():
    """Create a new task and return task ID."""
    task_id = task_manager.create_task()
    return {"task_id": task_id, "message": "Task created"}


@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get task status by ID."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(
        task_id=task.id,
        status=task.status.value,
        progress=task.progress,
        message=task.message,
        result=task.result,
        error=task.error,
    )


@app.get("/api/tasks", response_model=list[dict])
async def list_tasks():
    """List all tasks."""
    tasks = task_manager.list_tasks()
    return [
        {
            "task_id": t.id,
            "status": t.status.value,
            "progress": t.progress,
            "message": t.message,
            "created_at": t.created_at.isoformat(),
        }
        for t in tasks[-20:]  # Return last 20 tasks
    ]


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task."""
    if task_manager.remove_task(task_id):
        return {"message": "Task deleted"}
    raise HTTPException(status_code=404, detail="Task not found")


@app.post("/api/tasks/{task_id}/pause")
async def pause_task(task_id: str):
    """Pause a running task."""
    if task_manager.pause_task(task_id):
        return {"message": "Task paused", "task_id": task_id}
    raise HTTPException(status_code=400, detail="Task not found or not running")


@app.post("/api/tasks/{task_id}/resume")
async def resume_task(task_id: str):
    """Resume a paused task — signal the background task to continue."""
    from backend.routers.pipeline import signal_resume, PauseAction
    if signal_resume(task_id, PauseAction.RESUME):
        return {"message": "Task resumed", "task_id": task_id}
    raise HTTPException(status_code=400, detail="Task not found or not paused")


@app.post("/api/tasks/{task_id}/restart")
async def restart_task(task_id: str):
    """Restart a paused task — signal the background task to start over.
    
    将 RESTART 动作写入 _pause_actions 后，切换 status=RUNNING。
    这会让步骤线程中的 _check_pause 立即解除阻塞后返回，
    pipeline 主循环检测到 RESTART 后抛出 RestartPipelineSignal 重新执行。
    """
    from backend.routers.pipeline import signal_resume, PauseAction, _pause_actions, _lock
    
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    # 允许 RUNNING（步骤内暂停）或 PAUSED（轮次间暂停）
    if task.status not in (TaskStatus.RUNNING, TaskStatus.PAUSED):
        raise HTTPException(status_code=400, detail="Task is not running or paused")
    
    # ★ 写入 RESTART 动作
    with _lock:
        _pause_actions[task_id] = PauseAction.RESTART
    
    # ★ 切换为 RUNNING：解除步骤线程中的 _check_pause 阻塞
    #    pipeline 主循环在 to_thread 返回后会检测 _pause_actions 并重启
    task_manager.update_task(
        task_id,
        status=TaskStatus.RUNNING,
        progress=0.0,
        message="正在重新开始全流程...",
        result={"current_step": 1, "step_message": "正在重新开始...", "step_progress": 0, "current_round": 0},
    )
    
    # 尝试立即唤醒（如果 pipeline 在 asyncio.Event 等待中）
    signal_resume(task_id, PauseAction.RESTART)
    
    return {"message": "Task restart queued", "task_id": task_id}


@app.post("/api/tasks/{task_id}/heartbeat")
async def heartbeat_task(task_id: str):
    """接收前端心跳 —— 证明浏览器窗口仍打开。
    前端每 10s 发送一次。后端 30s 无心跳则自动暂停任务。
    """
    if task_manager.heartbeat(task_id):
        return {"message": "ok"}
    # 任务不存在也返回 ok（可能已被清理，但不影响心跳逻辑）
    return {"message": "ok"}


@app.post("/api/tasks/{task_id}/auto-stop")
async def auto_stop_task(task_id: str):
    """浏览器关闭时自动停止任务（通过 sendBeacon 调用）。
    不返回 400，因为 sendBeacon 是 fire-and-forget 且无法处理错误。
    """
    task = task_manager.get_task(task_id)
    if task and task.status == TaskStatus.RUNNING:
        print(f"[AutoStop] 浏览器窗口关闭，自动暂停任务 {task_id}", flush=True)
        task_manager.pause_task(task_id)
    return {"message": "ok"}


# ============================================================
# File Browser APIs
# ============================================================

class FileNode(BaseModel):
    """File or directory node."""
    name: str
    path: str
    is_dir: bool
    size: int | None = None
    children: list["FileNode"] | None = None


@app.get("/api/files/browse", response_model=list[FileNode])
async def browse_directory(
    path: str = Query(default="", description="Directory path to browse"),
    depth: int = Query(default=1, ge=1, le=3, description="Directory depth to traverse"),
):
    """
    Browse directory and return file tree structure.
    If path is empty, returns available drives on Windows or root on Linux.
    """
    def scan_directory(dir_path: Path, current_depth: int) -> FileNode | None:
        if not dir_path.exists():
            return None

        if not dir_path.is_dir():
            return None

        try:
            children = []
            if current_depth < depth:
                for item in sorted(dir_path.iterdir()):
                    try:
                        child = scan_directory(item, current_depth + 1)
                        if child:
                            children.append(child)
                    except PermissionError:
                        continue
                    except Exception:
                        continue

            return FileNode(
                name=dir_path.name or str(dir_path),
                path=str(dir_path),
                is_dir=True,
                children=children if children else None,
            )
        except PermissionError:
            return None

    def get_initial_nodes(base_path: str) -> list[FileNode]:
        if not base_path:
            # Windows: return drives
            if os.name == 'nt':
                import string
                drives = []
                for letter in string.ascii_uppercase:
                    drive_path = Path(f"{letter}:/")
                    if drive_path.exists():
                        node = scan_directory(drive_path, 1)
                        if node:
                            drives.append(node)
                return drives
            else:
                # Unix: return root
                return [scan_directory(Path("/"), 1)] or []

        path_obj = Path(base_path)
        if not path_obj.exists():
            raise HTTPException(status_code=404, detail="Path not found")

        if path_obj.is_file():
            return [FileNode(
                name=path_obj.name,
                path=str(path_obj),
                is_dir=False,
                size=path_obj.stat().st_size,
            )]

        # Directory
        nodes = []
        try:
            for item in sorted(path_obj.iterdir()):
                try:
                    child = scan_directory(item, 1)
                    if child:
                        nodes.append(child)
                except PermissionError:
                    continue
                except Exception:
                    continue
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")

        return nodes

    return get_initial_nodes(path)


@app.get("/api/files/scan-step1-csv")
async def scan_step1_csv(
    output_dir: str = Query(description="全流程输出根目录路径"),
):
    """
    扫描 output_dir/step1 目录下所有 .csv 文件，返回文件列表及基本信息。
    用于前端「使用已有步骤1结果」字段的快速选择。
    """
    import csv as csv_mod

    step1_dir = Path(output_dir) / "step1"
    if not step1_dir.exists():
        return {"files": [], "step1_dir": str(step1_dir), "exists": False}

    results = []
    for f in sorted(step1_dir.iterdir()):
        if f.suffix.lower() != ".csv":
            continue
        info = {
            "name": f.name,
            "path": str(f),
            "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
            "rows": None,
            "cols": None,
            "error": None,
        }
        # 尝试读取行列数（取前2行判断）
        try:
            with open(f, "r", encoding="utf-8-sig", errors="replace") as fh:
                lines = [fh.readline() for _ in range(2)]
            # 计算列数（第一行逗号数+1）
            info["cols"] = lines[0].count(",") + 1 if lines else 0
            # 统计总行数
            with open(f, "r", encoding="utf-8-sig", errors="replace") as fh:
                info["rows"] = sum(1 for _ in fh)
        except Exception as e:
            info["error"] = str(e)
        results.append(info)

    return {"files": results, "step1_dir": str(step1_dir), "exists": True}


@app.get("/api/files/read")
async def read_file_content(
    path: str = Query(description="File path to read"),
    encoding: str = Query(default="utf-8", description="File encoding"),
    max_lines: int = Query(default=100, ge=1, le=1000),
):
    """Read file content (for text files like CSV, TXT)."""
    file_path = Path(path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")

    # Check file size (limit to 5MB)
    if file_path.stat().st_size > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (>5MB)")

    try:
        lines = file_path.read_text(encoding=encoding).splitlines()[:max_lines]
        return {
            "path": str(file_path),
            "lines": lines,
            "total_lines": len(lines),
            "encoding": encoding,
        }
    except UnicodeDecodeError:
        # Try with different encodings
        for enc in ["gbk", "latin1"]:
            try:
                lines = file_path.read_text(encoding=enc).splitlines()[:max_lines]
                return {
                    "path": str(file_path),
                    "lines": lines,
                    "total_lines": len(lines),
                    "encoding": enc,
                }
            except:
                continue
        raise HTTPException(status_code=400, detail="Unable to decode file")


# ============================================================
# Import and include routers
# ============================================================

# Import routers to register them with the app
from backend.routers import step1, step2, step3, step4, step5, step6, pipeline, preview

app.include_router(step1.router, prefix="/api/step1", tags=["步骤1"])
app.include_router(step2.router, prefix="/api/step2", tags=["步骤2"])
app.include_router(step3.router, prefix="/api/step3", tags=["步骤3"])
app.include_router(step4.router, prefix="/api/step4", tags=["步骤4"])
app.include_router(step5.router, prefix="/api/step5", tags=["步骤5"])
app.include_router(step6.router, prefix="/api/step6", tags=["步骤6"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["全流程"])
app.include_router(preview.router, tags=["预览与转换"])


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        timeout_keep_alive=120,  # 长连接保活：防止服务器负载高时断开轮询请求
    )
