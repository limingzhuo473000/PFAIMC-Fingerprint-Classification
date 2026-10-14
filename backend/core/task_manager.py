"""Task status management utilities."""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
import threading


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """Task information container."""
    id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    # 暂停/恢复相关
    resume_state: Optional[dict] = None  # 暂停时保存的恢复状态
    # 心跳监控（浏览器关闭自动停止）
    last_heartbeat: Optional[datetime] = None  # 最后一次心跳时间
    heartbeat_timeout: bool = False  # 心跳是否超时


class TaskManager:
    """Thread-safe task status manager."""

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()

    def create_task(self) -> str:
        """Create a new task and return its ID."""
        task_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._tasks[task_id] = TaskInfo(id=task_id)
        return task_id

    def update_task(self, task_id: str, **kwargs) -> bool:
        """Update task status and metadata."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            task = self._tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.updated_at = datetime.now()
            return True

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task information by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> list[TaskInfo]:
        """List all tasks."""
        with self._lock:
            return list(self._tasks.values())

    def pause_task(self, task_id: str, state: Optional[dict] = None) -> bool:
        """Pause a running task."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            task = self._tasks[task_id]
            if task.status != TaskStatus.RUNNING:
                return False
            task.status = TaskStatus.PAUSED
            task.message = "已暂停"
            if state:
                task.resume_state = state
            task.updated_at = datetime.now()
            return True

    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            task = self._tasks[task_id]
            if task.status != TaskStatus.PAUSED:
                return False
            task.status = TaskStatus.RUNNING
            task.message = "恢复执行中..."
            task.updated_at = datetime.now()
            return True

    def heartbeat(self, task_id: str) -> bool:
        """Update the last heartbeat time for a task (called by frontend polling)."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            self._tasks[task_id].last_heartbeat = datetime.now()
            self._tasks[task_id].heartbeat_timeout = False
            return True

    def check_heartbeat_timeouts(self, timeout_seconds: int = 30) -> list[tuple[str, TaskInfo]]:
        """Check all RUNNING tasks for heartbeat timeout.
        Returns list of (task_id, TaskInfo) that timed out.
        Only checks tasks that have ever received a heartbeat (last_heartbeat is not None).
        """
        now = datetime.now()
        timed_out = []
        with self._lock:
            for task_id, task in self._tasks.items():
                if task.status != TaskStatus.RUNNING:
                    continue
                if task.last_heartbeat is None:
                    continue  # 还没收到过心跳，跳过
                elapsed = (now - task.last_heartbeat).total_seconds()
                if elapsed > timeout_seconds:
                    task.heartbeat_timeout = True
                    timed_out.append((task_id, task))
        return timed_out

    def remove_task(self, task_id: str) -> bool:
        """Remove a completed task."""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                return True
            return False


# Global task manager instance
task_manager = TaskManager()
