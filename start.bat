@echo off
chcp 65001 >nul
title 遥感分类系统 - 后端服务 (端口 8001)

cd /d "%~dp0"

REM ===========================================================
REM GPU加速环境变量设置（仅在已安装 CuPy 时启用）
REM 2026-09-18：nvidia-*/cupy 已卸载，无 CuPy 时自动跳过，不产生无用目录与链接。
REM 恢复方法：取消 requirements.txt 中 cupy-cuda12x[ctk] 的注释并 pip install，本段会自动生效。
REM 修复闪退：%~dp0 含括号路径，变量必须在 if 块外用引号定义，块内引用必须加引号，
REM 否则路径中的 ")" 会提前终止 if 括号块导致整个脚本退出。
REM ===========================================================
set "PROJDIR=%~dp0"
set "CUPY_CACHE_DIR=%~dp0.cupy_cache"
set "CUDA_CACHE_PATH=%~dp0.cuda_cache"

venv\Scripts\python.exe -c "import cupy" >nul 2>&1
if %errorlevel% equ 0 (
    if not exist "%CUPY_CACHE_DIR%" mkdir "%CUPY_CACHE_DIR%" >nul 2>&1
    if not exist "%CUDA_CACHE_PATH%" mkdir "%CUDA_CACHE_PATH%" >nul 2>&1
    if not exist "C:\rs_gpu" mklink /J "C:\rs_gpu" "%PROJDIR%" >nul 2>&1
    echo [GPU] CuPy 已就绪，GPU 加速可用
) else (
    echo [GPU] 未检测到 CuPy，跳过 GPU 初始化（使用 CPU 多进程模式）
)

REM 先停掉已占用的端口
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001.*LISTENING"') do (
    echo [清理] 结束占用端口 8001 的进程 PID=%%a
    taskkill /PID %%a /F >nul 2>&1
)

echo [启动] 正在启动 FastAPI 后端...
REM 注意：--reload 仅开发调试用；执行长任务时请务必去掉，否则代码保存/改动会触发服务重启，导致内存中的任务状态丢失。
venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8001

echo.
echo [退出] 后端服务已停止
pause
