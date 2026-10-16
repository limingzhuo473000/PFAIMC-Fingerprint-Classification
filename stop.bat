@echo off
chcp 65001 >nul
title 停止遥感分类系统后端

set KILLED=0
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001.*LISTENING"') do (
    echo [停止] 正在结束 PID=%%a ...
    taskkill /PID %%a /F >nul 2>&1
    set KILLED=1
)

if %KILLED%==1 (
    echo [完成] 端口 8001 已释放
) else (
    echo [提示] 端口 8001 当前没有进程在监听
)

pause
