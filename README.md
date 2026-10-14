# 遥感影像分类系统 · Remote Sensing Image Classification System

> 基于光谱指纹迭代匹配分类（SFIMC）的高光谱遥感影像智能分类 Web 平台。  
> An intelligent web platform for hyperspectral remote sensing image classification based on **SFIMC (Spectral Fingerprint Iterative Matching Classification)**.

---

## 项目简介 · Introduction

**中文**

本项目是一个面向高光谱遥感影像分类任务的全流程 Web 应用，采用前后端分离架构。系统集成了影像预处理、样本分析、多轮光谱指纹匹配、分类结果输出与精度评价等核心环节，支持一键式完整处理管道（Pipeline），并具备任务管理、心跳超时暂停、多进程加速等工程特性。

**English**

This project is a full-process web application for hyperspectral remote sensing image classification, built with a frontend–backend separated architecture. It integrates core stages including image preprocessing, sample analysis, multi-round spectral fingerprint matching, classification result export, and accuracy assessment, and supports a one-click full pipeline with task management, heartbeat timeout pausing, and multi-process acceleration.

---

## 核心特性 · Features

| 中文 | English |
|---|---|
| 六步模块化分类流程 | Six-step modular classification workflow |
| 多级递进迭代阈值匹配 | Multi-level progressive iterative threshold matching |
| 汉明距离与欧氏距离可选 | Hamming / Euclidean distance selectable for ablation |
| 异步后台任务与进度推送 | Async background tasks with progress reporting |
| 30 秒心跳超时自动暂停 | Auto-pause on 30-second heartbeat timeout |
| 多进程 + 可选 GPU（CuPy）加速 | Multi-process + optional GPU (CuPy) acceleration |
| 一键式全流程 Pipeline | One-click full processing pipeline |
| 容器化部署（Docker / docker-compose） | Containerized deployment (Docker / docker-compose) |

---

## 技术栈 · Tech Stack

- **后端 · Backend**: Python 3.11, FastAPI, Uvicorn, Pydantic
- **前端 · Frontend**: HTML5, CSS3, Vanilla JavaScript
- **数据处理 · Data**: NumPy, pandas, rasterio, geopandas, Pillow
- **机器学习 · ML**: scikit-learn (accuracy metrics)
- **GPU 加速 · GPU**: CuPy (optional, NVIDIA CUDA 12.x)
- **部署 · Deployment**: Docker, docker-compose

---

## 快速开始 · Quick Start

### 1. 克隆仓库 · Clone

```bash
git clone https://github.com/mfzou/Fingerprint-Classification.git
cd Fingerprint-Classification
```

### 2. 本地运行 · Run Locally

```bash
# 创建虚拟环境 · Create virtual environment
python -m venv venv

# Windows 激活 · Activate on Windows
venv\Scripts\activate

# 安装依赖 · Install dependencies
pip install -r requirements.txt

# 启动服务 · Start server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

浏览器访问 · Open in browser: [http://localhost:8000](http://localhost:8000)

### 3. Docker 部署 · Docker Deployment

```bash
docker-compose up --build
```

服务默认暴露端口 `8000`。  
The service is exposed on port `8000` by default.

---

## 处理流程 · Workflow

系统提供六个独立步骤与一个全自动 Pipeline：  
The system provides six independent steps and a fully automatic pipeline:

1. **影像填充合并 · Image Filling & Merging** — 影像 → CSV → 零值处理 → 分位数分类 → 按像元位置合并。
2. **样本进阶 · Sample Advancement** — 基于 SHP 样本提取像元光谱并做差异性分析。
3. **匹配测试 · Matching Test** — 多轮光谱指纹匹配（汉明 / 欧氏距离，消融实验）。
4. **ENVI 进阶 · ENVI Export** — CSV 转 ENVI `.dat` / `.hdr` 格式。
5. **重分类优化 · Reclassification** — 根据地物类别进行 ENVI 重分类。
6. **评价测试 · Accuracy Assessment** — 计算 OA、Kappa、PA、UA 等精度指标。

**Pipeline 终止条件 · Pipeline stop criteria**  
- Kappa ≥ 阈值（默认 0.8）直接终止。  
- 未分类像素比例 < 阈值（默认 10%）终止。  
- 达到最大迭代轮次强制终止。  
- 启用冻结策略时所有类别均冻结则提前终止。

---

## 项目结构 · Project Structure

```text
.
├── backend/
│   ├── core/              # 配置与任务管理 · Config & task manager
│   ├── routers/           # API 路由：step1 ~ step6, pipeline, preview
│   ├── scripts/           # 处理脚本 process01 ~ process06
│   └── main.py            # FastAPI 入口 · FastAPI entrypoint
├── static/                # 前端静态资源 · Frontend static assets
├── templates/             # HTML 模板 · HTML templates
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## API 文档 · API Documentation

启动服务后，可通过以下地址访问交互式文档：  
After starting the server, interactive API docs are available at:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 注意事项 · Notes

- 依赖 `rasterio` 和 `geopandas` 需要 GDAL 库；Docker 镜像已内置系统依赖。  
  `rasterio` and `geopandas` require GDAL; system dependencies are pre-installed in the Docker image.
- `cupy-cuda12x` 为可选 GPU 加速依赖，无 NVIDIA 显卡时仍可 CPU 运行。  
  `cupy-cuda12x` is optional for GPU acceleration; CPU mode works without an NVIDIA GPU.
- 生产环境建议使用 `gunicorn` 并配置反向代理。  
  For production, use `gunicorn` behind a reverse proxy.

---

## 许可 · License

本项目仅供学习与研究使用。  
This project is for research and educational purposes only.
