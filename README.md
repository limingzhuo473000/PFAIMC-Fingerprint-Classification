# 遥感影像分类系统 · Remote Sensing Image Classification System

> 基于像素时序-光谱指纹自适应迭代匹配分类（PFAIMC）的多时序多光谱遥感影像土地覆被分类 Web 平台。
> An intelligent web platform for multi-temporal multispectral land-cover classification based on **PFAIMC (Pixel Temporal-spectral Fingerprint Adaptive Iterative Matching Classification)**.

---

## 项目简介 · Introduction

**中文**

本项目是一个面向多时序多光谱遥感影像土地覆被分类任务的全流程 Web 应用，采用前后端分离架构。PFAIMC 将连续反射率分类重构为离散整数序列上的比特级匹配：全场景五级分位数离散化把含噪反射率时序映射为 1–5 级整数指纹；无均值聚合的多模态指纹模板库逐像元保留完整参考指纹，去除多元正态先验；匹配阶段以汉明距离逐级递进阈值，先归入高置信核心像元，再渐进接纳边界像元。方法无需训练、纯 CPU 运行，在郑州城乡结合部场景收敛 OA 89.91%（Kappa = 0.851），并可在无重训练条件下跨六个站点、两种传感器（Landsat-8/9、Sentinel-2）保持 80.99%–96.55% 的 OA。

**English**

This project is a full-process web application for multi-temporal multispectral land-cover classification, built with a frontend–backend separated architecture. PFAIMC reformulates continuous-reflectance classification as bit-level matching over discrete integer sequences: a scene-wide five-level quantile discretization maps noisy reflectance time series into integer fingerprints of levels 1–5; a mean-aggregation-free multimodal template library stores intact reference-pixel fingerprints without any normality prior; a progressive multi-level threshold then admits high-confidence core pixels first and edge-ambiguous pixels later, using Hamming distance. The method is training-free and runs on CPU only: on the Zhengzhou peri-urban site it converged at OA = 89.91% (Kappa = 0.851), and generalized without retraining across six sites and two sensors (Landsat-8/9, Sentinel-2) to 80.99%–96.55% OA.

---

## 核心特性 · Features

| 中文 | English |
|---|---|
| 五级分位数时序-光谱指纹离散化 | Five-level quantile temporal-spectral fingerprint discretization |
| 无均值聚合多模态指纹模板库 | Mean-aggregation-free multimodal fingerprint template library |
| 汉明距离比特级匹配 + 多级递进阈值 | Bit-level Hamming matching with progressive multi-level thresholds |
| 免训练、纯 CPU 推理 | Training-free, CPU-only inference |
| 跨区域 / 跨传感器迁移（Landsat-8/9、Sentinel-2） | Cross-site / cross-sensor transferability |
| 六位一体全流程 + 一键 Pipeline | Six-step workflow with one-click full pipeline |
| 异步后台任务与进度推送 | Async background tasks with progress reporting |
| 多进程 + 可选 GPU（CuPy）加速 | Multi-process + optional GPU (CuPy) acceleration |
| 容器化部署（Docker / docker-compose） | Containerized deployment (Docker / docker-compose) |

---

## 技术栈 · Tech Stack

- **后端 · Backend**: Python 3.11, FastAPI, Uvicorn, Pydantic
- **前端 · Frontend**: HTML5, CSS3, Vanilla JavaScript
- **数据处理 · Data**: NumPy, pandas, rasterio, geopandas, Pillow
- **精度评价 · Accuracy**: scikit-learn (OA / PA / UA / Kappa)
- **GPU 加速 · GPU**: CuPy (optional, NVIDIA CUDA 12.x)
- **部署 · Deployment**: Docker, docker-compose

---

## 快速开始 · Quick Start

### 1. 克隆仓库 · Clone

```bash
git clone https://github.com/limingzhuo473000/PFAIMC-Fingerprint-Classification.git
cd PFAIMC-Fingerprint-Classification
```

### 2. 本地运行 · Run Locally

```bash
# 创建虚拟环境 · Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

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

---

## 处理流程 · Workflow

系统提供六个独立步骤与一个全自动 Pipeline：

1. **影像填充合并 · Image Filling & Merging** — 影像 → CSV → 零值处理 → 分位数离散化 → 按像元位置合并。
2. **样本进阶 · Sample Advancement** — 基于 SHP 样本提取像元光谱并构建指纹模板库。
3. **匹配测试 · Matching Test** — 多轮汉明距离递进匹配（含消融实验开关）。
4. **ENVI 进阶 · ENVI Export** — CSV 转 ENVI `.dat` / `.hdr` 格式。
5. **重分类优化 · Reclassification** — 根据地物类别进行 ENVI 重分类。
6. **评价测试 · Accuracy Assessment** — 计算 OA、Kappa、PA、UA 等精度指标。

**Pipeline 终止条件 · Stop criteria**
- Kappa ≥ 阈值（默认 0.8）直接终止。
- 未分类像素比例 < 阈值（默认 10%）终止。
- 达到最大迭代轮次（默认 25）强制终止。
- 启用冻结策略时所有类别均冻结则提前终止。

---

## 项目结构 · Project Structure

```text
.
├── backend/
│   ├── core/              # 配置与任务管理 · Config & task manager
│   ├── routers/           # API 路由：step1 ~ step6, pipeline, preview
│   ├── scripts/          # 处理脚本 process01 ~ process06
│   └── main.py           # FastAPI 入口 · FastAPI entrypoint
├── static/                # 前端静态资源 · Frontend static assets
├── templates/             # HTML 模板 · HTML templates
├── 消融实验产物/           # Ablation experiment results
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## API 文档 · API Documentation

启动服务后，可通过以下地址访问交互式文档：

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 注意事项 · Notes

- 依赖 `rasterio` 和 `geopandas` 需要 GDAL 库；Docker 镜像已内置系统依赖。
- `cupy-cuda12x` 为可选 GPU 加速依赖，无 NVIDIA 显卡时仍可 CPU 运行。
- PFAIMC 推理本身不依赖 GPU；CuPy 仅用于大规模矩阵运算加速。
- 生产环境建议使用 `gunicorn` 并配置反向代理。

---

## 许可 · License

本项目仅供学习与研究使用。
This project is for research and educational purposes only.
