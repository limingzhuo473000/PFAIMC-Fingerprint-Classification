# 遥感影像分类系统 - Docker 配置

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（GDAL, rasterio 需要）
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY backend/ ./backend/
COPY templates/ ./templates/

# 创建临时目录
RUN mkdir -p /tmp/classification

# 暴露端口
EXPOSE 8000

# 环境变量
ENV PYTHONPATH=/app
ENV TEMP_DIR=/tmp/classification

# 启动命令
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
