FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY furniture_search/ /app/furniture_search/
COPY furniture_search/pyproject.toml /app/
COPY furniture_search/README.md /app/

# 安装 Python 依赖
RUN pip install --no-cache-dir -e /app

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["python", "-m", "furniture_search.server"]
