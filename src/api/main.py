# src/api/main.py
"""FastAPI 应用入口"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health, search
from src.api.deps import init_services, cleanup_services


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化服务
    try:
        await init_services()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize services", exc_info=True)
    yield
    # 关闭时清理资源
    await cleanup_services()


app = FastAPI(
    title="Furniture Search API",
    description="基于 AI 设计图的家具相似搜索服务",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, tags=["Health"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
