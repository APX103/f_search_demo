# src/api/routes/health.py
"""健康检查路由"""

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    健康检查端点
    
    用于负载均衡器和监控系统检测服务状态
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )
