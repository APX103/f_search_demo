# src/api/routes/health.py
"""健康检查路由"""

import logging
from typing import Optional, Dict

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    components: Optional[Dict[str, str]] = None


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """健康检查端点 — 检查服务组件状态"""
    components = {}
    
    # Check search service initialization
    try:
        from src.api.deps import get_search_service
        get_search_service()
        components["search_service"] = "ok"
    except RuntimeError:
        components["search_service"] = "not_initialized"
    except Exception as e:
        components["search_service"] = f"error: {e}"
    
    is_healthy = all(v == "ok" for v in components.values())
    
    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        version="1.0.0",
        components=components
    )
