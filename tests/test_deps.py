# tests/test_deps.py
"""依赖注入测试"""

import pytest
from unittest.mock import patch, MagicMock


def test_get_settings_returns_singleton():
    """测试配置单例"""
    from src.config import get_settings
    
    # 两次调用返回相同实例
    settings1 = get_settings()
    settings2 = get_settings()
    
    assert settings1 is settings2


@pytest.mark.asyncio
async def test_init_services_creates_search_service(monkeypatch):
    """测试服务初始化"""
    monkeypatch.setenv("ZILLIZ_CLOUD_URI", "https://test.zillizcloud.com")
    monkeypatch.setenv("ZILLIZ_CLOUD_TOKEN", "test_token")
    monkeypatch.setenv("ZILLIZ_CLOUD_COLLECTION", "products")
    monkeypatch.setenv("ZHIPU_API_KEY", "zhipu_key")
    monkeypatch.setenv("ALIYUN_DASHSCOPE_API_KEY", "aliyun_key")
    
    with patch("src.api.deps.ZillizClient") as mock_zilliz:
        from src.api.deps import init_services, get_search_service
        
        await init_services()
        
        # 验证 Zilliz 客户端被创建
        mock_zilliz.assert_called_once()
