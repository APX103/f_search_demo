# tests/test_health_api.py
"""健康检查 API 测试"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """创建测试客户端"""
    from src.api.main import app
    return TestClient(app)


def test_health_check_returns_ok(client):
    """测试健康检查端点"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_health_check_returns_version(client):
    """测试健康检查返回版本号"""
    response = client.get("/health")
    
    data = response.json()
    assert "version" in data
