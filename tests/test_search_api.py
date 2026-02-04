# tests/test_search_api.py
"""搜索 API 测试"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import io


@pytest.fixture
def mock_search_service():
    """模拟搜索服务"""
    mock = MagicMock()
    mock.search = AsyncMock(return_value=(
        [
            {
                "id": 1,
                "sku": "SF-001",
                "name": "Modern Sofa",
                "category": "sofa",
                "price": "$999",
                "description": "A comfortable sofa",
                "LLMDescription": "[COLOR] Gray",
                "url": "http://example.com/product/SF-001",
                "imageUrl": "http://example.com/1.jpg",
                "score": 0.85,
                "rank": 1
            }
        ],
        "[COLOR] Gray\n[STYLE] Modern"
    ))
    return mock


@pytest.fixture
def client(mock_search_service):
    """创建带模拟服务的测试客户端"""
    from src.api.main import app
    from src.api import deps
    
    # 替换依赖
    app.dependency_overrides[deps.get_search_service] = lambda: mock_search_service
    
    client = TestClient(app)
    yield client
    
    # 清理
    app.dependency_overrides.clear()


def test_search_image_returns_results(client, mock_search_service):
    """测试图像搜索返回结果"""
    # 创建测试图片
    image_content = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
    files = {"image": ("test.jpg", io.BytesIO(image_content), "image/jpeg")}
    
    response = client.post("/api/v1/search/image", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "results" in data["data"]
    assert len(data["data"]["results"]) == 1


def test_search_image_with_top_k(client, mock_search_service):
    """测试 top_k 参数"""
    image_content = b'\xff\xd8\xff\xe0\x00\x10JFIF'
    files = {"image": ("test.jpg", io.BytesIO(image_content), "image/jpeg")}
    
    response = client.post("/api/v1/search/image?top_k=5", files=files)
    
    assert response.status_code == 200
    # 验证 search 被调用时传入了 top_k=5
    mock_search_service.search.assert_called_once()
    call_kwargs = mock_search_service.search.call_args[1]
    assert call_kwargs["top_k"] == 5


def test_search_image_with_category_hint(client, mock_search_service):
    """测试 category_hint 参数"""
    image_content = b'\xff\xd8\xff\xe0\x00\x10JFIF'
    files = {"image": ("test.jpg", io.BytesIO(image_content), "image/jpeg")}
    
    response = client.post(
        "/api/v1/search/image?category_hint=sofa",
        files=files
    )
    
    assert response.status_code == 200
    call_kwargs = mock_search_service.search.call_args[1]
    assert call_kwargs["category_hint"] == "sofa"


def test_search_image_missing_file(client):
    """测试缺少图片文件"""
    response = client.post("/api/v1/search/image")
    
    assert response.status_code == 422  # Validation Error
