# tests/test_zilliz_client.py
"""Zilliz Cloud REST API 客户端测试"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


def test_zilliz_client_init():
    """测试 ZillizClient 初始化"""
    from src.storage.zilliz_client import ZillizClient
    
    client = ZillizClient(
        endpoint="https://test.cloud.zilliz.com",
        token="test_token",
        collection_name="products"
    )
    
    assert client.endpoint == "https://test.cloud.zilliz.com"
    assert client.collection_name == "products"
    assert "Bearer test_token" in client.headers["Authorization"]


@pytest.mark.asyncio
async def test_zilliz_client_hybrid_search():
    """测试混合搜索"""
    from src.storage.zilliz_client import ZillizClient
    
    client = ZillizClient(
        endpoint="https://test.cloud.zilliz.com",
        token="test_token",
        collection_name="products"
    )
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "code": 0,
        "data": [
            {"id": 1, "distance": 0.95, "name": "Sofa A"},
            {"id": 2, "distance": 0.85, "name": "Sofa B"},
        ]
    }
    mock_response.raise_for_status = MagicMock()
    
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock()
        
        results = await client.hybrid_search(
            vector_searches=[
                {"data": [0.1] * 2048, "anns_field": "vector", "limit": 10},
                {"data": [0.1] * 1024, "anns_field": "imageVector", "limit": 10},
            ],
            limit=10,
            rrf_k=60
        )
        
        # 验证 API 调用
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "hybrid_search" in call_args[0][0]
        
        # 验证结果
        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[0]["score"] == 0.95


@pytest.mark.asyncio
async def test_zilliz_client_search():
    """测试单路搜索"""
    from src.storage.zilliz_client import ZillizClient
    
    client = ZillizClient(
        endpoint="https://test.cloud.zilliz.com",
        token="test_token",
        collection_name="products"
    )
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "code": 0,
        "data": [{"id": 1, "distance": 0.95, "name": "Sofa A"}]
    }
    mock_response.raise_for_status = MagicMock()
    
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock()
        
        results = await client.search(
            vector=[0.1] * 2048,
            anns_field="vector",
            limit=10
        )
        
        assert len(results) == 1
        assert results[0]["id"] == 1
