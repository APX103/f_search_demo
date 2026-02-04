# tests/test_milvus_client.py
"""Milvus 客户端测试"""

import pytest
from unittest.mock import Mock, patch, MagicMock


def test_milvus_client_wrapper_init():
    """测试 MilvusClientWrapper 初始化"""
    from src.storage.milvus_client import MilvusClientWrapper
    
    with patch("src.storage.milvus_client.MilvusClient") as mock_client:
        wrapper = MilvusClientWrapper(
            uri="https://test.zillizcloud.com",
            token="test_token",
            collection_name="products"
        )
        
        mock_client.assert_called_once_with(
            uri="https://test.zillizcloud.com",
            token="test_token"
        )
        assert wrapper.collection_name == "products"


def test_milvus_client_wrapper_insert():
    """测试数据插入"""
    from src.storage.milvus_client import MilvusClientWrapper
    
    with patch("src.storage.milvus_client.MilvusClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        wrapper = MilvusClientWrapper(
            uri="https://test.zillizcloud.com",
            token="test_token",
            collection_name="products"
        )
        
        data = [{"product_code": "SF-001", "category": "sofa"}]
        wrapper.insert(data)
        
        mock_client.insert.assert_called_once_with(
            collection_name="products",
            data=data
        )


def test_milvus_client_wrapper_hybrid_search():
    """测试混合搜索"""
    from src.storage.milvus_client import MilvusClientWrapper
    
    with patch("src.storage.milvus_client.MilvusClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # 模拟 hybrid_search 返回结果
        mock_hit = MagicMock()
        mock_hit.__getitem__ = lambda self, key: {
            "id": 1,
            "distance": 0.95,
            "entity": {"product_code": "A", "category": "sofa"}
        }[key]
        mock_client.hybrid_search.return_value = [[mock_hit]]
        
        wrapper = MilvusClientWrapper(
            uri="https://test.zillizcloud.com",
            token="test_token",
            collection_name="products"
        )
        
        search_requests = [
            {"field_name": "image_embedding", "data": [0.1] * 1024, "search_type": "vector", "limit": 10},
            {"field_name": "text_embedding", "data": [0.1] * 2048, "search_type": "vector", "limit": 10},
            {"field_name": "description_ai", "data": "test query", "search_type": "text", "limit": 10},
        ]
        
        results = wrapper.hybrid_search(
            search_requests=search_requests,
            output_fields=["product_code", "category"],
            limit=10,
            rrf_k=60
        )
        
        # 验证 hybrid_search 被调用
        mock_client.hybrid_search.assert_called_once()
        call_kwargs = mock_client.hybrid_search.call_args[1]
        assert call_kwargs["collection_name"] == "products"
        assert call_kwargs["limit"] == 10
        assert len(call_kwargs["reqs"]) == 3


def test_milvus_client_wrapper_single_search():
    """测试单路向量搜索（保留的简单接口）"""
    from src.storage.milvus_client import MilvusClientWrapper
    
    with patch("src.storage.milvus_client.MilvusClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.search.return_value = [[]]
        
        wrapper = MilvusClientWrapper(
            uri="https://test.zillizcloud.com",
            token="test_token",
            collection_name="products"
        )
        
        embedding = [0.1] * 1024
        results = wrapper.search_by_vector(
            field_name="image_embedding",
            vector=embedding,
            limit=20,
            output_fields=["product_code"]
        )
        
        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["collection_name"] == "products"
        assert call_kwargs["anns_field"] == "image_embedding"
        assert call_kwargs["limit"] == 20
