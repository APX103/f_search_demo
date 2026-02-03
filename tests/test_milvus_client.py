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


def test_milvus_client_wrapper_search():
    """测试向量搜索"""
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
