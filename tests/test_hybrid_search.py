# tests/test_hybrid_search.py
"""混合搜索服务测试"""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture
def mock_services():
    """创建模拟服务"""
    mock_milvus = MagicMock()
    mock_image_encoder = MagicMock()
    mock_text_encoder = MagicMock()
    mock_desc_generator = MagicMock()
    
    # 设置模拟返回值
    mock_image_encoder.encode = AsyncMock(return_value=np.array([0.1] * 1024))
    mock_text_encoder.encode = AsyncMock(return_value=np.array([0.1] * 2048))
    mock_desc_generator.generate = AsyncMock(return_value="[COLOR] Gray\n[STYLE] Modern")
    
    # 模拟 hybrid_search 返回结果（与 Zilliz Cloud schema 对齐）
    mock_milvus.hybrid_search.return_value = [
        {"id": 1, "rank": 1, "sku": "A", "name": "Sofa A", "category": "sofa",
         "price": "$999", "description": "test", "LLMDescription": "test llm",
         "url": "http://example.com/a", "imageUrl": "http://example.com/1.jpg", "score": 0.95},
        {"id": 2, "rank": 2, "sku": "B", "name": "Sofa B", "category": "sofa",
         "price": "$899", "description": "test", "LLMDescription": "test llm",
         "url": "http://example.com/b", "imageUrl": "http://example.com/2.jpg", "score": 0.85},
    ]
    
    return {
        "milvus": mock_milvus,
        "image_encoder": mock_image_encoder,
        "text_encoder": mock_text_encoder,
        "desc_generator": mock_desc_generator
    }


@pytest.mark.asyncio
async def test_hybrid_search_service_search(mock_services):
    """测试混合搜索"""
    from src.search.hybrid_search import HybridSearchService
    
    service = HybridSearchService(
        milvus_client=mock_services["milvus"],
        image_encoder=mock_services["image_encoder"],
        text_encoder=mock_services["text_encoder"],
        description_generator=mock_services["desc_generator"]
    )
    
    results, description = await service.search(
        image_bytes=b'\xff\xd8\xff\xe0',
        top_k=10
    )
    
    # 验证调用了编码器和生成器
    mock_services["image_encoder"].encode.assert_called_once()
    mock_services["desc_generator"].generate.assert_called_once()
    mock_services["text_encoder"].encode.assert_called_once()
    
    # 验证调用了 hybrid_search（而不是分开的 search）
    mock_services["milvus"].hybrid_search.assert_called_once()
    
    # 验证返回了融合结果
    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[0]["score"] == 0.95


@pytest.mark.asyncio
async def test_hybrid_search_builds_correct_requests(mock_services):
    """测试构建正确的搜索请求"""
    from src.search.hybrid_search import HybridSearchService
    
    service = HybridSearchService(
        milvus_client=mock_services["milvus"],
        image_encoder=mock_services["image_encoder"],
        text_encoder=mock_services["text_encoder"],
        description_generator=mock_services["desc_generator"]
    )
    
    await service.search(image_bytes=b'\xff\xd8', top_k=5)
    
    # 验证 hybrid_search 被调用时的参数
    call_kwargs = mock_services["milvus"].hybrid_search.call_args[1]
    
    # 应该有 3 个搜索请求
    assert len(call_kwargs["search_requests"]) == 3
    
    # 验证字段名（与 Zilliz Cloud schema 对齐）
    field_names = [req["field_name"] for req in call_kwargs["search_requests"]]
    assert "imageVector" in field_names
    assert "vector" in field_names
    assert "LLMDescription" in field_names
    
    # 验证 limit 和 rrf_k
    assert call_kwargs["limit"] == 5
    assert call_kwargs["rrf_k"] == 60
