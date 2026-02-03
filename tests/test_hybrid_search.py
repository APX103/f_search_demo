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
    
    # 模拟 Milvus 搜索结果
    mock_services["milvus"].search_by_vector.return_value = [
        {"id": 1, "rank": 1, "product_code": "A", "category": "sofa",
         "description_ai": "test", "image_url": "http://example.com/1.jpg"},
        {"id": 2, "rank": 2, "product_code": "B", "category": "sofa",
         "description_ai": "test", "image_url": "http://example.com/2.jpg"},
    ]
    mock_services["milvus"].search_by_text.return_value = [
        {"id": 1, "rank": 1, "product_code": "A", "category": "sofa",
         "description_ai": "test", "image_url": "http://example.com/1.jpg"},
    ]
    
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
    
    # 验证返回了融合结果
    assert len(results) > 0
    assert all("score" in r for r in results)


@pytest.mark.asyncio
async def test_hybrid_search_respects_top_k(mock_services):
    """测试 top_k 参数"""
    from src.search.hybrid_search import HybridSearchService
    
    # 返回很多结果
    many_results = [
        {"id": i, "rank": i, "product_code": f"P{i}", "category": "sofa",
         "description_ai": "test", "image_url": f"http://example.com/{i}.jpg"}
        for i in range(1, 51)
    ]
    mock_services["milvus"].search_by_vector.return_value = many_results
    mock_services["milvus"].search_by_text.return_value = many_results[:10]
    
    service = HybridSearchService(
        milvus_client=mock_services["milvus"],
        image_encoder=mock_services["image_encoder"],
        text_encoder=mock_services["text_encoder"],
        description_generator=mock_services["desc_generator"]
    )
    
    results, _ = await service.search(image_bytes=b'\xff\xd8', top_k=5)
    
    assert len(results) <= 5
