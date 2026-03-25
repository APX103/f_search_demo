# tests/test_hybrid_search.py
"""混合搜索服务测试"""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def mock_services():
    """创建模拟服务"""
    mock_zilliz = MagicMock()
    mock_image_encoder = MagicMock()
    mock_text_encoder = MagicMock()
    mock_desc_generator = MagicMock()

    mock_image_encoder.encode = AsyncMock(return_value=np.array([0.1] * 1024))
    mock_text_encoder.encode = AsyncMock(return_value=np.array([0.1] * 2048))
    mock_desc_generator.generate = AsyncMock(return_value="[COLOR] Gray\n[STYLE] Modern")

    sample_results = [
        {"id": 1, "rank": 1, "sku": "A", "name": "Sofa A", "category": "sofa",
         "price": "$999", "description": "test", "LLMDescription": "test llm",
         "url": "http://example.com/a", "imageUrl": "http://example.com/1.jpg", "score": 0.95},
        {"id": 2, "rank": 2, "sku": "B", "name": "Sofa B", "category": "sofa",
         "price": "$899", "description": "test", "LLMDescription": "test llm",
         "url": "http://example.com/b", "imageUrl": "http://example.com/2.jpg", "score": 0.85},
    ]

    mock_zilliz.search = AsyncMock(side_effect=[
        sample_results,  # image ANN
        sample_results,  # text ANN
    ])
    mock_zilliz.search_by_text = AsyncMock(return_value=sample_results)  # BM25

    return {
        "zilliz": mock_zilliz,
        "image_encoder": mock_image_encoder,
        "text_encoder": mock_text_encoder,
        "desc_generator": mock_desc_generator,
    }


@pytest.mark.asyncio
async def test_hybrid_search_service_search(mock_services):
    """测试三路混合搜索"""
    from src.search.hybrid_search import HybridSearchService

    service = HybridSearchService(
        zilliz_client=mock_services["zilliz"],
        image_encoder=mock_services["image_encoder"],
        text_encoder=mock_services["text_encoder"],
        description_generator=mock_services["desc_generator"],
    )

    results, description = await service.search(
        image_bytes=b'\xff\xd8\xff\xe0',
        top_k=10
    )

    mock_services["image_encoder"].encode.assert_called_once()
    mock_services["desc_generator"].generate.assert_called_once()
    mock_services["text_encoder"].encode.assert_called_once()

    assert mock_services["zilliz"].search.call_count == 2
    mock_services["zilliz"].search_by_text.assert_called_once()

    assert len(results) > 0
    assert all("score" in r for r in results)


@pytest.mark.asyncio
async def test_hybrid_search_passes_category_hint(mock_services):
    """测试 category_hint 传递到融合"""
    from src.search.hybrid_search import HybridSearchService

    service = HybridSearchService(
        zilliz_client=mock_services["zilliz"],
        image_encoder=mock_services["image_encoder"],
        text_encoder=mock_services["text_encoder"],
        description_generator=mock_services["desc_generator"],
    )

    await service.search(
        image_bytes=b'\xff\xd8\xff\xe0',
        top_k=10,
        category_hint="sofa"
    )

    assert mock_services["zilliz"].search.call_count == 2
    mock_services["zilliz"].search_by_text.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_search_respects_top_k(mock_services):
    """测试 top_k 参数"""
    from src.search.hybrid_search import HybridSearchService

    many_results = [
        {"id": i, "rank": i, "sku": f"P{i}", "name": f"Product {i}", "category": "sofa",
         "price": f"${i * 100}", "description": "test", "LLMDescription": "test",
         "url": f"http://example.com/{i}", "imageUrl": f"http://example.com/{i}.jpg", "score": 0.9}
        for i in range(1, 51)
    ]

    mock_services["zilliz"].search = AsyncMock(side_effect=[
        many_results,
        many_results,
    ])
    mock_services["zilliz"].search_by_text = AsyncMock(return_value=many_results)

    service = HybridSearchService(
        zilliz_client=mock_services["zilliz"],
        image_encoder=mock_services["image_encoder"],
        text_encoder=mock_services["text_encoder"],
        description_generator=mock_services["desc_generator"],
    )

    results, _ = await service.search(image_bytes=b'\xff\xd8', top_k=5)

    assert len(results) <= 5
