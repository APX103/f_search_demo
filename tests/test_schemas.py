# tests/test_schemas.py
"""数据模型测试"""

import pytest
from pydantic import ValidationError


def test_search_result_model():
    """测试搜索结果模型"""
    from src.models.schemas import SearchResult
    
    result = SearchResult(
        product_id=123,
        product_code="SF-001",
        category="sofa",
        description_ai="[COLOR] Dark gray...",
        image_url="https://example.com/image.jpg",
        score=0.85,
        rank=1
    )
    
    assert result.product_id == 123
    assert result.product_code == "SF-001"
    assert result.score == 0.85


def test_search_request_model():
    """测试搜索请求模型"""
    from src.models.schemas import SearchRequest
    
    request = SearchRequest(top_k=10, category_hint="sofa")
    
    assert request.top_k == 10
    assert request.category_hint == "sofa"


def test_search_request_default_values():
    """测试搜索请求默认值"""
    from src.models.schemas import SearchRequest
    
    request = SearchRequest()
    
    assert request.top_k == 20
    assert request.category_hint is None


def test_product_input_model():
    """测试商品输入模型"""
    from src.models.schemas import ProductInput
    
    product = ProductInput(
        product_code="TB-001",
        category="table",
        image_path="/data/images/TB-001.jpg"
    )
    
    assert product.product_code == "TB-001"
    assert product.description_human is None
