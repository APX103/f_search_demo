# tests/test_schemas.py
"""数据模型测试"""

import pytest
from pydantic import ValidationError


def test_search_result_model():
    """测试搜索结果模型"""
    from src.models.schemas import SearchResult
    
    result = SearchResult(
        product_id=123,
        sku="SF-001",
        name="Modern Sofa",
        category="sofa",
        price="$999",
        description="A comfortable sofa",
        llm_description="[COLOR] Dark gray...",
        url="https://example.com/product/SF-001",
        image_url="https://example.com/image.jpg",
        score=0.85,
        rank=1
    )
    
    assert result.product_id == 123
    assert result.sku == "SF-001"
    assert result.name == "Modern Sofa"
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
        sku="TB-001",
        name="Dining Table",
        category="table",
        price="$599",
        description="Oak dining table",
        url="https://example.com/product/TB-001",
        image_url="https://example.com/images/TB-001.jpg"
    )
    
    assert product.sku == "TB-001"
    assert product.name == "Dining Table"
    assert product.image_path is None
