# tests/test_fusion.py
"""RRF 融合算法测试"""

import pytest


def test_rrf_fusion_basic():
    """测试基础 RRF 融合"""
    from src.search.fusion import rrf_fusion, FusionConfig
    
    image_results = [
        {"id": 1, "rank": 1, "product_code": "A"},
        {"id": 2, "rank": 2, "product_code": "B"},
        {"id": 3, "rank": 3, "product_code": "C"},
    ]
    text_results = [
        {"id": 2, "rank": 1, "product_code": "B"},
        {"id": 1, "rank": 2, "product_code": "A"},
        {"id": 4, "rank": 3, "product_code": "D"},
    ]
    bm25_results = [
        {"id": 1, "rank": 1, "product_code": "A"},
        {"id": 4, "rank": 2, "product_code": "D"},
        {"id": 5, "rank": 3, "product_code": "E"},
    ]
    
    config = FusionConfig(
        weight_image=0.35,
        weight_text_vector=0.40,
        weight_bm25=0.25,
        rrf_k=60
    )
    
    fused = rrf_fusion(
        image_results=image_results,
        text_results=text_results,
        bm25_results=bm25_results,
        config=config
    )
    
    # ID 1 出现在三路搜索中，应该排名靠前
    assert fused[0]["id"] == 1
    # 结果应该按分数降序排列
    scores = [r["score"] for r in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fusion_empty_results():
    """测试空结果处理"""
    from src.search.fusion import rrf_fusion, FusionConfig
    
    config = FusionConfig()
    
    fused = rrf_fusion(
        image_results=[],
        text_results=[],
        bm25_results=[],
        config=config
    )
    
    assert fused == []


def test_rrf_fusion_with_category_boost():
    """测试 category 加权"""
    from src.search.fusion import rrf_fusion, FusionConfig
    
    image_results = [
        {"id": 1, "rank": 1, "product_code": "A", "category": "sofa"},
        {"id": 2, "rank": 2, "product_code": "B", "category": "table"},
    ]
    text_results = [
        {"id": 2, "rank": 1, "product_code": "B", "category": "table"},
        {"id": 1, "rank": 2, "product_code": "A", "category": "sofa"},
    ]
    bm25_results = []
    
    config = FusionConfig(
        weight_image=0.5,
        weight_text_vector=0.5,
        weight_bm25=0.0,
        rrf_k=60,
        category_boost=1.5
    )
    
    # 无 category_hint 时，分数相同
    fused_no_hint = rrf_fusion(
        image_results=image_results,
        text_results=text_results,
        bm25_results=bm25_results,
        config=config
    )
    
    # 有 category_hint 时，匹配的 category 加权
    fused_with_hint = rrf_fusion(
        image_results=image_results,
        text_results=text_results,
        bm25_results=bm25_results,
        config=config,
        category_hint="sofa"
    )
    
    # sofa (id=1) 应该因为 category_boost 而排名更高
    sofa_rank_no_hint = next(i for i, r in enumerate(fused_no_hint) if r["id"] == 1)
    sofa_rank_with_hint = next(i for i, r in enumerate(fused_with_hint) if r["id"] == 1)
    
    assert sofa_rank_with_hint <= sofa_rank_no_hint
