# src/search/fusion.py
"""RRF (Reciprocal Rank Fusion) 融合算法"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class FusionConfig:
    """融合配置"""
    weight_image: float = 0.35
    weight_text_vector: float = 0.40
    weight_bm25: float = 0.25
    rrf_k: int = 60
    category_boost: float = 1.2


def rrf_fusion(
    image_results: List[Dict],
    text_results: List[Dict],
    bm25_results: List[Dict],
    config: FusionConfig,
    category_hint: Optional[str] = None
) -> List[Dict]:
    """
    RRF (Reciprocal Rank Fusion) 融合三路搜索结果
    
    RRF 公式: score = sum(weight_i / (k + rank_i))
    
    Args:
        image_results: 图像向量搜索结果，每项需包含 id, rank
        text_results: 文本向量搜索结果
        bm25_results: BM25 全文搜索结果
        config: 融合配置
        category_hint: 可选的品类提示，用于软过滤加权
    
    Returns:
        融合后的结果列表，按分数降序排列
    """
    if not image_results and not text_results and not bm25_results:
        return []
    
    # 建立 ID -> 排名 的映射
    image_ranks = {r["id"]: r["rank"] for r in image_results}
    text_ranks = {r["id"]: r["rank"] for r in text_results}
    bm25_ranks = {r["id"]: r["rank"] for r in bm25_results}
    
    # 收集所有候选 ID 及其元数据
    all_candidates: Dict[int, Dict] = {}
    for results in [image_results, text_results, bm25_results]:
        for r in results:
            if r["id"] not in all_candidates:
                all_candidates[r["id"]] = r.copy()
    
    # 计算 RRF 分数
    k = config.rrf_k
    fused_results = []
    
    for product_id, meta in all_candidates.items():
        score = 0.0
        debug_ranks = {}
        
        # 图像搜索贡献
        if product_id in image_ranks:
            rank = image_ranks[product_id]
            debug_ranks["image"] = rank
            score += config.weight_image / (k + rank)
        
        # 文本向量搜索贡献
        if product_id in text_ranks:
            rank = text_ranks[product_id]
            debug_ranks["text_vector"] = rank
            score += config.weight_text_vector / (k + rank)
        
        # BM25 搜索贡献
        if product_id in bm25_ranks:
            rank = bm25_ranks[product_id]
            debug_ranks["bm25"] = rank
            score += config.weight_bm25 / (k + rank)
        
        # Category 软过滤加权
        if category_hint and meta.get("category"):
            if _category_match(category_hint, meta["category"]):
                score *= config.category_boost
        
        result = meta.copy()
        result["score"] = score
        result["debug_ranks"] = debug_ranks
        fused_results.append(result)
    
    # 按分数降序排序
    fused_results.sort(key=lambda x: x["score"], reverse=True)
    
    # 添加最终排名
    for i, result in enumerate(fused_results, 1):
        result["final_rank"] = i
    
    return fused_results


def _category_match(hint: str, actual: str) -> bool:
    """
    判断 category 是否匹配（支持模糊匹配）
    
    Args:
        hint: 用户提供的品类提示
        actual: 商品实际品类
    
    Returns:
        是否匹配
    """
    hint_lower = hint.lower().strip()
    actual_lower = actual.lower().strip()
    return hint_lower in actual_lower or actual_lower in hint_lower
