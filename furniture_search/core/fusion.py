# furniture_search/core/fusion.py
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
    if not image_results and not text_results and not bm25_results:
        return []
    
    image_ranks = {r["id"]: r["rank"] for r in image_results}
    text_ranks = {r["id"]: r["rank"] for r in text_results}
    bm25_ranks = {r["id"]: r["rank"] for r in bm25_results}
    
    all_candidates: Dict[int, Dict] = {}
    for results in [image_results, text_results, bm25_results]:
        for r in results:
            if r["id"] not in all_candidates:
                all_candidates[r["id"]] = r.copy()
    
    k = config.rrf_k
    fused_results = []
    
    for product_id, meta in all_candidates.items():
        score = 0.0
        debug_ranks = {}
        
        if product_id in image_ranks:
            rank = image_ranks[product_id]
            debug_ranks["image"] = rank
            score += config.weight_image / (k + rank)
        
        if product_id in text_ranks:
            rank = text_ranks[product_id]
            debug_ranks["text_vector"] = rank
            score += config.weight_text_vector / (k + rank)
        
        if product_id in bm25_ranks:
            rank = bm25_ranks[product_id]
            debug_ranks["bm25"] = rank
            score += config.weight_bm25 / (k + rank)
        
        if category_hint and meta.get("category"):
            if _category_match(category_hint, meta["category"]):
                score *= config.category_boost
        
        result = meta.copy()
        result["score"] = score
        result["debug_ranks"] = debug_ranks
        fused_results.append(result)
    
    fused_results.sort(key=lambda x: x["score"], reverse=True)
    
    for i, result in enumerate(fused_results, 1):
        result["final_rank"] = i
    
    return fused_results


def _category_match(hint: str, actual: str) -> bool:
    hint_lower = hint.lower().strip()
    actual_lower = actual.lower().strip()
    return hint_lower in actual_lower or actual_lower in hint_lower
