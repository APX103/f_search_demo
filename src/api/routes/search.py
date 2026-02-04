# src/api/routes/search.py
"""搜索路由"""

import time
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Query, Depends, HTTPException

from src.api.deps import get_search_service
from src.search.hybrid_search import HybridSearchService
from src.models.schemas import SearchResponse, SearchData, SearchMeta, SearchResult


router = APIRouter()


@router.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    image: UploadFile = File(..., description="输入图片"),
    top_k: int = Query(default=20, ge=1, le=100, description="返回结果数量"),
    category_hint: Optional[str] = Query(default=None, description="品类提示"),
    search_service: HybridSearchService = Depends(get_search_service)
) -> SearchResponse:
    """
    基于图片的家具相似搜索
    
    上传一张家具图片（如 AI 生成的设计图），返回最相似的商品列表。
    """
    start_time = time.time()
    
    # 读取图片数据
    image_bytes = await image.read()
    
    # 验证图片
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")
    
    # 执行搜索
    try:
        results, query_description = await search_service.search(
            image_bytes=image_bytes,
            top_k=top_k,
            category_hint=category_hint
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    # 构建响应
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    return SearchResponse(
        success=True,
        data=SearchData(
            query_description=query_description,
            results=[
                SearchResult(
                    product_id=r["id"],
                    sku=r.get("sku", ""),
                    name=r.get("name", ""),
                    category=r.get("category", ""),
                    price=r.get("price", ""),
                    description=r.get("description", ""),
                    llm_description=r.get("LLMDescription", ""),
                    url=r.get("url", ""),
                    image_url=r.get("imageUrl", ""),
                    score=r["score"],
                    rank=r.get("rank", 0)
                )
                for r in results
            ]
        ),
        meta=SearchMeta(
            took_ms=elapsed_ms,
            total_candidates=len(results)
        )
    )
