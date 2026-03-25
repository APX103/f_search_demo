"""
家具搜索 HTTP 服务（供其他语言调用）
"""

import base64
import logging
import time
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .client import FurnitureSearchClient, SearchConfig, SearchResult
from .config import get_settings


# 请求/响应模型
class ImageSearchRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 编码的图片")
    top_k: int = Field(default=20, ge=1, le=100, description="返回结果数量")
    category_hint: Optional[str] = Field(default=None, description="品类提示")


class SearchResponseData(BaseModel):
    query_description: str
    results: list[SearchResult]


class SearchResponse(BaseModel):
    success: bool = True
    data: SearchResponseData
    meta: dict


# 全局客户端实例
_client: Optional[FurnitureSearchClient] = None


def get_client() -> FurnitureSearchClient:
    """获取客户端实例"""
    global _client
    if _client is None:
        settings = get_settings()
        _client = FurnitureSearchClient(
            zilliz_endpoint=settings.zilliz_cloud_uri,
            zilliz_token=settings.zilliz_cloud_token,
            aliyun_api_key=settings.aliyun_dashscope_api_key,
            zhipu_api_key=settings.zhipu_api_key,
            collection_name=settings.zilliz_cloud_collection,
            config=SearchConfig(
                rrf_k=settings.search_rrf_k,
                candidate_multiplier=settings.search_candidate_multiplier
            )
        )
    return _client


# 创建 FastAPI 应用
app = FastAPI(
    title="家具搜索服务 API",
    description="基于图像的家具相似搜索服务，支持其他语言后端调用",
    version="0.1.0"
)


@app.on_event("startup")
async def startup():
    """启动时初始化客户端"""
    get_client()


@app.on_event("shutdown")
async def shutdown():
    """关闭时清理资源"""
    global _client
    if _client:
        await _client.close()


@app.post("/search", response_model=SearchResponse)
async def search_by_image(request: ImageSearchRequest):
    """
    基于图片的家具相似搜索

    请求格式：
    ```json
    {
      "image_base64": "base64_encoded_image_data",
      "top_k": 20,
      "category_hint": "sofa"
    }
    ```

    响应格式：
    ```json
    {
      "success": true,
      "data": {
        "query_description": "...",
        "results": [...]
      },
      "meta": {
        "took_ms": 156
      }
    }
    ```
    """
    start_time = time.time()

    try:
        client = get_client()

        # 解码图片
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        # 执行搜索
        results, query_description = await client.search(
            image_bytes=image_bytes,
            top_k=request.top_k,
            category_hint=request.category_hint
        )

        # 构建响应
        elapsed_ms = int((time.time() - start_time) * 1000)

        return SearchResponse(
            success=True,
            data=SearchResponseData(
                query_description=query_description,
                results=results
            ),
            meta={
                "took_ms": elapsed_ms,
                "total_candidates": len(results)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "furniture-search-sdk"}


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "furniture_search.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
