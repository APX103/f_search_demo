# src/models/schemas.py
"""Pydantic 数据模型定义"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """搜索结果（与 Zilliz Cloud schema 对齐）"""
    product_id: int
    sku: str
    name: str
    category: str
    price: str
    description: str
    llm_description: str
    url: str
    image_url: str
    score: float
    rank: int


class SearchRequest(BaseModel):
    """搜索请求参数"""
    top_k: int = Field(default=20, ge=1, le=100)
    category_hint: Optional[str] = None


class SearchResponse(BaseModel):
    """搜索响应"""
    success: bool = True
    data: "SearchData"
    meta: "SearchMeta"


class SearchData(BaseModel):
    """搜索数据"""
    query_description: str
    results: List[SearchResult]


class SearchMeta(BaseModel):
    """搜索元数据"""
    took_ms: int
    total_candidates: int


class ProductInput(BaseModel):
    """商品输入（用于批量导入）"""
    sku: str
    name: str
    category: str
    price: str
    description: str
    url: str
    image_url: str
    image_path: Optional[str] = None  # 本地图片路径，用于生成向量


class ProductResponse(BaseModel):
    """商品创建响应"""
    success: bool = True
    data: "ProductData"


class ProductData(BaseModel):
    """商品数据"""
    product_id: int
    description_ai: str


class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = False
    error_code: int
    message: str


# 解决循环引用
SearchResponse.model_rebuild()
ProductResponse.model_rebuild()
