# src/search/hybrid_search.py
"""混合搜索服务"""

import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional

from src.encoders.image_encoder import ImageEncoder
from src.encoders.text_encoder import TextEncoder
from src.generators.description import DescriptionGenerator


@dataclass
class SearchConfig:
    """搜索配置"""
    rrf_k: int = 60
    candidate_multiplier: int = 3


class HybridSearchService:
    """三路混合搜索服务（使用 Zilliz Cloud 原生 hybrid_search）"""
    
    # 输出字段（与 Zilliz Cloud schema 对齐）
    OUTPUT_FIELDS = ["sku", "name", "category", "price", "description", "LLMDescription", "url", "imageUrl"]
    
    def __init__(
        self,
        milvus_client,
        image_encoder: ImageEncoder,
        text_encoder: TextEncoder,
        description_generator: DescriptionGenerator,
        config: Optional[SearchConfig] = None
    ):
        self.milvus = milvus_client
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.desc_generator = description_generator
        self.config = config or SearchConfig()
    
    async def search(
        self,
        image_bytes: bytes,
        top_k: int = 20,
        category_hint: Optional[str] = None
    ) -> Tuple[List[dict], str]:
        """
        执行三路混合搜索（使用 Zilliz Cloud 原生 hybrid_search + RRF）
        
        Args:
            image_bytes: 输入图片的二进制数据
            top_k: 返回结果数量
            category_hint: 可选的品类提示（暂未使用，可扩展为 filter）
        
        Returns:
            (搜索结果列表, 生成的描述)
        """
        # 1. 并行处理：图像编码 + 描述生成
        image_emb_task = self.image_encoder.encode(image_bytes)
        description_task = self.desc_generator.generate(image_bytes)
        
        image_emb, description = await asyncio.gather(
            image_emb_task, description_task
        )
        
        # 2. 文本编码（依赖描述生成结果）
        text_emb = await self.text_encoder.encode(description)
        
        # 3. 构建三路搜索请求（字段名与 Zilliz Cloud schema 对齐）
        candidate_limit = top_k * self.config.candidate_multiplier
        
        search_requests = [
            {
                "field_name": "imageVector",  # 图像向量 (1024维)
                "data": image_emb.tolist(),
                "search_type": "vector",
                "limit": candidate_limit
            },
            {
                "field_name": "vector",  # 文本向量 (2048维)
                "data": text_emb.tolist(),
                "search_type": "vector",
                "limit": candidate_limit
            },
            {
                "field_name": "LLMDescription",  # BM25 全文搜索
                "data": description,
                "search_type": "text",
                "limit": candidate_limit
            }
        ]
        
        # 4. 执行原生 hybrid_search（一次 API 调用，服务端 RRF 融合）
        results = self.milvus.hybrid_search(
            search_requests=search_requests,
            output_fields=self.OUTPUT_FIELDS,
            limit=top_k,
            rrf_k=self.config.rrf_k
        )
        
        return results, description
