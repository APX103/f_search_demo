# src/search/hybrid_search.py
"""混合搜索服务"""

import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional

from src.encoders.image_encoder import ImageEncoder
from src.encoders.text_encoder import TextEncoder
from src.generators.description import DescriptionGenerator
from src.storage.zilliz_client import ZillizClient


@dataclass
class SearchConfig:
    """搜索配置"""
    rrf_k: int = 60
    candidate_multiplier: int = 3


class HybridSearchService:
    """双路混合搜索服务（图像向量 + 文本向量，使用 Zilliz Cloud REST API）"""
    
    # 输出字段（与 Zilliz Cloud schema 对齐）
    OUTPUT_FIELDS = ["primary_key", "sku", "name", "category", "price", "discription", "LLMDescription", "url", "imageUrl"]
    
    def __init__(
        self,
        zilliz_client: ZillizClient,
        image_encoder: ImageEncoder,
        text_encoder: TextEncoder,
        description_generator: DescriptionGenerator,
        config: Optional[SearchConfig] = None
    ):
        self.zilliz = zilliz_client
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
        执行双路混合搜索（图像向量 + 文本向量 + RRF 融合）
        
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
        
        # 3. 构建双路搜索请求
        candidate_limit = top_k * self.config.candidate_multiplier
        
        vector_searches = [
            {
                "data": text_emb.tolist(),
                "anns_field": "vector",  # 文本向量 (2048维)
                "limit": candidate_limit
            },
            {
                "data": image_emb.tolist(),
                "anns_field": "imageVector",  # 图像向量 (1024维)
                "limit": candidate_limit
            }
        ]
        
        # 4. 执行 hybrid_search（REST API，服务端 RRF 融合）
        results = await self.zilliz.hybrid_search(
            vector_searches=vector_searches,
            limit=top_k,
            rrf_k=self.config.rrf_k,
            output_fields=self.OUTPUT_FIELDS
        )
        
        return results, description
