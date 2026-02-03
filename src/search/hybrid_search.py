# src/search/hybrid_search.py
"""混合搜索服务"""

import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional

from src.encoders.image_encoder import ImageEncoder
from src.encoders.text_encoder import TextEncoder
from src.generators.description import DescriptionGenerator
from src.search.fusion import rrf_fusion, FusionConfig


@dataclass
class SearchConfig:
    """搜索配置"""
    weight_image: float = 0.35
    weight_text_vector: float = 0.40
    weight_bm25: float = 0.25
    rrf_k: int = 60
    candidate_multiplier: int = 3
    category_boost: float = 1.2
    
    def to_fusion_config(self) -> FusionConfig:
        """转换为融合配置"""
        return FusionConfig(
            weight_image=self.weight_image,
            weight_text_vector=self.weight_text_vector,
            weight_bm25=self.weight_bm25,
            rrf_k=self.rrf_k,
            category_boost=self.category_boost
        )


class HybridSearchService:
    """三路混合搜索服务"""
    
    OUTPUT_FIELDS = ["product_code", "category", "description_ai", "image_url"]
    
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
        执行三路混合搜索
        
        Args:
            image_bytes: 输入图片的二进制数据
            top_k: 返回结果数量
            category_hint: 可选的品类提示
        
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
        
        # 3. 并行执行三路搜索
        candidate_limit = top_k * self.config.candidate_multiplier
        
        image_task = self._search_by_image(image_emb, candidate_limit)
        text_task = self._search_by_text_vector(text_emb, candidate_limit)
        bm25_task = self._search_by_bm25(description, candidate_limit)
        
        image_results, text_results, bm25_results = await asyncio.gather(
            image_task, text_task, bm25_task
        )
        
        # 4. RRF 融合
        fusion_config = self.config.to_fusion_config()
        fused_results = rrf_fusion(
            image_results=image_results,
            text_results=text_results,
            bm25_results=bm25_results,
            config=fusion_config,
            category_hint=category_hint
        )
        
        # 5. 返回 Top K
        return fused_results[:top_k], description
    
    async def _search_by_image(self, embedding, limit: int) -> List[dict]:
        """图像向量搜索"""
        return self.milvus.search_by_vector(
            field_name="image_embedding",
            vector=embedding.tolist(),
            limit=limit,
            output_fields=self.OUTPUT_FIELDS
        )
    
    async def _search_by_text_vector(self, embedding, limit: int) -> List[dict]:
        """文本向量搜索"""
        return self.milvus.search_by_vector(
            field_name="text_embedding",
            vector=embedding.tolist(),
            limit=limit,
            output_fields=self.OUTPUT_FIELDS
        )
    
    async def _search_by_bm25(self, query_text: str, limit: int) -> List[dict]:
        """BM25 全文搜索"""
        return self.milvus.search_by_text(
            field_name="description_ai",
            query_text=query_text,
            limit=limit,
            output_fields=self.OUTPUT_FIELDS
        )
