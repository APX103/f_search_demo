"""混合搜索服务"""

import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional

from ..encoders.image_encoder import ImageEncoder
from ..encoders.text_encoder import TextEncoder
from ..generators.description import DescriptionGenerator
from ..storage.zilliz_client import ZillizClient
from ..fusion import rrf_fusion, FusionConfig


@dataclass
class SearchConfig:
    """搜索配置"""
    weight_image: float = 0.35
    weight_text_vector: float = 0.40
    weight_bm25: float = 0.25
    rrf_k: int = 60
    candidate_multiplier: int = 3
    category_boost: float = 1.2

    def to_fusion_config(self) -> "FusionConfig":
        return FusionConfig(
            weight_image=self.weight_image,
            weight_text_vector=self.weight_text_vector,
            weight_bm25=self.weight_bm25,
            rrf_k=self.rrf_k,
            category_boost=self.category_boost,
        )


class HybridSearchService:
    """三路混合搜索服务（图像向量 + 文本向量 + BM25，客户端 RRF 融合）"""

    OUTPUT_FIELDS = ["primary_key", "sku", "name", "category", "price", "discription", "LLMDescription", "url", "imageUrl"]

    def __init__(
        self,
        zilliz_client: ZillizClient,
        image_encoder: ImageEncoder,
        text_encoder: TextEncoder,
        description_generator: DescriptionGenerator,
        config: Optional[SearchConfig] = None,
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
        category_hint: Optional[str] = None,
    ) -> Tuple[List[dict], str]:
        # 1. 并行：图像编码 + 描述生成
        image_emb, description = await asyncio.gather(
            self.image_encoder.encode(image_bytes),
            self.desc_generator.generate(image_bytes),
        )

        # 2. 文本编码（依赖 VLM 描述）
        text_emb = await self.text_encoder.encode(description)

        # 3. 三路并行搜索
        candidate_limit = top_k * self.config.candidate_multiplier

        image_results, text_results, bm25_results = await asyncio.gather(
            self.zilliz.search(
                vector=image_emb.tolist(),
                anns_field="imageVector",
                limit=candidate_limit,
                output_fields=self.OUTPUT_FIELDS,
            ),
            self.zilliz.search(
                vector=text_emb.tolist(),
                anns_field="vector",
                limit=candidate_limit,
                output_fields=self.OUTPUT_FIELDS,
            ),
            self.zilliz.search_by_text(
                query_text=description,
                anns_field="LLMDescription",
                limit=candidate_limit,
                output_fields=self.OUTPUT_FIELDS,
            ),
        )

        # 4. 客户端 RRF 融合
        fused = rrf_fusion(
            image_results=image_results,
            text_results=text_results,
            bm25_results=bm25_results,
            config=self.config.to_fusion_config(),
            category_hint=category_hint,
        )

        return fused[:top_k], description
