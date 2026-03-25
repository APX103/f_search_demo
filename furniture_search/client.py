"""简化的客户端接口"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import asyncio

from .config import Settings
from .core.hybrid_search import HybridSearchService, SearchConfig
from .core.encoders.image_encoder import AliyunImageEncoder
from .core.encoders.text_encoder import ZhipuTextEncoder
from .core.generators.description import ZhipuDescriptionGenerator
from .core.storage.zilliz_client import ZillizClient


@dataclass
class SearchResult:
    """搜索结果"""
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


class FurnitureSearchClient:
    """家具搜索客户端（简化版）"""

    def __init__(
        self,
        zilliz_endpoint: str,
        zilliz_token: str,
        aliyun_api_key: str,
        zhipu_api_key: str,
        collection_name: str = "furniture_products",
        config: Optional[SearchConfig] = None
    ):
        """
        初始化客户端

        Args:
            zilliz_endpoint: Zilliz Cloud endpoint
            zilliz_token: Zilliz API token
            aliyun_api_key: 阿里云 API key
            zhipu_api_key: 智谱 API key
            collection_name: 集合名称
            config: 搜索配置
        """
        self.config = config or SearchConfig()
        self._initialized = False
        self._service: Optional[HybridSearchService] = None
        self._zilliz_client: Optional[ZillizClient] = None
        self._encoders: List = []

        # 存储配置
        self.zilliz_endpoint = zilliz_endpoint
        self.zilliz_token = zilliz_token
        self.collection_name = collection_name
        self.aliyun_api_key = aliyun_api_key
        self.zhipu_api_key = zhipu_api_key

    async def _init(self):
        """延迟初始化服务"""
        if self._initialized:
            return

        # 初始化 Zilliz 客户端
        zilliz_client = ZillizClient(
            endpoint=self.zilliz_endpoint,
            token=self.zilliz_token,
            collection_name=self.collection_name
        )
        self._zilliz_client = zilliz_client

        # 初始化编码器
        image_encoder = AliyunImageEncoder(api_key=self.aliyun_api_key)
        text_encoder = ZhipuTextEncoder(api_key=self.zhipu_api_key)
        desc_generator = ZhipuDescriptionGenerator(api_key=self.zhipu_api_key)

        self._encoders = [image_encoder, text_encoder, desc_generator]

        # 初始化搜索服务
        self._service = HybridSearchService(
            zilliz_client=zilliz_client,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            description_generator=desc_generator,
            config=self.config
        )

        self._initialized = True

    async def search(
        self,
        image_bytes: bytes,
        top_k: int = 20,
        category_hint: Optional[str] = None
    ) -> tuple[List[SearchResult], str]:
        """
        执行图像搜索

        Args:
            image_bytes: 图片二进制数据
            top_k: 返回结果数量
            category_hint: 品类提示（可选）

        Returns:
            (搜索结果列表, 查询描述)
        """
        await self._init()

        results, description = await self._service.search(
            image_bytes=image_bytes,
            top_k=top_k,
            category_hint=category_hint
        )

        return [
            SearchResult(
                product_id=r.get("primary_key") or r.get("id", 0),
                sku=r.get("sku", ""),
                name=r.get("name", ""),
                category=r.get("category", ""),
                price=r.get("price", ""),
                description=r.get("description", ""),
                llm_description=r.get("LLMDescription", ""),
                url=r.get("url", ""),
                image_url=r.get("imageUrl", ""),
                score=r.get("score", 0),
                rank=r.get("rank", 0)
            )
            for r in results
        ], description

    async def close(self):
        """关闭客户端，释放资源"""
        if self._zilliz_client:
            await self._zilliz_client.close()
        for encoder in self._encoders:
            if hasattr(encoder, 'close'):
                await encoder.close()

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
