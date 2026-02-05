# src/api/deps.py
"""FastAPI 依赖注入"""

from typing import Optional

from src.config import get_settings, Settings
from src.storage.zilliz_client import ZillizClient
from src.encoders.image_encoder import AliyunImageEncoder
from src.encoders.text_encoder import ZhipuTextEncoder
from src.generators.description import ZhipuDescriptionGenerator
from src.search.hybrid_search import HybridSearchService, SearchConfig


# 全局服务实例
_search_service: Optional[HybridSearchService] = None
_zilliz_client: Optional[ZillizClient] = None
_image_encoder: Optional[AliyunImageEncoder] = None
_text_encoder: Optional[ZhipuTextEncoder] = None
_desc_generator: Optional[ZhipuDescriptionGenerator] = None


async def init_services() -> None:
    """初始化所有服务"""
    global _search_service, _zilliz_client
    global _image_encoder, _text_encoder, _desc_generator
    
    settings = get_settings()
    
    # 初始化 Zilliz 客户端
    _zilliz_client = ZillizClient(
        endpoint=settings.zilliz_cloud_uri,
        token=settings.zilliz_cloud_token,
        collection_name=settings.zilliz_cloud_collection
    )
    
    # 初始化编码器和生成器
    _image_encoder = AliyunImageEncoder(api_key=settings.aliyun_dashscope_api_key)
    _text_encoder = ZhipuTextEncoder(api_key=settings.zhipu_api_key)
    _desc_generator = ZhipuDescriptionGenerator(api_key=settings.zhipu_api_key)
    
    # 初始化搜索服务
    search_config = SearchConfig(
        rrf_k=settings.search_rrf_k,
        candidate_multiplier=settings.search_candidate_multiplier
    )
    
    _search_service = HybridSearchService(
        zilliz_client=_zilliz_client,
        image_encoder=_image_encoder,
        text_encoder=_text_encoder,
        description_generator=_desc_generator,
        config=search_config
    )


async def cleanup_services() -> None:
    """清理所有服务资源"""
    global _image_encoder, _text_encoder, _desc_generator
    
    if _image_encoder:
        await _image_encoder.close()
    if _text_encoder:
        await _text_encoder.close()
    if _desc_generator:
        await _desc_generator.close()


def get_search_service() -> HybridSearchService:
    """获取搜索服务（FastAPI 依赖）"""
    if _search_service is None:
        raise RuntimeError("Services not initialized. Call init_services() first.")
    return _search_service


def get_zilliz_client() -> ZillizClient:
    """获取 Zilliz 客户端（FastAPI 依赖）"""
    if _zilliz_client is None:
        raise RuntimeError("Services not initialized. Call init_services() first.")
    return _zilliz_client
