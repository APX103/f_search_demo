# src/config.py
"""配置管理模块"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Zilliz Cloud (Milvus)
    zilliz_cloud_uri: str
    zilliz_cloud_token: str
    zilliz_cloud_collection: str = "products"
    
    # 智谱 AI
    zhipu_api_key: str
    
    # 阿里云
    aliyun_dashscope_api_key: str
    
    # 搜索配置（使用 Zilliz Cloud 原生 hybrid_search + RRF）
    search_rrf_k: int = 60
    search_candidate_multiplier: int = 3


@lru_cache
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
