"""配置管理"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """SDK 配置"""

    # Zilliz Cloud
    zilliz_cloud_uri: str = ""
    zilliz_cloud_token: str = ""
    zilliz_cloud_collection: str = "furniture_products"

    # AI 服务
    aliyun_dashscope_api_key: str = ""
    zhipu_api_key: str = ""

    # 搜索参数
    search_rrf_k: int = 60
    search_candidate_multiplier: int = 3

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """获取配置（单例）"""
    return Settings()
