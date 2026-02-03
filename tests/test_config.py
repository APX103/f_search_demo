# tests/test_config.py
"""配置管理测试"""

import os
import pytest


def test_settings_loads_from_env(monkeypatch):
    """测试配置从环境变量加载"""
    monkeypatch.setenv("ZILLIZ_CLOUD_URI", "https://test.zillizcloud.com")
    monkeypatch.setenv("ZILLIZ_CLOUD_TOKEN", "test_token")
    monkeypatch.setenv("ZILLIZ_CLOUD_COLLECTION", "test_products")
    monkeypatch.setenv("ZHIPU_API_KEY", "zhipu_key")
    monkeypatch.setenv("ALIYUN_DASHSCOPE_API_KEY", "aliyun_key")
    
    # 重新导入以获取新的环境变量
    from src.config import Settings
    settings = Settings()
    
    assert settings.zilliz_cloud_uri == "https://test.zillizcloud.com"
    assert settings.zilliz_cloud_token == "test_token"
    assert settings.zilliz_cloud_collection == "test_products"
    assert settings.zhipu_api_key == "zhipu_key"
    assert settings.aliyun_dashscope_api_key == "aliyun_key"


def test_settings_has_default_collection_name(monkeypatch):
    """测试 collection 名称有默认值"""
    monkeypatch.setenv("ZILLIZ_CLOUD_URI", "https://test.zillizcloud.com")
    monkeypatch.setenv("ZILLIZ_CLOUD_TOKEN", "test_token")
    monkeypatch.setenv("ZHIPU_API_KEY", "zhipu_key")
    monkeypatch.setenv("ALIYUN_DASHSCOPE_API_KEY", "aliyun_key")
    # 不设置 ZILLIZ_CLOUD_COLLECTION
    monkeypatch.delenv("ZILLIZ_CLOUD_COLLECTION", raising=False)
    
    from src.config import Settings
    settings = Settings()
    
    assert settings.zilliz_cloud_collection == "products"
