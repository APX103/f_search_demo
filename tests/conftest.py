# tests/conftest.py
"""pytest 配置和 fixtures"""

import os
import sys
import pytest

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """为所有测试设置测试环境变量"""
    monkeypatch.setenv("ZILLIZ_CLOUD_URI", "https://test.zillizcloud.com")
    monkeypatch.setenv("ZILLIZ_CLOUD_TOKEN", "test_token")
    monkeypatch.setenv("ZILLIZ_CLOUD_COLLECTION", "test_products")
    monkeypatch.setenv("ZHIPU_API_KEY", "test_zhipu_key")
    monkeypatch.setenv("ALIYUN_DASHSCOPE_API_KEY", "test_aliyun_key")


@pytest.fixture
def sample_image_bytes():
    """生成测试用的最小有效 JPEG"""
    # 最小有效 JPEG 头部
    return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
