"""家具搜索 SDK - 简化版

支持模式：
1. Python 项目直接使用：from furniture_search import FurnitureSearchClient
2. HTTP 服务模式：启动 HTTP 服务供其他语言调用
"""

from .client import FurnitureSearchClient, SearchConfig
from .config import get_settings

__version__ = "0.1.0"
__all__ = ["FurnitureSearchClient", "SearchConfig", "get_settings"]
