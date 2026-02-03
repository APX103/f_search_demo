# 家具混合搜索系统实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现一个基于三路混合搜索（图像向量、文本向量、BM25）的家具相似搜索 API 服务。

**Architecture:** 使用 FastAPI 构建 API 服务，集成阿里云 Multimodal-Embedding 做图像编码，智谱 GLM-4.6V-Flash 做描述生成，智谱 Embedding-3 做文本编码，Zilliz Cloud (Milvus) 做向量存储和检索，通过 RRF 算法融合三路搜索结果。

**Tech Stack:** Python 3.10+, FastAPI, pymilvus, httpx, numpy, Pillow, pydantic-settings

---

## Task 1: 项目初始化

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`

**Step 1: 创建 requirements.txt**

```txt
# Web Framework
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6

# Async HTTP Client
httpx>=0.26.0
aiofiles>=23.2.1

# Vector Database
pymilvus>=2.4.0

# Data Processing
numpy>=1.26.0
Pillow>=10.2.0
pandas>=2.2.0

# Configuration
python-dotenv>=1.0.0
pydantic>=2.6.0
pydantic-settings>=2.1.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0

# CLI
tqdm>=4.66.0
```

**Step 2: 创建 pyproject.toml**

```toml
[project]
name = "furniture-search"
version = "0.1.0"
description = "基于 AI 设计图的家具相似搜索服务"
requires-python = ">=3.10"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

**Step 3: 创建 .env.example**

```bash
# Zilliz Cloud (Milvus)
ZILLIZ_CLOUD_URI=https://xxx.api.gcp-us-west1.zillizcloud.com
ZILLIZ_CLOUD_TOKEN=your_api_token
ZILLIZ_CLOUD_COLLECTION=products

# 智谱 AI（描述生成 + 文本 Embedding）
ZHIPU_API_KEY=your_zhipu_api_key

# 阿里云（图像 Embedding）
ALIYUN_DASHSCOPE_API_KEY=your_aliyun_api_key
```

**Step 4: 创建 .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
.eggs/

# Environment
.env
.venv/
venv/

# IDE
.idea/
.vscode/
*.swp

# Testing
.pytest_cache/
.coverage
htmlcov/

# Data
data/images/
*.jpg
*.png
!data/sample/*.jpg
```

**Step 5: 创建项目目录结构**

Run:
```bash
mkdir -p src/{api/routes,encoders,generators,search,storage,models}
mkdir -p tests
mkdir -p scripts
mkdir -p data/sample
touch src/__init__.py src/api/__init__.py src/api/routes/__init__.py
touch src/encoders/__init__.py src/generators/__init__.py
touch src/search/__init__.py src/storage/__init__.py src/models/__init__.py
touch tests/__init__.py
```

**Step 6: 提交**

```bash
git init
git add .
git commit -m "chore: initialize project structure with dependencies"
```

---

## Task 2: 配置管理模块

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

**Step 1: 创建配置测试文件**

```python
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
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: 实现配置模块**

```python
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
    
    # 搜索配置
    search_weight_image: float = 0.35
    search_weight_text_vector: float = 0.40
    search_weight_bm25: float = 0.25
    search_rrf_k: int = 60
    search_candidate_multiplier: int = 3
    search_category_boost: float = 1.2


@lru_cache
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add configuration management with pydantic-settings"
```

---

## Task 3: Pydantic 数据模型

**Files:**
- Create: `src/models/schemas.py`
- Create: `tests/test_schemas.py`

**Step 1: 创建数据模型测试**

```python
# tests/test_schemas.py
"""数据模型测试"""

import pytest
from pydantic import ValidationError


def test_search_result_model():
    """测试搜索结果模型"""
    from src.models.schemas import SearchResult
    
    result = SearchResult(
        product_id=123,
        product_code="SF-001",
        category="sofa",
        description_ai="[COLOR] Dark gray...",
        image_url="https://example.com/image.jpg",
        score=0.85,
        rank=1
    )
    
    assert result.product_id == 123
    assert result.product_code == "SF-001"
    assert result.score == 0.85


def test_search_request_model():
    """测试搜索请求模型"""
    from src.models.schemas import SearchRequest
    
    request = SearchRequest(top_k=10, category_hint="sofa")
    
    assert request.top_k == 10
    assert request.category_hint == "sofa"


def test_search_request_default_values():
    """测试搜索请求默认值"""
    from src.models.schemas import SearchRequest
    
    request = SearchRequest()
    
    assert request.top_k == 20
    assert request.category_hint is None


def test_product_input_model():
    """测试商品输入模型"""
    from src.models.schemas import ProductInput
    
    product = ProductInput(
        product_code="TB-001",
        category="table",
        image_path="/data/images/TB-001.jpg"
    )
    
    assert product.product_code == "TB-001"
    assert product.description_human is None
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL with "ImportError"

**Step 3: 实现数据模型**

```python
# src/models/schemas.py
"""Pydantic 数据模型定义"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """搜索结果"""
    product_id: int
    product_code: str
    category: str
    description_ai: str
    image_url: str
    score: float
    rank: int
    debug_ranks: Optional[Dict[str, int]] = None


class SearchRequest(BaseModel):
    """搜索请求参数"""
    top_k: int = Field(default=20, ge=1, le=100)
    category_hint: Optional[str] = None


class SearchResponse(BaseModel):
    """搜索响应"""
    success: bool = True
    data: "SearchData"
    meta: "SearchMeta"


class SearchData(BaseModel):
    """搜索数据"""
    query_description: str
    results: List[SearchResult]


class SearchMeta(BaseModel):
    """搜索元数据"""
    took_ms: int
    total_candidates: int


class ProductInput(BaseModel):
    """商品输入"""
    product_code: str
    category: str
    image_path: str
    description_human: Optional[str] = None


class ProductResponse(BaseModel):
    """商品创建响应"""
    success: bool = True
    data: "ProductData"


class ProductData(BaseModel):
    """商品数据"""
    product_id: int
    description_ai: str


class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = False
    error_code: int
    message: str


# 解决循环引用
SearchResponse.model_rebuild()
ProductResponse.model_rebuild()
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/models/schemas.py tests/test_schemas.py
git commit -m "feat: add Pydantic data models for API request/response"
```

---

## Task 4: Milvus 客户端封装

**Files:**
- Create: `src/storage/milvus_client.py`
- Create: `tests/test_milvus_client.py`

**Step 1: 创建 Milvus 客户端测试**

```python
# tests/test_milvus_client.py
"""Milvus 客户端测试"""

import pytest
from unittest.mock import Mock, patch, MagicMock


def test_milvus_client_wrapper_init():
    """测试 MilvusClientWrapper 初始化"""
    from src.storage.milvus_client import MilvusClientWrapper
    
    with patch("src.storage.milvus_client.MilvusClient") as mock_client:
        wrapper = MilvusClientWrapper(
            uri="https://test.zillizcloud.com",
            token="test_token",
            collection_name="products"
        )
        
        mock_client.assert_called_once_with(
            uri="https://test.zillizcloud.com",
            token="test_token"
        )
        assert wrapper.collection_name == "products"


def test_milvus_client_wrapper_insert():
    """测试数据插入"""
    from src.storage.milvus_client import MilvusClientWrapper
    
    with patch("src.storage.milvus_client.MilvusClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        wrapper = MilvusClientWrapper(
            uri="https://test.zillizcloud.com",
            token="test_token",
            collection_name="products"
        )
        
        data = [{"product_code": "SF-001", "category": "sofa"}]
        wrapper.insert(data)
        
        mock_client.insert.assert_called_once_with(
            collection_name="products",
            data=data
        )


def test_milvus_client_wrapper_search():
    """测试向量搜索"""
    from src.storage.milvus_client import MilvusClientWrapper
    
    with patch("src.storage.milvus_client.MilvusClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.search.return_value = [[]]
        
        wrapper = MilvusClientWrapper(
            uri="https://test.zillizcloud.com",
            token="test_token",
            collection_name="products"
        )
        
        embedding = [0.1] * 1024
        results = wrapper.search_by_vector(
            field_name="image_embedding",
            vector=embedding,
            limit=20,
            output_fields=["product_code"]
        )
        
        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["collection_name"] == "products"
        assert call_kwargs["anns_field"] == "image_embedding"
        assert call_kwargs["limit"] == 20
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_milvus_client.py -v`
Expected: FAIL with "ImportError"

**Step 3: 实现 Milvus 客户端封装**

```python
# src/storage/milvus_client.py
"""Zilliz Cloud (Milvus) 客户端封装"""

from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient, DataType


class MilvusClientWrapper:
    """Milvus 客户端封装类"""
    
    def __init__(self, uri: str, token: str, collection_name: str):
        self.client = MilvusClient(uri=uri, token=token)
        self.collection_name = collection_name
    
    def has_collection(self) -> bool:
        """检查 collection 是否存在"""
        return self.client.has_collection(self.collection_name)
    
    def create_collection(self, schema, index_params) -> None:
        """创建 collection"""
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
    
    def insert(self, data: List[Dict[str, Any]]) -> None:
        """插入数据"""
        self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
    
    def search_by_vector(
        self,
        field_name: str,
        vector: List[float],
        limit: int,
        output_fields: List[str],
        search_params: Optional[Dict] = None
    ) -> List[Dict]:
        """向量搜索"""
        params = search_params or {"metric_type": "COSINE", "params": {"ef": 64}}
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field=field_name,
            limit=limit,
            output_fields=output_fields,
            search_params=params
        )
        
        return self._parse_results(results)
    
    def search_by_text(
        self,
        field_name: str,
        query_text: str,
        limit: int,
        output_fields: List[str]
    ) -> List[Dict]:
        """BM25 全文搜索"""
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_text],
            anns_field=field_name,
            limit=limit,
            output_fields=output_fields,
            search_params={"metric_type": "BM25"}
        )
        
        return self._parse_results(results)
    
    def _parse_results(self, results) -> List[Dict]:
        """解析搜索结果"""
        parsed = []
        for hits in results:
            for rank, hit in enumerate(hits, 1):
                parsed.append({
                    "id": hit["id"],
                    "rank": rank,
                    "score": hit["distance"],
                    **hit["entity"]
                })
        return parsed
    
    def close(self) -> None:
        """关闭连接"""
        self.client.close()


def get_collection_schema(client: MilvusClient):
    """获取 collection schema"""
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("product_code", DataType.VARCHAR, max_length=64)
    schema.add_field("category", DataType.VARCHAR, max_length=128)
    schema.add_field("description_human", DataType.VARCHAR, max_length=2048)
    schema.add_field("description_ai", DataType.VARCHAR, max_length=4096,
                     enable_analyzer=True, analyzer_params={"type": "chinese"})
    schema.add_field("image_embedding", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("text_embedding", DataType.FLOAT_VECTOR, dim=2048)
    schema.add_field("image_url", DataType.VARCHAR, max_length=512)
    schema.add_field("created_at", DataType.INT64)
    
    return schema


def get_index_params(client: MilvusClient):
    """获取索引参数"""
    index_params = client.prepare_index_params()
    
    index_params.add_index(
        field_name="image_embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256}
    )
    index_params.add_index(
        field_name="text_embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256}
    )
    index_params.add_index(
        field_name="description_ai",
        index_type="AUTOINDEX",
        index_name="description_ai_bm25"
    )
    
    return index_params
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_milvus_client.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/storage/milvus_client.py tests/test_milvus_client.py
git commit -m "feat: add Milvus client wrapper with vector and BM25 search"
```

---

## Task 5: 图像编码器（阿里云 Multimodal-Embedding）

**Files:**
- Create: `src/encoders/image_encoder.py`
- Create: `tests/test_image_encoder.py`

**Step 1: 创建图像编码器测试**

```python
# tests/test_image_encoder.py
"""图像编码器测试"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.mark.asyncio
async def test_aliyun_image_encoder_encode():
    """测试阿里云图像编码器"""
    from src.encoders.image_encoder import AliyunImageEncoder
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "output": {
            "embeddings": [
                {"embedding": [0.1] * 1024}
            ]
        }
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        encoder = AliyunImageEncoder(api_key="test_key")
        
        # 创建测试图片数据（最小有效 JPEG）
        image_bytes = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        embedding = await encoder.encode(image_bytes)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)
        mock_post.assert_called_once()


def test_image_encoder_protocol():
    """测试图像编码器遵循协议"""
    from src.encoders.image_encoder import ImageEncoder, AliyunImageEncoder
    
    # 验证 AliyunImageEncoder 实现了 ImageEncoder 协议
    encoder = AliyunImageEncoder(api_key="test_key")
    assert hasattr(encoder, "encode")
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_image_encoder.py -v`
Expected: FAIL with "ImportError"

**Step 3: 实现图像编码器**

```python
# src/encoders/image_encoder.py
"""图像编码器实现"""

from abc import ABC, abstractmethod
from typing import Protocol
import base64
import numpy as np
import httpx


class ImageEncoder(Protocol):
    """图像编码器协议"""
    
    async def encode(self, image_bytes: bytes) -> np.ndarray:
        """将图片编码为向量"""
        ...


class AliyunImageEncoder:
    """阿里云多模态向量服务 - 图像编码器"""
    
    ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
    
    def __init__(
        self,
        api_key: str,
        model: str = "multimodal-embedding-v1",
        timeout: float = 30.0
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端（懒加载）"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def encode(self, image_bytes: bytes) -> np.ndarray:
        """
        将图片编码为 1024 维向量
        
        Args:
            image_bytes: 图片二进制数据
        
        Returns:
            1024 维 numpy 数组
        """
        image_b64 = base64.b64encode(image_bytes).decode()
        
        client = await self._get_client()
        response = await client.post(
            self.ENDPOINT,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "input": {
                    "contents": [
                        {"image": f"data:image/jpeg;base64,{image_b64}"}
                    ]
                }
            }
        )
        
        response.raise_for_status()
        result = response.json()
        
        embedding = result["output"]["embeddings"][0]["embedding"]
        return np.array(embedding, dtype=np.float32)
    
    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_image_encoder.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/encoders/image_encoder.py tests/test_image_encoder.py
git commit -m "feat: add Aliyun Multimodal-Embedding image encoder"
```

---

## Task 6: 文本编码器（智谱 Embedding-3）

**Files:**
- Create: `src/encoders/text_encoder.py`
- Create: `tests/test_text_encoder.py`

**Step 1: 创建文本编码器测试**

```python
# tests/test_text_encoder.py
"""文本编码器测试"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.mark.asyncio
async def test_zhipu_text_encoder_encode():
    """测试智谱文本编码器"""
    from src.encoders.text_encoder import ZhipuTextEncoder
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1] * 2048}
        ]
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        encoder = ZhipuTextEncoder(api_key="test_key")
        
        text = "三人位布艺沙发，北欧风格"
        embedding = await encoder.encode(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (2048,)
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_zhipu_text_encoder_sends_correct_payload():
    """测试智谱文本编码器发送正确的请求"""
    from src.encoders.text_encoder import ZhipuTextEncoder
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1] * 2048}]
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        encoder = ZhipuTextEncoder(api_key="test_key")
        await encoder.encode("test text")
        
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "embedding-3"
        assert call_kwargs["json"]["input"] == "test text"
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_text_encoder.py -v`
Expected: FAIL with "ImportError"

**Step 3: 实现文本编码器**

```python
# src/encoders/text_encoder.py
"""文本编码器实现"""

from typing import Protocol
import numpy as np
import httpx


class TextEncoder(Protocol):
    """文本编码器协议"""
    
    async def encode(self, text: str) -> np.ndarray:
        """将文本编码为向量"""
        ...


class ZhipuTextEncoder:
    """智谱 Embedding-3 文本编码器"""
    
    ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    
    def __init__(
        self,
        api_key: str,
        model: str = "embedding-3",
        timeout: float = 30.0
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端（懒加载）"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def encode(self, text: str) -> np.ndarray:
        """
        将文本编码为 2048 维向量
        
        Args:
            text: 输入文本
        
        Returns:
            2048 维 numpy 数组
        """
        client = await self._get_client()
        response = await client.post(
            self.ENDPOINT,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "input": text
            }
        )
        
        response.raise_for_status()
        result = response.json()
        
        embedding = result["data"][0]["embedding"]
        return np.array(embedding, dtype=np.float32)
    
    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_text_encoder.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/encoders/text_encoder.py tests/test_text_encoder.py
git commit -m "feat: add Zhipu Embedding-3 text encoder"
```

---

## Task 7: 描述生成器（智谱 GLM-4.6V-Flash）

**Files:**
- Create: `src/generators/description.py`
- Create: `tests/test_description_generator.py`

**Step 1: 创建描述生成器测试**

```python
# tests/test_description_generator.py
"""描述生成器测试"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.mark.asyncio
async def test_zhipu_description_generator():
    """测试智谱描述生成器"""
    from src.generators.description import ZhipuDescriptionGenerator
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "[COLOR] Dark gray\n[MATERIAL] Fabric\n[STYLE] Modern"
                }
            }
        ]
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        generator = ZhipuDescriptionGenerator(api_key="test_key")
        
        image_bytes = b'\xff\xd8\xff\xe0\x00\x10JFIF'
        description = await generator.generate(image_bytes)
        
        assert "[COLOR]" in description
        assert "Dark gray" in description
        mock_post.assert_called_once()


def test_unified_description_prompt_content():
    """测试统一描述 prompt 包含必要标签"""
    from src.generators.description import UNIFIED_DESCRIPTION_PROMPT
    
    assert "[COLOR]" in UNIFIED_DESCRIPTION_PROMPT
    assert "[MATERIAL]" in UNIFIED_DESCRIPTION_PROMPT
    assert "[STYLE]" in UNIFIED_DESCRIPTION_PROMPT
    assert "[SHAPE]" in UNIFIED_DESCRIPTION_PROMPT
    assert "[SIZE]" in UNIFIED_DESCRIPTION_PROMPT
    assert "[SUMMARY]" in UNIFIED_DESCRIPTION_PROMPT


@pytest.mark.asyncio
async def test_generator_uses_correct_model():
    """测试生成器使用正确的模型"""
    from src.generators.description import ZhipuDescriptionGenerator
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "test"}}]
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        generator = ZhipuDescriptionGenerator(api_key="test_key")
        await generator.generate(b'\xff\xd8\xff\xe0')
        
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "glm-4v-flash"
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_description_generator.py -v`
Expected: FAIL with "ImportError"

**Step 3: 实现描述生成器**

```python
# src/generators/description.py
"""描述生成服务"""

from typing import Protocol
import base64
import httpx


UNIFIED_DESCRIPTION_PROMPT = """
Analyze this furniture image and generate a structured description.

Output format (use exactly these labels):
[COLOR] Primary and secondary colors (e.g., dark gray, oak with white accents)
[MATERIAL] Visible materials (e.g., fabric, leather, solid wood, metal frame, glass)
[STYLE] Design style (e.g., modern minimalist, Scandinavian, mid-century modern, industrial)
[SHAPE] Form and structure (e.g., L-shaped, round, with armrests, tapered legs, tufted)
[SIZE] Apparent scale (e.g., 3-seater, compact, oversized, slim profile)
[SUMMARY] One fluent sentence describing this furniture piece

Rules:
- Describe only what is visible, do not assume
- Use common, searchable terms
- Be consistent: same furniture should produce similar descriptions
- If uncertain, omit rather than guess
""".strip()


class DescriptionGenerator(Protocol):
    """描述生成器协议"""
    
    async def generate(self, image_bytes: bytes) -> str:
        """为图片生成结构化描述"""
        ...


class ZhipuDescriptionGenerator:
    """智谱 GLM-4.6V-Flash 描述生成器"""
    
    ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    
    def __init__(
        self,
        api_key: str,
        model: str = "glm-4v-flash",
        timeout: float = 30.0,
        max_tokens: int = 500
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端（懒加载）"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def generate(self, image_bytes: bytes) -> str:
        """
        为图片生成结构化描述
        
        Args:
            image_bytes: 图片二进制数据
        
        Returns:
            结构化描述文本
        """
        image_b64 = base64.b64encode(image_bytes).decode()
        
        client = await self._get_client()
        response = await client.post(
            self.ENDPOINT,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                            },
                            {"type": "text", "text": UNIFIED_DESCRIPTION_PROMPT}
                        ]
                    }
                ],
                "max_tokens": self.max_tokens
            }
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result["choices"][0]["message"]["content"]
    
    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_description_generator.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/generators/description.py tests/test_description_generator.py
git commit -m "feat: add Zhipu GLM-4.6V-Flash description generator"
```

---

## Task 8: RRF 融合算法

**Files:**
- Create: `src/search/fusion.py`
- Create: `tests/test_fusion.py`

**Step 1: 创建 RRF 融合测试**

```python
# tests/test_fusion.py
"""RRF 融合算法测试"""

import pytest


def test_rrf_fusion_basic():
    """测试基础 RRF 融合"""
    from src.search.fusion import rrf_fusion, FusionConfig
    
    image_results = [
        {"id": 1, "rank": 1, "product_code": "A"},
        {"id": 2, "rank": 2, "product_code": "B"},
        {"id": 3, "rank": 3, "product_code": "C"},
    ]
    text_results = [
        {"id": 2, "rank": 1, "product_code": "B"},
        {"id": 1, "rank": 2, "product_code": "A"},
        {"id": 4, "rank": 3, "product_code": "D"},
    ]
    bm25_results = [
        {"id": 1, "rank": 1, "product_code": "A"},
        {"id": 4, "rank": 2, "product_code": "D"},
        {"id": 5, "rank": 3, "product_code": "E"},
    ]
    
    config = FusionConfig(
        weight_image=0.35,
        weight_text_vector=0.40,
        weight_bm25=0.25,
        rrf_k=60
    )
    
    fused = rrf_fusion(
        image_results=image_results,
        text_results=text_results,
        bm25_results=bm25_results,
        config=config
    )
    
    # ID 1 出现在三路搜索中，应该排名靠前
    assert fused[0]["id"] == 1
    # 结果应该按分数降序排列
    scores = [r["score"] for r in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fusion_empty_results():
    """测试空结果处理"""
    from src.search.fusion import rrf_fusion, FusionConfig
    
    config = FusionConfig()
    
    fused = rrf_fusion(
        image_results=[],
        text_results=[],
        bm25_results=[],
        config=config
    )
    
    assert fused == []


def test_rrf_fusion_with_category_boost():
    """测试 category 加权"""
    from src.search.fusion import rrf_fusion, FusionConfig
    
    image_results = [
        {"id": 1, "rank": 1, "product_code": "A", "category": "sofa"},
        {"id": 2, "rank": 2, "product_code": "B", "category": "table"},
    ]
    text_results = [
        {"id": 2, "rank": 1, "product_code": "B", "category": "table"},
        {"id": 1, "rank": 2, "product_code": "A", "category": "sofa"},
    ]
    bm25_results = []
    
    config = FusionConfig(
        weight_image=0.5,
        weight_text_vector=0.5,
        weight_bm25=0.0,
        rrf_k=60,
        category_boost=1.5
    )
    
    # 无 category_hint 时，分数相同
    fused_no_hint = rrf_fusion(
        image_results=image_results,
        text_results=text_results,
        bm25_results=bm25_results,
        config=config
    )
    
    # 有 category_hint 时，匹配的 category 加权
    fused_with_hint = rrf_fusion(
        image_results=image_results,
        text_results=text_results,
        bm25_results=bm25_results,
        config=config,
        category_hint="sofa"
    )
    
    # sofa (id=1) 应该因为 category_boost 而排名更高
    sofa_rank_no_hint = next(i for i, r in enumerate(fused_no_hint) if r["id"] == 1)
    sofa_rank_with_hint = next(i for i, r in enumerate(fused_with_hint) if r["id"] == 1)
    
    assert sofa_rank_with_hint <= sofa_rank_no_hint
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_fusion.py -v`
Expected: FAIL with "ImportError"

**Step 3: 实现 RRF 融合算法**

```python
# src/search/fusion.py
"""RRF (Reciprocal Rank Fusion) 融合算法"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class FusionConfig:
    """融合配置"""
    weight_image: float = 0.35
    weight_text_vector: float = 0.40
    weight_bm25: float = 0.25
    rrf_k: int = 60
    category_boost: float = 1.2


def rrf_fusion(
    image_results: List[Dict],
    text_results: List[Dict],
    bm25_results: List[Dict],
    config: FusionConfig,
    category_hint: Optional[str] = None
) -> List[Dict]:
    """
    RRF (Reciprocal Rank Fusion) 融合三路搜索结果
    
    RRF 公式: score = sum(weight_i / (k + rank_i))
    
    Args:
        image_results: 图像向量搜索结果，每项需包含 id, rank
        text_results: 文本向量搜索结果
        bm25_results: BM25 全文搜索结果
        config: 融合配置
        category_hint: 可选的品类提示，用于软过滤加权
    
    Returns:
        融合后的结果列表，按分数降序排列
    """
    if not image_results and not text_results and not bm25_results:
        return []
    
    # 建立 ID -> 排名 的映射
    image_ranks = {r["id"]: r["rank"] for r in image_results}
    text_ranks = {r["id"]: r["rank"] for r in text_results}
    bm25_ranks = {r["id"]: r["rank"] for r in bm25_results}
    
    # 收集所有候选 ID 及其元数据
    all_candidates: Dict[int, Dict] = {}
    for results in [image_results, text_results, bm25_results]:
        for r in results:
            if r["id"] not in all_candidates:
                all_candidates[r["id"]] = r.copy()
    
    # 计算 RRF 分数
    k = config.rrf_k
    fused_results = []
    
    for product_id, meta in all_candidates.items():
        score = 0.0
        debug_ranks = {}
        
        # 图像搜索贡献
        if product_id in image_ranks:
            rank = image_ranks[product_id]
            debug_ranks["image"] = rank
            score += config.weight_image / (k + rank)
        
        # 文本向量搜索贡献
        if product_id in text_ranks:
            rank = text_ranks[product_id]
            debug_ranks["text_vector"] = rank
            score += config.weight_text_vector / (k + rank)
        
        # BM25 搜索贡献
        if product_id in bm25_ranks:
            rank = bm25_ranks[product_id]
            debug_ranks["bm25"] = rank
            score += config.weight_bm25 / (k + rank)
        
        # Category 软过滤加权
        if category_hint and meta.get("category"):
            if _category_match(category_hint, meta["category"]):
                score *= config.category_boost
        
        result = meta.copy()
        result["score"] = score
        result["debug_ranks"] = debug_ranks
        fused_results.append(result)
    
    # 按分数降序排序
    fused_results.sort(key=lambda x: x["score"], reverse=True)
    
    # 添加最终排名
    for i, result in enumerate(fused_results, 1):
        result["final_rank"] = i
    
    return fused_results


def _category_match(hint: str, actual: str) -> bool:
    """
    判断 category 是否匹配（支持模糊匹配）
    
    Args:
        hint: 用户提供的品类提示
        actual: 商品实际品类
    
    Returns:
        是否匹配
    """
    hint_lower = hint.lower().strip()
    actual_lower = actual.lower().strip()
    return hint_lower in actual_lower or actual_lower in hint_lower
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_fusion.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/search/fusion.py tests/test_fusion.py
git commit -m "feat: add RRF fusion algorithm for hybrid search"
```

---

## Task 9: 混合搜索服务

**Files:**
- Create: `src/search/hybrid_search.py`
- Create: `tests/test_hybrid_search.py`

**Step 1: 创建混合搜索服务测试**

```python
# tests/test_hybrid_search.py
"""混合搜索服务测试"""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture
def mock_services():
    """创建模拟服务"""
    mock_milvus = MagicMock()
    mock_image_encoder = MagicMock()
    mock_text_encoder = MagicMock()
    mock_desc_generator = MagicMock()
    
    # 设置模拟返回值
    mock_image_encoder.encode = AsyncMock(return_value=np.array([0.1] * 1024))
    mock_text_encoder.encode = AsyncMock(return_value=np.array([0.1] * 2048))
    mock_desc_generator.generate = AsyncMock(return_value="[COLOR] Gray\n[STYLE] Modern")
    
    return {
        "milvus": mock_milvus,
        "image_encoder": mock_image_encoder,
        "text_encoder": mock_text_encoder,
        "desc_generator": mock_desc_generator
    }


@pytest.mark.asyncio
async def test_hybrid_search_service_search(mock_services):
    """测试混合搜索"""
    from src.search.hybrid_search import HybridSearchService
    
    # 模拟 Milvus 搜索结果
    mock_services["milvus"].search_by_vector.return_value = [
        {"id": 1, "rank": 1, "product_code": "A", "category": "sofa",
         "description_ai": "test", "image_url": "http://example.com/1.jpg"},
        {"id": 2, "rank": 2, "product_code": "B", "category": "sofa",
         "description_ai": "test", "image_url": "http://example.com/2.jpg"},
    ]
    mock_services["milvus"].search_by_text.return_value = [
        {"id": 1, "rank": 1, "product_code": "A", "category": "sofa",
         "description_ai": "test", "image_url": "http://example.com/1.jpg"},
    ]
    
    service = HybridSearchService(
        milvus_client=mock_services["milvus"],
        image_encoder=mock_services["image_encoder"],
        text_encoder=mock_services["text_encoder"],
        description_generator=mock_services["desc_generator"]
    )
    
    results, description = await service.search(
        image_bytes=b'\xff\xd8\xff\xe0',
        top_k=10
    )
    
    # 验证调用了编码器和生成器
    mock_services["image_encoder"].encode.assert_called_once()
    mock_services["desc_generator"].generate.assert_called_once()
    mock_services["text_encoder"].encode.assert_called_once()
    
    # 验证返回了融合结果
    assert len(results) > 0
    assert all("score" in r for r in results)


@pytest.mark.asyncio
async def test_hybrid_search_respects_top_k(mock_services):
    """测试 top_k 参数"""
    from src.search.hybrid_search import HybridSearchService
    
    # 返回很多结果
    many_results = [
        {"id": i, "rank": i, "product_code": f"P{i}", "category": "sofa",
         "description_ai": "test", "image_url": f"http://example.com/{i}.jpg"}
        for i in range(1, 51)
    ]
    mock_services["milvus"].search_by_vector.return_value = many_results
    mock_services["milvus"].search_by_text.return_value = many_results[:10]
    
    service = HybridSearchService(
        milvus_client=mock_services["milvus"],
        image_encoder=mock_services["image_encoder"],
        text_encoder=mock_services["text_encoder"],
        description_generator=mock_services["desc_generator"]
    )
    
    results, _ = await service.search(image_bytes=b'\xff\xd8', top_k=5)
    
    assert len(results) <= 5
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_hybrid_search.py -v`
Expected: FAIL with "ImportError"

**Step 3: 实现混合搜索服务**

```python
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
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_hybrid_search.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/search/hybrid_search.py tests/test_hybrid_search.py
git commit -m "feat: add hybrid search service with parallel execution"
```

---

## Task 10: API 依赖注入

**Files:**
- Create: `src/api/deps.py`
- Create: `tests/test_deps.py`

**Step 1: 创建依赖注入测试**

```python
# tests/test_deps.py
"""依赖注入测试"""

import pytest
from unittest.mock import patch, MagicMock


def test_get_settings_returns_singleton():
    """测试配置单例"""
    from src.config import get_settings
    
    # 两次调用返回相同实例
    settings1 = get_settings()
    settings2 = get_settings()
    
    assert settings1 is settings2


@pytest.mark.asyncio
async def test_init_services_creates_search_service(monkeypatch):
    """测试服务初始化"""
    monkeypatch.setenv("ZILLIZ_CLOUD_URI", "https://test.zillizcloud.com")
    monkeypatch.setenv("ZILLIZ_CLOUD_TOKEN", "test_token")
    monkeypatch.setenv("ZILLIZ_CLOUD_COLLECTION", "products")
    monkeypatch.setenv("ZHIPU_API_KEY", "zhipu_key")
    monkeypatch.setenv("ALIYUN_DASHSCOPE_API_KEY", "aliyun_key")
    
    with patch("src.api.deps.MilvusClientWrapper") as mock_milvus:
        from src.api.deps import init_services, get_search_service, _search_service
        
        await init_services()
        
        # 验证 Milvus 客户端被创建
        mock_milvus.assert_called_once()
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_deps.py -v`
Expected: FAIL with "ImportError"

**Step 3: 实现依赖注入**

```python
# src/api/deps.py
"""FastAPI 依赖注入"""

from typing import Optional

from src.config import get_settings, Settings
from src.storage.milvus_client import MilvusClientWrapper
from src.encoders.image_encoder import AliyunImageEncoder
from src.encoders.text_encoder import ZhipuTextEncoder
from src.generators.description import ZhipuDescriptionGenerator
from src.search.hybrid_search import HybridSearchService, SearchConfig


# 全局服务实例
_search_service: Optional[HybridSearchService] = None
_milvus_client: Optional[MilvusClientWrapper] = None
_image_encoder: Optional[AliyunImageEncoder] = None
_text_encoder: Optional[ZhipuTextEncoder] = None
_desc_generator: Optional[ZhipuDescriptionGenerator] = None


async def init_services() -> None:
    """初始化所有服务"""
    global _search_service, _milvus_client
    global _image_encoder, _text_encoder, _desc_generator
    
    settings = get_settings()
    
    # 初始化 Milvus 客户端
    _milvus_client = MilvusClientWrapper(
        uri=settings.zilliz_cloud_uri,
        token=settings.zilliz_cloud_token,
        collection_name=settings.zilliz_cloud_collection
    )
    
    # 初始化编码器和生成器
    _image_encoder = AliyunImageEncoder(api_key=settings.aliyun_dashscope_api_key)
    _text_encoder = ZhipuTextEncoder(api_key=settings.zhipu_api_key)
    _desc_generator = ZhipuDescriptionGenerator(api_key=settings.zhipu_api_key)
    
    # 初始化搜索服务
    search_config = SearchConfig(
        weight_image=settings.search_weight_image,
        weight_text_vector=settings.search_weight_text_vector,
        weight_bm25=settings.search_weight_bm25,
        rrf_k=settings.search_rrf_k,
        candidate_multiplier=settings.search_candidate_multiplier,
        category_boost=settings.search_category_boost
    )
    
    _search_service = HybridSearchService(
        milvus_client=_milvus_client,
        image_encoder=_image_encoder,
        text_encoder=_text_encoder,
        description_generator=_desc_generator,
        config=search_config
    )


async def cleanup_services() -> None:
    """清理所有服务资源"""
    global _milvus_client, _image_encoder, _text_encoder, _desc_generator
    
    if _milvus_client:
        _milvus_client.close()
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


def get_milvus_client() -> MilvusClientWrapper:
    """获取 Milvus 客户端（FastAPI 依赖）"""
    if _milvus_client is None:
        raise RuntimeError("Services not initialized. Call init_services() first.")
    return _milvus_client
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_deps.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/api/deps.py tests/test_deps.py
git commit -m "feat: add dependency injection for FastAPI services"
```

---

## Task 11: 健康检查 API

**Files:**
- Create: `src/api/routes/health.py`
- Create: `tests/test_health_api.py`

**Step 1: 创建健康检查测试**

```python
# tests/test_health_api.py
"""健康检查 API 测试"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """创建测试客户端"""
    from src.api.main import app
    return TestClient(app)


def test_health_check_returns_ok(client):
    """测试健康检查端点"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_health_check_returns_version(client):
    """测试健康检查返回版本号"""
    response = client.get("/health")
    
    data = response.json()
    assert "version" in data
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_health_api.py -v`
Expected: FAIL with "ImportError"

**Step 3: 实现健康检查路由**

```python
# src/api/routes/health.py
"""健康检查路由"""

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    健康检查端点
    
    用于负载均衡器和监控系统检测服务状态
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )
```

**Step 4: 创建 FastAPI 应用入口（需要先创建才能运行测试）**

```python
# src/api/main.py
"""FastAPI 应用入口"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health
from src.api.deps import init_services, cleanup_services


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化服务
    try:
        await init_services()
    except Exception:
        # 允许在没有完整配置时启动（用于测试）
        pass
    yield
    # 关闭时清理资源
    await cleanup_services()


app = FastAPI(
    title="Furniture Search API",
    description="基于 AI 设计图的家具相似搜索服务",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, tags=["Health"])
```

**Step 5: 运行测试验证通过**

Run: `pytest tests/test_health_api.py -v`
Expected: PASS

**Step 6: 提交**

```bash
git add src/api/routes/health.py src/api/main.py tests/test_health_api.py
git commit -m "feat: add health check API endpoint"
```

---

## Task 12: 搜索 API

**Files:**
- Create: `src/api/routes/search.py`
- Create: `tests/test_search_api.py`

**Step 1: 创建搜索 API 测试**

```python
# tests/test_search_api.py
"""搜索 API 测试"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import io


@pytest.fixture
def mock_search_service():
    """模拟搜索服务"""
    mock = MagicMock()
    mock.search = AsyncMock(return_value=(
        [
            {
                "id": 1,
                "product_code": "SF-001",
                "category": "sofa",
                "description_ai": "[COLOR] Gray",
                "image_url": "http://example.com/1.jpg",
                "score": 0.85,
                "final_rank": 1
            }
        ],
        "[COLOR] Gray\n[STYLE] Modern"
    ))
    return mock


@pytest.fixture
def client(mock_search_service):
    """创建带模拟服务的测试客户端"""
    from src.api.main import app
    from src.api import deps
    
    # 替换依赖
    app.dependency_overrides[deps.get_search_service] = lambda: mock_search_service
    
    client = TestClient(app)
    yield client
    
    # 清理
    app.dependency_overrides.clear()


def test_search_image_returns_results(client, mock_search_service):
    """测试图像搜索返回结果"""
    # 创建测试图片
    image_content = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
    files = {"image": ("test.jpg", io.BytesIO(image_content), "image/jpeg")}
    
    response = client.post("/api/v1/search/image", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "results" in data["data"]
    assert len(data["data"]["results"]) == 1


def test_search_image_with_top_k(client, mock_search_service):
    """测试 top_k 参数"""
    image_content = b'\xff\xd8\xff\xe0\x00\x10JFIF'
    files = {"image": ("test.jpg", io.BytesIO(image_content), "image/jpeg")}
    
    response = client.post("/api/v1/search/image?top_k=5", files=files)
    
    assert response.status_code == 200
    # 验证 search 被调用时传入了 top_k=5
    mock_search_service.search.assert_called_once()
    call_kwargs = mock_search_service.search.call_args[1]
    assert call_kwargs["top_k"] == 5


def test_search_image_with_category_hint(client, mock_search_service):
    """测试 category_hint 参数"""
    image_content = b'\xff\xd8\xff\xe0\x00\x10JFIF'
    files = {"image": ("test.jpg", io.BytesIO(image_content), "image/jpeg")}
    
    response = client.post(
        "/api/v1/search/image?category_hint=sofa",
        files=files
    )
    
    assert response.status_code == 200
    call_kwargs = mock_search_service.search.call_args[1]
    assert call_kwargs["category_hint"] == "sofa"


def test_search_image_missing_file(client):
    """测试缺少图片文件"""
    response = client.post("/api/v1/search/image")
    
    assert response.status_code == 422  # Validation Error
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_search_api.py -v`
Expected: FAIL with "ImportError" or router not found

**Step 3: 实现搜索路由**

```python
# src/api/routes/search.py
"""搜索路由"""

import time
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Query, Depends, HTTPException

from src.api.deps import get_search_service
from src.search.hybrid_search import HybridSearchService
from src.models.schemas import SearchResponse, SearchData, SearchMeta, SearchResult


router = APIRouter()


@router.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    image: UploadFile = File(..., description="输入图片"),
    top_k: int = Query(default=20, ge=1, le=100, description="返回结果数量"),
    category_hint: Optional[str] = Query(default=None, description="品类提示"),
    search_service: HybridSearchService = Depends(get_search_service)
) -> SearchResponse:
    """
    基于图片的家具相似搜索
    
    上传一张家具图片（如 AI 生成的设计图），返回最相似的商品列表。
    """
    start_time = time.time()
    
    # 读取图片数据
    image_bytes = await image.read()
    
    # 验证图片
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")
    
    # 执行搜索
    try:
        results, query_description = await search_service.search(
            image_bytes=image_bytes,
            top_k=top_k,
            category_hint=category_hint
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    # 构建响应
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    return SearchResponse(
        success=True,
        data=SearchData(
            query_description=query_description,
            results=[
                SearchResult(
                    product_id=r["id"],
                    product_code=r["product_code"],
                    category=r["category"],
                    description_ai=r["description_ai"],
                    image_url=r["image_url"],
                    score=r["score"],
                    rank=r.get("final_rank", 0),
                    debug_ranks=r.get("debug_ranks")
                )
                for r in results
            ]
        ),
        meta=SearchMeta(
            took_ms=elapsed_ms,
            total_candidates=len(results)
        )
    )
```

**Step 4: 更新 main.py 注册搜索路由**

```python
# 在 src/api/main.py 中添加
from src.api.routes import health, search

# 注册路由部分更新为：
app.include_router(health.router, tags=["Health"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
```

**Step 5: 运行测试验证通过**

Run: `pytest tests/test_search_api.py -v`
Expected: PASS

**Step 6: 提交**

```bash
git add src/api/routes/search.py src/api/main.py tests/test_search_api.py
git commit -m "feat: add image search API endpoint"
```

---

## Task 13: Collection 初始化脚本

**Files:**
- Create: `scripts/init_collection.py`

**Step 1: 创建初始化脚本**

```python
# scripts/init_collection.py
"""初始化 Zilliz Cloud Collection"""

import os
import sys

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from pymilvus import MilvusClient

from src.storage.milvus_client import get_collection_schema, get_index_params


def create_collection():
    """创建 Milvus Collection"""
    load_dotenv()
    
    uri = os.getenv("ZILLIZ_CLOUD_URI")
    token = os.getenv("ZILLIZ_CLOUD_TOKEN")
    collection_name = os.getenv("ZILLIZ_CLOUD_COLLECTION", "products")
    
    if not uri or not token:
        print("Error: ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_TOKEN must be set")
        sys.exit(1)
    
    client = MilvusClient(uri=uri, token=token)
    
    # 检查是否已存在
    if client.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        response = input("Do you want to drop and recreate it? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        client.drop_collection(collection_name)
        print(f"Dropped existing collection '{collection_name}'")
    
    # 创建 schema 和索引
    schema = get_collection_schema(client)
    index_params = get_index_params(client)
    
    # 创建 collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    
    print(f"Collection '{collection_name}' created successfully!")
    
    # 显示 collection 信息
    info = client.describe_collection(collection_name)
    print(f"\nCollection info:")
    print(f"  - Name: {info['collection_name']}")
    print(f"  - Fields: {len(info['fields'])}")
    for field in info['fields']:
        print(f"    - {field['name']}: {field['type']}")
    
    client.close()


if __name__ == "__main__":
    create_collection()
```

**Step 2: 提交**

```bash
git add scripts/init_collection.py
git commit -m "feat: add collection initialization script"
```

---

## Task 14: 批量导入脚本

**Files:**
- Create: `scripts/batch_import.py`

**Step 1: 创建批量导入脚本**

```python
# scripts/batch_import.py
"""批量导入商品数据"""

import os
import sys
import json
import csv
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from tqdm import tqdm

from src.storage.milvus_client import MilvusClientWrapper
from src.encoders.image_encoder import AliyunImageEncoder
from src.encoders.text_encoder import ZhipuTextEncoder
from src.generators.description import ZhipuDescriptionGenerator


@dataclass
class ProductInput:
    """商品输入"""
    product_code: str
    category: str
    image_path: str
    description_human: Optional[str] = None


class BatchImporter:
    """批量导入器"""
    
    def __init__(
        self,
        milvus_client: MilvusClientWrapper,
        image_encoder: AliyunImageEncoder,
        text_encoder: ZhipuTextEncoder,
        desc_generator: ZhipuDescriptionGenerator,
        batch_size: int = 10
    ):
        self.milvus = milvus_client
        self.img_encoder = image_encoder
        self.text_encoder = text_encoder
        self.desc_gen = desc_generator
        self.batch_size = batch_size
    
    async def import_from_csv(self, csv_path: str):
        """从 CSV 文件导入"""
        products = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                products.append(ProductInput(
                    product_code=row['product_code'],
                    category=row['category'],
                    image_path=row['image_path'],
                    description_human=row.get('description_human', '')
                ))
        await self._import_products(products)
    
    async def import_from_json(self, json_path: str):
        """从 JSON 文件导入"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        products = [
            ProductInput(
                product_code=p['product_code'],
                category=p['category'],
                image_path=p['image_path'],
                description_human=p.get('description_human', '')
            )
            for p in data['products']
        ]
        await self._import_products(products)
    
    async def import_from_directory(self, data_dir: str):
        """从目录结构导入"""
        products = []
        data_path = Path(data_dir)
        
        for category_dir in data_path.iterdir():
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for image_file in category_dir.glob(ext):
                    products.append(ProductInput(
                        product_code=image_file.stem,
                        category=category,
                        image_path=str(image_file),
                        description_human=''
                    ))
        
        await self._import_products(products)
    
    async def _import_products(self, products: List[ProductInput]):
        """批量处理并导入商品"""
        print(f"Total products to import: {len(products)}")
        
        for i in tqdm(range(0, len(products), self.batch_size), desc="Importing"):
            batch = products[i:i + self.batch_size]
            records = await self._process_batch(batch)
            
            if records:
                self.milvus.insert(records)
        
        print(f"Successfully imported {len(products)} products!")
    
    async def _process_batch(self, batch: List[ProductInput]) -> List[Dict]:
        """处理一批商品"""
        tasks = [self._process_single(p) for p in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        records = []
        for product, result in zip(batch, results):
            if isinstance(result, Exception):
                print(f"\nError processing {product.product_code}: {result}")
                continue
            records.append(result)
        
        return records
    
    async def _process_single(self, product: ProductInput) -> Dict:
        """处理单个商品"""
        # 读取图片
        with open(product.image_path, 'rb') as f:
            image_bytes = f.read()
        
        # 并行生成描述和图像 embedding
        desc_task = self.desc_gen.generate(image_bytes)
        img_emb_task = self.img_encoder.encode(image_bytes)
        
        description_ai, image_embedding = await asyncio.gather(desc_task, img_emb_task)
        
        # 生成文本 embedding
        text_embedding = await self.text_encoder.encode(description_ai)
        
        return {
            "product_code": product.product_code,
            "category": product.category,
            "description_human": product.description_human or "",
            "description_ai": description_ai,
            "image_embedding": image_embedding.tolist(),
            "text_embedding": text_embedding.tolist(),
            "image_url": product.image_path,
            "created_at": int(time.time())
        }


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量导入商品数据')
    parser.add_argument('--input', '-i', required=True, help='输入文件或目录路径')
    parser.add_argument('--format', '-f', choices=['csv', 'json', 'dir'], default='csv',
                        help='输入格式: csv, json, dir')
    parser.add_argument('--batch-size', '-b', type=int, default=10, help='批处理大小')
    args = parser.parse_args()
    
    load_dotenv()
    
    # 初始化组件
    milvus_client = MilvusClientWrapper(
        uri=os.getenv("ZILLIZ_CLOUD_URI"),
        token=os.getenv("ZILLIZ_CLOUD_TOKEN"),
        collection_name=os.getenv("ZILLIZ_CLOUD_COLLECTION", "products")
    )
    image_encoder = AliyunImageEncoder(os.getenv("ALIYUN_DASHSCOPE_API_KEY"))
    text_encoder = ZhipuTextEncoder(os.getenv("ZHIPU_API_KEY"))
    desc_generator = ZhipuDescriptionGenerator(os.getenv("ZHIPU_API_KEY"))
    
    importer = BatchImporter(
        milvus_client=milvus_client,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        desc_generator=desc_generator,
        batch_size=args.batch_size
    )
    
    try:
        if args.format == 'csv':
            await importer.import_from_csv(args.input)
        elif args.format == 'json':
            await importer.import_from_json(args.input)
        else:
            await importer.import_from_directory(args.input)
    finally:
        milvus_client.close()
        await image_encoder.close()
        await text_encoder.close()
        await desc_generator.close()


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: 提交**

```bash
git add scripts/batch_import.py
git commit -m "feat: add batch import script for products"
```

---

## Task 15: pytest 配置和 conftest

**Files:**
- Create: `tests/conftest.py`

**Step 1: 创建 pytest 配置**

```python
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
```

**Step 2: 运行所有测试**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 3: 提交**

```bash
git add tests/conftest.py
git commit -m "feat: add pytest configuration and fixtures"
```

---

## Task 16: 创建 README

**Files:**
- Create: `README.md`

**Step 1: 创建项目文档**

```markdown
# 家具图像混合搜索系统

基于 AI 设计图的家具相似搜索 API 服务，使用三路混合搜索（图像向量、文本向量、BM25）和 RRF 融合算法。

## 功能特性

- **三路混合搜索**：结合图像向量、文本向量和 BM25 全文搜索
- **RRF 融合排序**：智能融合多路搜索结果
- **AI 描述生成**：使用 VLM 自动生成商品描述
- **品类加权**：支持品类提示的软过滤加权

## 技术栈

- **Web 框架**: FastAPI
- **向量数据库**: Zilliz Cloud (Milvus)
- **图像编码**: 阿里云 Multimodal-Embedding
- **文本编码**: 智谱 Embedding-3
- **描述生成**: 智谱 GLM-4.6V-Flash

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入实际的 API 密钥
```

### 3. 初始化数据库

```bash
python scripts/init_collection.py
```

### 4. 导入商品数据

```bash
# 从 CSV 导入
python scripts/batch_import.py -i data/products.csv -f csv

# 从目录结构导入
python scripts/batch_import.py -i data/images/ -f dir
```

### 5. 启动服务

```bash
# 开发模式
uvicorn src.api.main:app --reload --port 8000

# 生产模式
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API 使用

### 图像搜索

```bash
curl -X POST "http://localhost:8000/api/v1/search/image" \
  -F "image=@design.jpg" \
  -F "top_k=20" \
  -F "category_hint=sofa"
```

### 健康检查

```bash
curl http://localhost:8000/health
```

## 项目结构

```
furniture_search_demo/
├── src/
│   ├── api/           # FastAPI 应用
│   ├── encoders/      # 图像/文本编码器
│   ├── generators/    # 描述生成器
│   ├── search/        # 搜索服务
│   └── storage/       # 数据存储
├── tests/             # 测试
├── scripts/           # 脚本
└── docs/              # 文档
```

## 测试

```bash
pytest tests/ -v
```

## License

MIT
```

**Step 2: 提交**

```bash
git add README.md
git commit -m "docs: add project README"
```

---

## 完成检查清单

- [ ] Task 1: 项目初始化
- [ ] Task 2: 配置管理模块
- [ ] Task 3: Pydantic 数据模型
- [ ] Task 4: Milvus 客户端封装
- [ ] Task 5: 图像编码器
- [ ] Task 6: 文本编码器
- [ ] Task 7: 描述生成器
- [ ] Task 8: RRF 融合算法
- [ ] Task 9: 混合搜索服务
- [ ] Task 10: API 依赖注入
- [ ] Task 11: 健康检查 API
- [ ] Task 12: 搜索 API
- [ ] Task 13: Collection 初始化脚本
- [ ] Task 14: 批量导入脚本
- [ ] Task 15: pytest 配置
- [ ] Task 16: 创建 README

---

## 后续优化（可选）

完成核心功能后，可以考虑以下优化：

1. **商品入库 API** - 添加 `POST /api/v1/products` 端点
2. **错误处理增强** - 添加降级策略和详细错误码
3. **监控指标** - 集成 Prometheus metrics
4. **Docker 部署** - 创建 Dockerfile 和 docker-compose.yml
5. **评估脚本** - 创建搜索效果评估工具
