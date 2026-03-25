# 家具图像混合搜索系统 — 架构文档

> 生成时间：2026-03-25
> 项目版本：0.1.0

---

## 1. 项目概览

### 1.1 业务定位

本系统解决的核心问题：**用户使用 AI 生成家具设计效果图后，希望在家具商品库中找到外观最相近的真实商品。**

核心挑战：

| 挑战 | 描述 |
|------|------|
| **Domain Gap** | AI 渲染图（无噪点、理想光影）vs 真实商品照片（有环境反射、真实光影） |
| **语义匹配** | 用户期望的"相似"不是像素级匹配，而是风格/用途/外观的语义相似 |
| **多维度特征** | 颜色、材质、风格、造型、尺寸等多个维度都可能影响匹配结果 |

### 1.2 解决方案：三路混合搜索 + RRF 融合

```
用户上传设计图
    │
    ├── 路径A: 图像 → Aliyun Multimodal-Embedding → 1024D 向量 → ANN 搜索（视觉相似）
    │
    ├── 路径B: 图像 → GLM-4V-Flash VLM → 文本描述 → Zhipu Embedding-3 → 2048D 向量 → ANN 搜索（语义相似）
    │
    └── 路径C: VLM 描述文本 → Zilliz BM25 全文索引 → 关键词搜索（精确属性匹配）
         │
         ▼
    RRF (Reciprocal Rank Fusion) 融合排序
         │
         ▼
    Top-K 结果返回
```

### 1.3 项目双模式架构

项目实际包含 **两套实现**，共享相同的核心逻辑：

| 维度 | `src/` 模式（原始实现） | `furniture_search/` 模式（SDK 封装） |
|------|------|------|
| **定位** | FastAPI 后端服务 | 可复用 SDK + HTTP 服务 |
| **存储客户端** | Zilliz REST API (`zilliz_client.py`) | Zilliz REST API (`zilliz_client.py`) |
| **RRF 融合** | 服务端（Zilliz hybrid_search API 内置） | 服务端（Zilliz hybrid_search API 内置） |
| **搜索路径数** | 2 路（图像向量 + 文本向量） | 2 路（图像向量 + 文本向量） |
| **BM25** | 通过 Zilliz BM25 索引（hybrid_search 可选） | 通过 Zilliz BM25 索引 |
| **前端** | Vue 3 单页 HTML | 无前端 |
| **Python 调用** | 仅 HTTP | 直接 import SDK 或 HTTP |
| **Docker** | 无 | 有 Dockerfile |
| **状态** | 代码中有遗留引用 (`milvus_client`) | 干净、可用 |

> **注意**：设计文档中描述的是"三路混合搜索"（含 BM25），但当前实际实现已简化为 **双路搜索**（图像向量 + 文本向量），RRF 融合由 Zilliz 服务端 hybrid_search API 内置完成。BM25 作为 Zilliz Collection 的可选索引存在，但未被搜索流程主动调用。`src/search/fusion.py` 中的客户端 RRF 融合代码目前未被使用（仅用于测试）。

---

## 2. 系统架构

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              客户端层                                        │
│                                                                              │
│   ┌─────────────────────┐          ┌──────────────────────────────────┐      │
│   │  Vue 3 前端 (CDN)   │          │  SDK / 第三方 HTTP 客户端        │      │
│   │  frontend/index.html│          │  (Java, Go, Python, JS, cURL)   │      │
│   └──────────┬──────────┘          └──────────────┬───────────────────┘      │
└──────────────┼────────────────────────────────────┼──────────────────────────┘
               │ multipart/form-data                │ JSON (base64 image)
               ▼                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              API 层                                          │
│                                                                              │
│  ┌──────────────────────────────────┐  ┌──────────────────────────────────┐  │
│  │  FastAPI (src/api/)              │  │  FastAPI (furniture_search/)    │  │
│  │  POST /api/v1/search/image       │  │  POST /search                   │  │
│  │  GET  /health                    │  │  GET  /health                   │  │
│  └──────────────┬───────────────────┘  └──────────────┬───────────────────┘  │
└─────────────────┼────────────────────────────────────┼────────────────────────┘
                  │                                    │
                  └──────────────┬─────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            服务层 (Service Layer)                              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                    HybridSearchService                               │     │
│  │                                                                     │     │
│  │  Step 1 (并行):                                                     │     │
│  │  ┌──────────────────────┐  ┌───────────────────────────────────┐   │     │
│  │  │  AliyunImageEncoder  │  │  ZhipuDescriptionGenerator        │   │     │
│  │  │  Image → 1024D vec   │  │  Image → Structured Text          │   │     │
│  │  │  (阿里云 DashScope)  │  │  (GLM-4V-Flash VLM)               │   │     │
│  │  └──────────┬───────────┘  └───────────────┬───────────────────┘   │     │
│  │             │                              │                       │     │
│  │             │                              ▼                       │     │
│  │             │              ┌───────────────────────────────────┐   │     │
│  │             │              │  ZhipuTextEncoder                 │   │     │
│  │             │              │  Text → 2048D vec                │   │     │
│  │             │              │  (Embedding-3)                    │   │     │
│  │             │              └───────────────┬───────────────────┘   │     │
│  │             │                              │                       │     │
│  │  Step 2:    └──────────────┬───────────────┘                       │     │
│  │             ▼              ▼                                       │     │
│  │  ┌─────────────────────────────────────────────────────────────┐ │     │
│  │  │         Zilliz Cloud Hybrid Search API                      │ │     │
│  │  │  ┌──────────────────┐  ┌──────────────────────────────────┐ │ │     │
│  │  │  │ imageVector ANN  │  │ vector ANN (2048D)               │ │ │     │
│  │  │  │ COSINE, HNSW     │  │ COSINE, HNSW                    │ │ │     │
│  │  │  └────────┬─────────┘  └──────────────┬───────────────────┘ │ │     │
│  │  │           └─────────────┬─────────────┘                    │ │     │
│  │  │                         ▼                                  │ │     │
│  │  │              RRF 服务端融合 (k=60)                         │ │     │
│  │  └────────────────────────┬───────────────────────────────────┘ │     │
│  └───────────────────────────┼─────────────────────────────────────┘     │
│                              │                                            │
└──────────────────────────────┼────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           数据层 (Data Layer)                                 │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │              Zilliz Cloud (Milvus 托管服务)                          │   │
│  │                                                                      │   │
│  │  Collection: products / furniture_products                          │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │ 字段             │ 类型             │ 索引                   │   │   │
│  │  ├──────────────────┼──────────────────┼────────────────────────┤   │   │
│  │  │ id               │ INT64 (PK)       │ -                      │   │   │
│  │  │ sku              │ VARCHAR          │ -                      │   │   │
│  │  │ name             │ VARCHAR          │ -                      │   │   │
│  │  │ category         │ VARCHAR          │ -                      │   │   │
│  │  │ price            │ VARCHAR          │ -                      │   │   │
│  │  │ discription      │ VARCHAR          │ - (注意: 拼写)         │   │   │
│  │  │ LLMDescription   │ VARCHAR          │ BM25 (可选)            │   │   │
│  │  │ imageVector      │ FLOAT_VECTOR     │ HNSW (M=16)           │   │   │
│  │  │                  │ 1024D            │ COSINE                 │   │   │
│  │  │ vector           │ FLOAT_VECTOR     │ HNSW (M=16)           │   │   │
│  │  │                  │ 2048D            │ COSINE                 │   │   │
│  │  │ url              │ VARCHAR          │ -                      │   │   │
│  │  │ imageUrl         │ VARCHAR          │ -                      │   │   │
│  │  └──────────────────┴──────────────────┴────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
                    ┌─────────────┐
                    │   config    │
                    │  (Settings) │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌─────────────┐ ┌──────────┐ ┌──────────────┐
     │ zilliz_     │ │ encoder  │ │ description  │
     │ client      │ │ (img/txt)│ │ _generator   │
     └──────┬──────┘ └────┬─────┘ └──────┬───────┘
            │             │              │
            ▼             ▼              ▼
     ┌─────────────────────────────────────────┐
     │         HybridSearchService              │
     │         (hybrid_search.py)               │
     ├─────────────────────────────────────────┤
     │  - 并行: image_encode + desc_generate   │
     │  - 串行: text_encode (依赖 desc)        │
     │  - Zilliz hybrid_search (服务端 RRF)    │
     └──────────────────┬──────────────────────┘
                        │
              ┌─────────┼──────────┐
              ▼                    ▼
     ┌──────────────┐    ┌───────────────┐
     │  deps.py     │    │ fusion.py     │
     │  (DI 容器)   │    │ (客户端 RRF)  │
     │  全局单例    │    │ 当前未使用     │
     └──────┬───────┘    └───────────────┘
            │
            ▼
     ┌──────────────┐
     │  routes/     │
     │  search.py   │
     │  health.py   │
     └──────────────┘
```

### 2.3 请求处理时序

```
Client                   FastAPI              HybridSearchSvc         External APIs
  │                         │                         │                       │
  │  POST /search/image     │                         │                       │
  │  (multipart/form-data)  │                         │                       │
  │────────────────────────>│                         │                       │
  │                         │                         │                       │
  │                         │  search(image_bytes)    │                       │
  │                         │────────────────────────>│                       │
  │                         │                         │                       │
  │                         │                         │  encode(image_bytes)   │
  │                         │                         │  generate(image_bytes) │
  │                         │                         │──────────────┐        │
  │                         │                         │  asyncio.gather │       │
  │                         │                         │  ┌─────────────┘       │
  │                         │                         │  │                    │
  │                         │                         │  ├─ POST 阿里云 DashScope│
  │                         │                         │  │  (图像 → 1024D)     │
  │                         │                         │  │                     │
  │                         │                         │  ├─ POST 智谱 GLM-4V  │
  │                         │                         │  │  (图像 → 文本描述)  │
  │                         │                         │  │                     │
  │                         │                         │  ◀── image_emb, desc   │
  │                         │                         │                       │
  │                         │                         │  encode(desc)         │
  │                         │                         │─────────────────────> │
  │                         │                         │  POST 智谱 Embedding-3 │
  │                         │                         │  (文本 → 2048D)       │
  │                         │                         │◀── text_emb           │
  │                         │                         │                       │
  │                         │                         │  hybrid_search()      │
  │                         │                         │─────────────────────> │
  │                         │                         │  POST Zilliz Cloud    │
  │                         │                         │  /v2/vectordb/entities │
  │                         │                         │    /hybrid_search     │
  │                         │                         │  (服务端 RRF 融合)     │
  │                         │                         │◀── results            │
  │                         │                         │                       │
  │                         │◀── (results, desc)      │                       │
  │◀── SearchResponse       │                         │                       │
  │                         │                         │                       │
```

### 2.4 数据入库时序（批量导入）

```
BatchImporter              Encoders/Generators           Storage
     │                           │                        │
     │  for batch in products    │                        │
     │──────────────────────────>│                        │
     │                           │                        │
     │                           │  asyncio.gather:       │
     │                           │  ├─ generate(image)    │
     │                           │  └─ encode(image)      │
     │                           │───┐  ┌──────────┐     │
     │                           │   │  │ POST API │     │
     │                           │◀──┘  └──────────┘     │
     │                           │                        │
     │                           │  encode(description)   │
     │                           │─── POST API ────>      │
     │                           │◀── text_emb            │
     │                           │                        │
     │◀── record                 │                        │
     │                           │                        │
     │  insert(records)          │                        │
     │──────────────────────────────────────────────────>│
     │                           │                        │
     │                           │                        │── POST Zilliz API
```

---

## 3. 核心模块详解

### 3.1 配置管理 (`src/config.py`)

基于 `pydantic-settings`，从 `.env` 文件和环境变量加载配置。使用 `@lru_cache` 保证单例。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `zilliz_cloud_uri` | str | (必填) | Zilliz Cloud 集群 endpoint |
| `zilliz_cloud_token` | str | (必填) | Zilliz Cloud API token |
| `zilliz_cloud_collection` | str | `"products"` | Collection 名称 |
| `zhipu_api_key` | str | (必填) | 智谱 AI API key |
| `aliyun_dashscope_api_key` | str | (必填) | 阿里云 DashScope API key |
| `search_rrf_k` | int | 60 | RRF 融合平滑常数 |
| `search_candidate_multiplier` | int | 3 | 候选数量倍数 (top_k × N) |

### 3.2 编码器

#### 3.2.1 图像编码器 (`src/encoders/image_encoder.py`)

- **协议**: `ImageEncoder` (Python Protocol)
- **实现**: `AliyunImageEncoder`
- **API**: 阿里云 DashScope Multimodal-Embedding
- **输出**: 1024 维 float32 向量
- **图片预处理**: 自动压缩至 2.8MB 以下（渐进式降低质量 + 缩放尺寸）
- **HTTP 客户端**: httpx.AsyncClient（懒加载，超时 30s）

#### 3.2.2 文本编码器 (`src/encoders/text_encoder.py`)

- **协议**: `TextEncoder` (Python Protocol)
- **实现**: `ZhipuTextEncoder`
- **API**: 智谱 Embedding-3
- **输出**: 2048 维 float32 向量
- **HTTP 客户端**: httpx.AsyncClient（懒加载，超时 30s）

### 3.3 描述生成器 (`src/generators/description.py`)

- **协议**: `DescriptionGenerator` (Python Protocol)
- **实现**: `ZhipuDescriptionGenerator`
- **模型**: GLM-4V-Flash (VLM)
- **Prompt**: 结构化描述，包含 `[COLOR]`, `[MATERIAL]`, `[STYLE]`, `[SHAPE]`, `[SIZE]`, `[SUMMARY]` 标签
- **设计要点**: 入库和搜索使用 **同一个 prompt**，确保描述在同一语义空间
- **HTTP 客户端**: httpx.AsyncClient（懒加载，超时 30s，max_tokens=500）

### 3.4 混合搜索服务 (`src/search/hybrid_search.py`)

核心编排逻辑，将编码、生成、搜索串联：

```python
async def search(image_bytes, top_k, category_hint):
    # Step 1: 并行 — 图像编码 + 描述生成
    image_emb, description = await asyncio.gather(
        image_encoder.encode(image_bytes),
        desc_generator.generate(image_bytes)
    )
    
    # Step 2: 串行 — 文本编码（依赖描述）
    text_emb = await text_encoder.encode(description)
    
    # Step 3: Zilliz hybrid_search（服务端双路 ANN + RRF）
    results = await zilliz.hybrid_search(
        vector_searches=[
            {"data": text_emb, "anns_field": "vector"},
            {"data": image_emb, "anns_field": "imageVector"},
        ],
        limit=top_k,
        rrf_k=60
    )
    
    return results, description
```

**并行化分析**：
- ✅ 图像编码 ∥ 描述生成（约节省 ~500ms）
- ❌ 文本编码 ⊥ 描述生成（因为依赖 VLM 输出）

### 3.5 RRF 融合 (`src/search/fusion.py`)

客户端 RRF 实现，支持三路搜索结果融合 + category 软过滤加权：

```
RRF_score(d) = Σ weight_i / (k + rank_i(d))
```

**当前状态**：代码完整且经过测试，但 `HybridSearchService` 实际使用 Zilliz 服务端 RRF，此客户端融合代码 **未被调用**。

### 3.6 Zilliz 客户端 (`src/storage/zilliz_client.py`)

纯 HTTP REST API 客户端，零 SDK 依赖（不使用 pymilvus）：

| 方法 | API 端点 | 说明 |
|------|---------|------|
| `hybrid_search()` | `/v2/vectordb/entities/hybrid_search` | 多路向量搜索 + 服务端 RRF |
| `search()` | `/v2/vectordb/entities/search` | 单路向量搜索 |
| `insert()` | `/v2/vectordb/entities/insert` | 数据插入 |

**设计特点**：每次请求创建新的 `httpx.AsyncClient`（`async with` 语句），不维护长连接。

### 3.7 API 层

#### `src/api/main.py` — FastAPI 应用入口
- 使用 `lifespan` 管理服务生命周期
- CORS 全开放（`allow_origins=["*"]`）
- 路由：`/health` + `/api/v1/search/image`

#### `src/api/routes/search.py` — 搜索接口
- 接收 `multipart/form-data`（图片文件 + query 参数）
- 字段映射处理 Zilliz 返回与 API schema 的差异（如 `discription` 拼写修正）
- 计时并返回 `took_ms`

#### `src/api/deps.py` — 依赖注入
- 全局单例模式（模块级变量）
- `init_services()` / `cleanup_services()` 管理生命周期
- 注意：未初始化时抛出 `RuntimeError`

### 3.8 数据模型 (`src/models/schemas.py`)

Pydantic v2 模型，定义 API 请求/响应契约：

| 模型 | 用途 | 关键字段 |
|------|------|---------|
| `SearchResult` | 搜索结果项 | product_id, sku, name, category, price, score, rank |
| `SearchRequest` | 搜索参数 | top_k (1-100), category_hint (可选) |
| `SearchResponse` | 搜索响应 | success, data (SearchData), meta (SearchMeta) |
| `ProductInput` | 商品导入 | sku, name, category, price, image_path |
| `ErrorResponse` | 错误响应 | success=false, error_code, message |

### 3.9 SDK 模式 (`furniture_search/`)

独立的 SDK 包，提供两种使用方式：

1. **Python 直接调用**：`FurnitureSearchClient` 类，支持 async context manager
2. **HTTP 服务**：`server.py` 提供 REST API，接收 base64 编码图片

SDK 内部是 `src/` 核心逻辑的 **精简复刻**，有自己的 `core/` 目录，包含 `hybrid_search.py`, `encoders/`, `generators/`, `storage/`。

### 3.10 前端 (`frontend/index.html`)

Vue 3 CDN 单文件应用，零构建：

- 拖拽/点击上传图片
- top_k 滑块控制（1-100）
- category_hint 文本输入
- 结果网格展示（卡片式，含图片、名称、价格、得分）
- 可配置 API 地址
- 错误 Toast 提示

---

## 4. 技术栈

| 层级 | 技术 | 版本要求 | 说明 |
|------|------|---------|------|
| **语言** | Python | ≥ 3.10 | 使用 `match`, `X \| Y` 类型语法 |
| **Web 框架** | FastAPI | ≥ 0.109.0 | 异步 API 框架 |
| **ASGI 服务器** | Uvicorn | ≥ 0.27.0 | 支持多 worker |
| **HTTP 客户端** | httpx | ≥ 0.26.0 | 异步 HTTP，所有外部 API 调用 |
| **向量数据库** | Zilliz Cloud REST API | - | 零 SDK 依赖 |
| **图像编码** | 阿里云 Multimodal-Embedding | v1 | 1024 维 |
| **文本编码** | 智谱 Embedding-3 | - | 2048 维 |
| **描述生成** | 智谱 GLM-4V-Flash | - | VLM 图像理解 |
| **数据验证** | Pydantic v2 | ≥ 2.6.0 | 模型定义 + settings |
| **图像处理** | Pillow | ≥ 10.2.0 | 图片压缩 |
| **数据格式** | NumPy | ≥ 1.26.0 | 向量处理 |
| **前端** | Vue 3 (CDN) | - | 单文件 HTML |
| **测试** | pytest + pytest-asyncio | ≥ 8.0.0 | 单元测试 |
| **Lint** | Ruff | - | line-length=100 |
| **配置** | python-dotenv + pydantic-settings | - | .env + 类型安全 |

---

## 5. 文件结构详图

```
f_search_demo/
│
├── src/                             # 后端主代码
│   ├── __init__.py
│   ├── config.py                    # [配置] pydantic-settings 单例
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                  # [入口] FastAPI app + lifespan + CORS
│   │   ├── deps.py                  # [DI] 全局单例服务工厂
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── search.py            # [路由] POST /api/v1/search/image
│   │       └── health.py            # [路由] GET /health
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── image_encoder.py         # [编码] 阿里云 1024D 图像向量
│   │   └── text_encoder.py          # [编码] 智谱 2048D 文本向量
│   ├── generators/
│   │   ├── __init__.py
│   │   └── description.py           # [VLM] 智谱 GLM-4V-Flash 描述生成
│   ├── search/
│   │   ├── __init__.py
│   │   ├── hybrid_search.py         # [核心] 混合搜索编排服务
│   │   └── fusion.py                # [算法] 客户端 RRF 融合（当前未使用）
│   ├── storage/
│   │   ├── __init__.py
│   │   └── zilliz_client.py         # [存储] Zilliz Cloud REST API 客户端
│   └── models/
│       ├── __init__.py
│       └── schemas.py               # [模型] Pydantic 请求/响应模型
│
├── furniture_search/                # SDK 封装（独立可分发）
│   ├── __init__.py                  # SDK 入口
│   ├── client.py                    # [客户端] FurnitureSearchClient
│   ├── config.py                    # [配置] Settings
│   ├── server.py                    # [HTTP] 独立 FastAPI 服务
│   ├── Dockerfile                   # [部署] Docker 镜像
│   ├── start.sh                     # [部署] 启动脚本
│   ├── core/
│   │   ├── hybrid_search.py         # 混合搜索（同 src/）
│   │   ├── encoders/                # 编码器（同 src/）
│   │   ├── generators/              # 生成器（同 src/）
│   │   └── storage/                 # 存储客户端（同 src/）
│   └── examples/                    # 使用示例
│
├── frontend/
│   └── index.html                   # Vue 3 单页前端
│
├── scripts/
│   ├── init_collection.py           # [工具] 初始化 Zilliz Collection
│   └── batch_import.py              # [工具] 批量导入商品
│
├── tests/                           # 单元测试
│   ├── conftest.py                  # 全局 fixtures + env 设置
│   ├── test_config.py
│   ├── test_schemas.py
│   ├── test_text_encoder.py
│   ├── test_image_encoder.py
│   ├── test_description_generator.py
│   ├── test_fusion.py
│   ├── test_hybrid_search.py
│   ├── test_zilliz_client.py
│   ├── test_search_api.py
│   ├── test_health_api.py
│   └── test_deps.py
│
├── docs/
│   └── plans/                       # 设计与实现文档
│
├── pyproject.toml                   # 项目配置 (ruff, pytest)
├── requirements.txt                 # Python 依赖
├── Makefile                         # 构建/运行命令
├── .env.example                     # 环境变量模板
└── README.md                        # 项目说明
```

---

## 6. 关键设计决策

### 6.1 Zilliz REST API vs pymilvus SDK

**决策**：使用 Zilliz Cloud REST API 而非 pymilvus SDK。

**原因**：
- 零 Python SDK 依赖，减少安装复杂度
- REST API 足够覆盖所有需要的功能（search, insert, hybrid_search）
- 服务端 RRF 融合，减少客户端计算

**遗留问题**：`scripts/init_collection.py` 和 `scripts/batch_import.py` 仍然 import `src.storage.milvus_client`，但该文件已被替换为 `zilliz_client.py`。这些脚本当前**无法运行**。

### 6.2 服务端 RRF vs 客户端 RRF

**决策**：当前使用 Zilliz Cloud 的 `hybrid_search` API，RRF 融合在服务端完成。

**优势**：减少网络往返，减少客户端代码复杂度。

**劣势**：无法自定义权重（Zilliz API 的 RRF 不支持自定义 per-path 权重），无法添加 category 软过滤。

### 6.3 双路 vs 三路搜索

**决策**：实际实现为双路搜索（图像向量 + 文本向量）。

**原因**：Zilliz `hybrid_search` API 的 `search` 数组支持多路 ANN 搜索，BM25 作为全文索引可选附加。

**与设计文档的偏差**：设计文档描述的三路搜索（含独立 BM25 调用）在客户端 RRF 代码中有完整实现（`fusion.py`），但未被 `HybridSearchService` 调用。

### 6.4 并行化策略

```python
# 可并行的操作
image_encode ∥ desc_generate     # Step 1: ~800ms → ~600ms

# 必须串行的操作
desc_generate → text_encode      # Step 2: 依赖 VLM 输出

# 可并行的操作（未实现，因使用服务端 RRF）
image_search ∥ text_search ∥ bm25_search  # Step 3
```

### 6.5 图片压缩策略

`AliyunImageEncoder._compress_image()` 的渐进式压缩：

1. 质量递减：85 → 70 → 55 → 40 → 25
2. 尺寸递减：100% → 75% → 50% → 35% → 25%
3. 格式强制转 JPEG（去除 alpha 通道）

目标：压缩至 2.8MB 以下（阿里云 API 限制 3MB，留余量）。

---

## 7. 延迟模型

```
┌─────────────────────────────────────────────────────┐
│                    请求总延迟                         │
│                                                     │
│  ┌─────────────────────┐ ┌───────────────────┐     │
│  │  Step 1 (并行)       │ │  Step 2 (串行)     │     │
│  │  max(图像编码, 描述)  │ │  文本编码          │     │
│  │  ~600ms              │ │  ~100ms           │     │
│  └─────────┬───────────┘ └─────────┬─────────┘     │
│            │                         │               │
│  ┌─────────┴─────────────────────────┴───────────┐ │
│  │  Step 3: Zilliz hybrid_search                  │ │
│  │  ~50-150ms (取决于候选数量和网络)                │ │
│  └────────────────────┬───────────────────────────┘ │
│                       │                             │
│  ┌────────────────────┴───────────────────────────┐ │
│  │  Step 4: 响应构建 + 序列化                       │ │
│  │  ~5ms                                          │ │
│  └────────────────────┬───────────────────────────┘ │
│                       │                             │
│  预计总延迟: ~750-850ms (P50)                        │
│  预计总延迟: ~1.0-1.5s   (P95)                       │
└─────────────────────────────────────────────────────┘
```

---

## 8. 优化建议

### 8.1 效果优化（Search Quality）

#### P0 — 高优先级

| # | 优化项 | 现状 | 改进方案 | 预期收益 |
|---|--------|------|---------|---------|
| 1 | **恢复客户端 RRF 融合** | 当前依赖 Zilliz 服务端 RRF，无法自定义权重、无 category 软过滤 | 改为客户端三路搜索 + RRF 融合（`fusion.py` 已有完整实现），恢复 `category_hint` 功能 | 直接提升搜索精准度，尤其是指定品类时的相关性 |
| 2 | **修复 BM25 搜索路径** | 设计有 BM25 三路搜索，但未接入搜索流程 | 在 `hybrid_search.py` 中增加 BM25 搜索调用（查询 VLM 描述在 `LLMDescription` 字段的全文匹配） | 补充精确关键词匹配能力，对颜色/材质等属性搜索有提升 |
| 3 | **VLM Prompt 优化 — 中文场景** | 当前 prompt 为英文，生成英文描述；入库时商品名多为中文 | 增加中文 prompt 变体，或在现有英文 prompt 中要求附加中文翻译标签 | 解决语言不匹配导致的语义鸿沟，提升中文场景召回率 |

#### P1 — 中优先级

| # | 优化项 | 现状 | 改进方案 | 预期收益 |
|---|--------|------|---------|---------|
| 4 | **搜索权重可配置化** | `fusion.py` 中权重硬编码 (0.35/0.40/0.25) | 将权重移入 `Settings` / `FusionConfig`，支持通过环境变量或 API 参数调整 | 方便 A/B 测试和场景化调参 |
| 5 | **候选数量动态调整** | `candidate_multiplier` 固定为 3 | 根据数据量规模动态计算；小数据集可设为 1（全量搜索后排序） | 小数据集场景避免不必要的计算开销 |
| 6 | **统一 Prompt 一致性验证** | 入库 prompt 和搜索 prompt 应完全相同 | 在代码中提取 `UNIFIED_DESCRIPTION_PROMPT` 为共享常量，导入时和搜索时使用同一实例 | 确保描述在同一语义空间，避免"描述漂移" |
| 7 | **Zilliz 字段名规范化** | 数据库中 `discription` 存在拼写错误，代码中多处做兼容处理 (`r.get("discription", "") or r.get("description", "")`) | 数据迁移修正字段名，代码中统一使用 `description` | 消除隐患，降低维护成本 |
| 8 | **同义词扩展 / Query Expansion** | VLM 生成的描述是单次输出，可能遗漏关键特征 | 对 VLM 描述做同义词扩展（如 "grey" → "gray", "fabric" → "cloth", "sofa" → "couch"），生成多个变体查询 | 提升跨语言/跨表达方式的召回 |

#### P2 — 低优先级（进阶）

| # | 优化项 | 现状 | 改进方案 | 预期收益 |
|---|--------|------|---------|---------|
| 9 | **Cross-Encoder 重排序** | RRF 融合后直接返回 | 对 Top-50 候选用 Cross-Encoder (如 BGE-reranker) 做精排 | 显著提升 Top-K 精度 |
| 10 | **CLIP 端到端向量** | 图像和文本使用不同模型的向量 | 使用统一 CLIP 模型（如 OpenCLIP ViT-L-14）同时编码图像和文本 | 消除跨模型语义鸿沟 |
| 11 | **评估数据集 + 自动化评测** | 无评估数据集和自动化评测流程 | 构建标注数据集（200+ 条），实现 Recall@K, MRR, NDCG 自动评测脚本 | 量化衡量每次优化的效果 |
| 12 | **用户反馈闭环** | 无反馈机制 | 增加"点击/收藏/不相关"反馈收集，用于优化排序模型 | 长期搜索效果持续提升 |

---

### 8.2 性能优化（Performance）

#### P0 — 高优先级

| # | 优化项 | 现状 | 改进方案 | 预期收益 |
|---|--------|------|---------|---------|
| 1 | **Zilliz 客户端连接池** | `zilliz_client.py` 每次请求创建新的 `httpx.AsyncClient`（`async with httpx.AsyncClient() as client:`） | 复用单一 AsyncClient 实例（类似 encoders 的懒加载模式），在 `close()` 中清理 | 减少 TCP 握手开销，每次搜索节省 ~10-30ms |
| 2 | **修复脚本中断裂的 import** | `scripts/init_collection.py` import `src.storage.milvus_client`（不存在）；`scripts/batch_import.py` import `src.storage.milvus_client.MilvusClientWrapper`（不存在） | 更新为 `src.storage.zilliz_client.ZillizClient` 或提供独立的 collection 初始化 REST API 调用 | 恢复批量导入能力 |
| 3 | **图片压缩 CPU 优化** | `_compress_image()` 在事件循环中同步执行 PIL 图片处理（CPU 密集） | 使用 `asyncio.to_thread()` 包装图片压缩，避免阻塞事件循环 | 高并发时不阻塞其他请求 |
| 4 | **批量导入并行度优化** | `batch_import.py` 的 `batch_size=10`，batch 内并发但 batch 间串行 | 增加批次间并行（semaphore 控制总并发数），或使用 asyncio TaskGroup | 导入速度提升 2-3x |

#### P1 — 中优先级

| # | 优化项 | 现状 | 改进方案 | 预期收益 |
|---|--------|------|---------|---------|
| 5 | **VLM 描述缓存** | 每次搜索都调用 GLM-4V-Flash 生成描述 | 对相同/相似图片做描述缓存（基于图片 hash 或感知 hash），减少重复调用 | 相同图片搜索延迟从 ~800ms 降至 ~300ms |
| 6 | **HTTP 客户端超时精细化** | 所有编码器/生成器统一 30s 超时 | 按接口特性设置差异化超时：图片编码 15s（含压缩）、文本编码 5s、VLM 描述 20s、Zilliz 搜索 10s | 更快的故障发现和降级响应 |
| 7 | **请求大小限制** | 无图片大小限制，大文件直接传入 VLM | 在 API 层增加图片大小上限（如 10MB），提前拒绝超大请求 | 避免无效计算和内存浪费 |
| 8 | **日志和可观测性** | 仅 `print()` 输出，无结构化日志 | 引入 `structlog` 或 Python `logging`，增加 request_id 追踪、分步骤计时 | 生产环境问题排查效率提升 |
| 9 | **健康检查深度化** | `/health` 仅返回固定字符串 | 增加依赖服务连通性检查（Zilliz、智谱、阿里云），返回各组件状态 | 负载均衡器更准确的健康判断 |

#### P2 — 低优先级（进阶）

| # | 优化项 | 现状 | 改进方案 | 预期收益 |
|---|--------|------|---------|---------|
| 10 | **Embedding 结果缓存** | 每次搜索都调用外部 API 计算 embedding | 对高频查询做 embedding 缓存（Redis / 本地 LRU），减少 API 调用 | 降低 API 成本，减少延迟 |
| 11 | **优雅降级** | 任一外部 API 失败则整个搜索 500 | 实现 fallback 策略：VLM 超时 → 纯图像搜索；文本编码失败 → 纯图像向量搜索 | 提高服务可用性 |
| 12 | **批量导入断点续传** | `batch_import.py` 失败后需从头开始 | 记录处理进度（已处理的 SKU 列表），支持从断点恢复 | 大规模数据导入的可靠性 |
| 13 | **Prometheus 指标** | 无监控指标 | 增加搜索延迟 histogram、API 调用计数、错误率等 Prometheus 指标 | 生产环境可观测性 |
| 14 | **API 并发限流** | 无限流 | 增加 slowapi 或 FastAPI middleware 限流，保护后端服务 | 防止突发流量打垮服务 |

---

## 9. 代码质量问题

| # | 问题 | 位置 | 严重度 | 说明 |
|---|------|------|--------|------|
| 1 | **断裂的 import** | `scripts/init_collection.py` L13, `scripts/batch_import.py` L20 | 🔴 高 | 引用 `src.storage.milvus_client`，该文件不存在 |
| 2 | **字段拼写错误** | `schemas.py` L65, `search.py` L64, `hybrid_search.py` L25 | 🟡 中 | `discription` 应为 `description`，已在多处做兼容 |
| 3 | **未使用的代码** | `src/search/fusion.py` | 🟡 中 | 客户端 RRF 融合已实现但未被调用 |
| 4 | **`category_hint` 未生效** | `hybrid_search.py` L53 | 🟡 中 | 参数已接收但传递给了不存在的客户端 RRF，当前被忽略 |
| 5 | **CORS 全开放** | `main.py` L37-42 | 🟡 中 | `allow_origins=["*"]` 生产环境需限制 |
| 6 | **错误处理不统一** | `search.py` L46-48 | 🟢 低 | `import traceback; traceback.print_exc()` 应使用 logger |
| 7 | **缺少 `__init__.py` 导出** | 多个 `__init__.py` 为空 | 🟢 低 | 不影响功能，但不利于代码导航 |
| 8 | **SDK 和 src 代码重复** | `furniture_search/core/` vs `src/` | 🟡 中 | 两套几乎相同的代码，维护成本高 |
| 9 | **`SearchMeta.total_candidates` 语义** | `search.py` L76 | 🟢 低 | 当前等于 `len(results)`（已融合后的数量），不是候选池总量 |
