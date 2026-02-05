# 家具图像混合搜索系统

基于 AI 设计图的家具相似搜索服务，使用三路混合搜索（图像向量、文本向量、BM25）和 RRF 融合算法。

## 功能特性

- **三路混合搜索**：结合图像向量、文本向量和 BM25 全文搜索
- **RRF 融合排序**：智能融合多路搜索结果
- **AI 描述生成**：使用 VLM 自动生成商品描述
- **品类加权**：支持品类提示的软过滤加权
- **Web 前端**：开箱即用的搜索界面

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Frontend (Vue 3)                               │
│                         浏览器直接打开 index.html                            │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ HTTP POST (multipart/form-data)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FastAPI Application                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Routes    │  │    Deps     │  │   Models    │  │     Middleware      │ │
│  │  /health    │  │  DI 容器    │  │   Schemas   │  │   CORS / Logging    │ │
│  │  /search    │  │  生命周期   │  │   Pydantic  │  │                     │ │
│  └──────┬──────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────┼───────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HybridSearchService                                 │
│                                                                             │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│   │  ImageEncoder   │   │  TextEncoder    │   │ DescGenerator   │          │
│   │  阿里云多模态    │   │  智谱 Embedding │   │ 智谱 GLM-4V     │          │
│   │  1024 维向量    │   │  2048 维向量    │   │ 图像→文本描述    │          │
│   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘          │
│            │                     │                     │                    │
│            └─────────────────────┴─────────────────────┘                    │
│                                  │                                          │
│                                  ▼                                          │
│                    ┌─────────────────────────┐                              │
│                    │     Hybrid Search       │                              │
│                    │   (三路并行 + RRF)       │                              │
│                    └────────────┬────────────┘                              │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Zilliz Cloud (Milvus)                               │
│                                                                             │
│   Collection: furniture_products                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  id │ sku │ name │ category │ imageVector │ vector │ LLMDescription │   │
│   │     │     │      │          │  (1024D)    │(2048D) │   (BM25 索引)   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 搜索流程

```
                              用户上传图片
                                   │
                                   ▼
              ┌────────────────────┴────────────────────┐
              │                                         │
              ▼                                         ▼
     ┌────────────────┐                      ┌────────────────┐
     │  图像编码器     │                      │  描述生成器     │
     │  Image → 1024D │                      │  Image → Text  │
     └───────┬────────┘                      └───────┬────────┘
             │                                       │
             │                                       ▼
             │                              ┌────────────────┐
             │                              │  文本编码器     │
             │                              │  Text → 2048D  │
             │                              └───────┬────────┘
             │                                       │
             └───────────────────┬───────────────────┘
                                 │
                                 ▼
              ┌──────────────────┴──────────────────┐
              │      Zilliz Cloud Hybrid Search     │
              │  ┌──────────────────────────────┐   │
              │  │ 三路并行搜索 (单次 API 调用)   │   │
              │  │  • imageVector (ANN)         │   │
              │  │  • vector (ANN)              │   │
              │  │  • LLMDescription (BM25)     │   │
              │  └──────────────┬───────────────┘   │
              │                 │                   │
              │                 ▼                   │
              │  ┌──────────────────────────────┐   │
              │  │   RRF 服务端融合排序          │   │
              │  │   score = Σ(1/(k + rank_i))  │   │
              │  └──────────────────────────────┘   │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼
                          返回 Top-K 结果
```

## RRF 融合算法

**Reciprocal Rank Fusion (RRF)** 是一种简单高效的多路召回融合算法：

```
                    n
RRF_score(d) = Σ  ────────────
                   k + rank_i(d)
                  i=1

其中：
  • d = 候选文档
  • k = 平滑常数 (默认 60)
  • rank_i(d) = 文档 d 在第 i 路搜索中的排名
```

**三路搜索权重分配：**

```
┌─────────────────────┬────────┬─────────────────────────────────┐
│       搜索路径       │  权重  │             说明                 │
├─────────────────────┼────────┼─────────────────────────────────┤
│ 图像向量 (ANN)       │  35%   │ 视觉相似度，外观/颜色/形状      │
│ 文本向量 (ANN)       │  40%   │ 语义相似度，风格/用途/材质      │
│ BM25 全文搜索        │  25%   │ 关键词匹配，精确属性命中        │
└─────────────────────┴────────┴─────────────────────────────────┘
```

## 技术栈

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| **前端** | Vue 3 (CDN) | 单文件 HTML，零构建 |
| **后端** | FastAPI | 异步 API 框架 |
| **向量库** | Zilliz Cloud REST API | 零 SDK 依赖，直接调 HTTP |
| **图像编码** | 阿里云 Multimodal-Embedding | 1024 维视觉向量 |
| **文本编码** | 智谱 Embedding-3 | 2048 维语义向量 |
| **描述生成** | 智谱 GLM-4V-Flash | VLM 图像理解 |

## 项目结构

```
furniture_search_demo/
├── frontend/
│   └── index.html        # Vue 3 单页前端
├── src/
│   ├── api/
│   │   ├── main.py       # FastAPI 入口
│   │   ├── deps.py       # 依赖注入
│   │   └── routes/       # 路由模块
│   │       ├── health.py
│   │       └── search.py
│   ├── encoders/
│   │   ├── image_encoder.py   # 图像向量编码
│   │   └── text_encoder.py    # 文本向量编码
│   ├── generators/
│   │   └── description.py     # VLM 描述生成
│   ├── search/
│   │   ├── hybrid_search.py   # 混合搜索服务
│   │   └── fusion.py          # RRF 融合算法
│   ├── storage/
│   │   └── zilliz_client.py   # Zilliz Cloud REST API 客户端
│   ├── models/
│   │   └── schemas.py         # Pydantic 模型
│   └── config.py              # 配置管理
├── scripts/
│   ├── init_collection.py     # 初始化向量集合
│   └── batch_import.py        # 批量导入商品
├── tests/                     # 单元测试
├── docs/plans/                # 设计文档
├── requirements.txt
└── .env.example
```

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

### 5. 启动后端服务

```bash
# 开发模式
uvicorn src.api.main:app --reload --port 8000

# 生产模式
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 6. 打开前端

直接用浏览器打开 `frontend/index.html`，或启动静态服务：

```bash
cd frontend && python -m http.server 3000
# 访问 http://localhost:3000
```

## API 参考

### 图像搜索

```bash
POST /api/v1/search/image

# 参数
- image: 图片文件 (multipart/form-data)
- top_k: 返回数量 (1-100, 默认 20)
- category_hint: 品类提示 (可选)

# 示例
curl -X POST "http://localhost:8000/api/v1/search/image" \
  -F "image=@design.jpg" \
  -F "top_k=20" \
  -F "category_hint=sofa"
```

**响应示例：**

```json
{
  "success": true,
  "data": {
    "query_description": "现代简约风格的三人沙发，灰色布艺材质...",
    "results": [
      {
        "product_id": 12345,
        "sku": "SOFA-001",
        "name": "北欧简约三人沙发",
        "category": "沙发",
        "price": "¥2,999",
        "image_url": "https://...",
        "score": 0.0156,
        "rank": 1
      }
    ]
  },
  "meta": {
    "took_ms": 156,
    "total_candidates": 20
  }
}
```

### 健康检查

```bash
GET /health
```

## 测试

```bash
pytest tests/ -v
```

## License

MIT
