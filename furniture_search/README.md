# Furniture Search SDK

基于 AI 设计图的家具相似搜索服务 SDK，支持在任意语言后端项目中集成。

## 特性

- ✅ **图像相似搜索**：上传家具设计图，返回最相似的商品
- ✅ **三路混合搜索**：图像向量 + 文本向量 + BM25 全文
- ✅ **RRF 融合排序**：智能融合多路搜索结果
- ✅ **AI 描述生成**：自动生成商品描述
- ✅ **HTTP API 模式**：启动独立服务，支持任意语言调用
- ✅ **零依赖部署**：只需 Python 3.10+，无需向量库 SDK

## 快速开始

### 1. 安装 SDK

```bash
pip install -e .
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
ZILLIZ_CLOUD_URI=https://your-cluster.cloud.zilliz.com
ZILLIZ_CLOUD_TOKEN=your-api-token
ALIYUN_DASHSCOPE_API_KEY=your-aliyun-key
ZHIPU_API_KEY=your-zhipu-key
ZILLIZ_CLOUD_COLLECTION=furniture_products
```

### 3. 启动 HTTP 服务

```bash
# 方式 1: 直接运行 Python
python -m furniture_search.server

# 方式 2: 使用 pip 安装的命令
furniture-search-server

# 方式 3: 使用 uvicorn
uvicorn furniture_search.server:app --host 0.0.0.0 --port 8000
```

服务启动后，访问 `http://localhost:8000/docs` 查看 API 文档。

### 4. 调用 API

#### cURL 示例

```bash
# 读取图片并编码为 base64
IMAGE_BASE64=$(base64 -i design.jpg)

# 发送搜索请求
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64\": \"$IMAGE_BASE64\",
    \"top_k\": 20,
    \"category_hint\": \"sofa\"
  }"
```

#### Python 示例

```python
import requests
import base64
from pathlib import Path

def search_furniture(image_path: str) -> dict:
    """调用搜索服务"""
    image_bytes = Path(image_path).read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode()
    
    response = requests.post(
        "http://localhost:8000/search",
        json={
            "image_base64": image_base64,
            "top_k": 20,
            "category_hint": "sofa"
        }
    )
    
    return response.json()

result = search_furniture("design.jpg")
print(f"Query: {result['data']['query_description']}")
```

#### JavaScript/Node.js 示例

```javascript
const axios = require('axios');
const fs = require('fs');

const imageBase64 = fs.readFileSync('design.jpg').toString('base64');

axios.post('http://localhost:8000/search', {
  image_base64: imageBase64,
  top_k: 20,
  category_hint: 'sofa'
}).then(response => {
  console.log('Query:', response.data.data.query_description);
  console.log('Results:', response.data.data.results);
});
```

## API 文档

### POST /search

基于图片的家具相似搜索。

**请求体：**

```json
{
  "image_base64": "base64_encoded_image_data",
  "top_k": 20,
  "category_hint": "sofa"
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| image_base64 | string | 是 | Base64 编码的图片数据 |
| top_k | int | 否 | 返回结果数量 (1-100, 默认 20) |
| category_hint | string | 否 | 品类提示 (如 "sofa", "chair" 等) |

**响应体：**

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
        "description": "商品描述...",
        "llm_description": "AI生成的描述...",
        "url": "https://...",
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

### GET /health

健康检查接口。

**响应：**

```json
{
  "status": "ok",
  "service": "furniture-search-sdk"
}
```

## 其他语言调用示例

SDK 提供 HTTP API，支持任意语言调用。详细示例请查看 [examples/http_client_examples.md](examples/http_client_examples.md)：

- ✅ **Java**: 使用 `java.net.http.HttpClient`
- ✅ **Go**: 使用 `net/http`
- ✅ **Python**: 使用 `requests`
- ✅ **JavaScript/Node.js**: 使用 `axios`
- ✅ **cURL**: 命令行工具

## Python 项目直接使用

如果需要在 Python 项目中直接使用 SDK（不经过 HTTP），可以这样用：

```python
from furniture_search import FurnitureSearchClient, SearchConfig
import asyncio

async def search_example():
    client = FurnitureSearchClient(
        zilliz_endpoint="https://your-cluster.cloud.zilliz.com",
        zilliz_token="your-api-token",
        aliyun_api_key="your-aliyun-key",
        zhipu_api_key="your-zhipu-key",
        collection_name="furniture_products",
        config=SearchConfig(rrf_k=60, candidate_multiplier=3)
    )
    
    try:
        image_bytes = Path("design.jpg").read_bytes()
        results, query_description = await client.search(
            image_bytes=image_bytes,
            top_k=20,
            category_hint="sofa"
        )
        
        for result in results[:5]:
            print(f"- {result.name} (score: {result.score:.4f})")
    finally:
        await client.close()

asyncio.run(search_example())
```

## 项目结构

```
furniture_search/
├── __init__.py           # SDK 入口
├── client.py             # 简化的客户端接口
├── config.py             # 配置管理
├── server.py             # HTTP 服务器（供其他语言调用）
├── core/                 # 核心服务
│   ├── hybrid_search.py  # 混合搜索服务
│   ├── encoders/         # 图像/文本编码器
│   ├── generators/       # AI 描述生成器
│   └── storage/          # Zilliz 客户端
└── examples/             # 使用示例
    ├── python_usage.py
    └── http_client_examples.md
```

## 技术栈

- **Web 框架**: FastAPI
- **向量搜索**: Zilliz Cloud REST API
- **图像编码**: 阿里云 Multimodal-Embedding (1024 维)
- **文本编码**: 智谱 Embedding-3 (2048 维)
- **描述生成**: 智谱 GLM-4V-Flash

## 配置说明

### 搜索参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rrf_k` | 60 | RRF 融合算法的平滑常数 |
| `candidate_multiplier` | 3 | 候选数量倍数 (top_k * multiplier) |

### 环境变量

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `ZILLIZ_CLOUD_URI` | Zilliz 集群 endpoint | `https://xxx.cloud.zilliz.com` |
| `ZILLIZ_CLOUD_TOKEN` | Zilliz API token | `your-api-token` |
| `ZILLIZ_CLOUD_COLLECTION` | 集合名称 | `furniture_products` |
| `ALIYUN_DASHSCOPE_API_KEY` | 阿里云 API key | `sk-xxx` |
| `ZHIPU_API_KEY` | 智谱 API key | `xxx.xxx` |
| `SEARCH_RRF_K` | RRF 参数 k | `60` |
| `SEARCH_CANDIDATE_MULTIPLIER` | 候选倍数 | `3` |

## 错误处理

### 400 Bad Request

```json
{
  "detail": "Invalid base64 image data"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Search failed: Error message..."
}
```

## 性能指标

- **搜索延迟**: 150-300ms（取决于图片大小和网络）
- **并发支持**: 建议部署多实例 (uvicorn --workers N)
- **内存占用**: ~200MB per worker

## License

MIT License
