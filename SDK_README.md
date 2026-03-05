# 家具搜索 SDK

这是一个可复用的家具搜索 SDK，支持在其他项目中集成家具相似搜索功能。

## 📁 目录说明

- **`furniture_search/`**: SDK 核心包
  - `client.py`: 简化的客户端接口
  - `server.py`: HTTP 服务器（供其他语言调用）
  - `core/`: 核心搜索服务
  
- **`src/`**: 原项目的完整代码（FastAPI 应用）

- **`scripts/`**: 数据库初始化和批量导入脚本

- **`frontend/`**: Vue 3 单页应用

## 🚀 快速使用

### 方式 1: HTTP 服务（推荐，支持其他语言）

```bash
cd furniture_search
cp .env.example .env
# 编辑 .env 填入 API 密钥

# 启动服务
python -m furniture_search.server

# 调用 API
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "...", "top_k": 20}'
```

详细文档：[furniture_search/README.md](furniture_search/README.md)

### 方式 2: Python 项目直接使用

```python
from furniture_search import FurnitureSearchClient

client = FurnitureSearchClient(
    zilliz_endpoint="https://...",
    zilliz_token="...",
    aliyun_api_key="...",
    zhipu_api_key="..."
)

results, description = await client.search(
    image_bytes=image_data,
    top_k=20
)
```

详细文档：[furniture_search/examples/python_usage.py](furniture_search/examples/python_usage.py)

## 🌐 支持的语言

- ✅ Python
- ✅ Java
- ✅ Go
- ✅ JavaScript/Node.js
- ✅ 任意支持 HTTP 的语言

查看更多示例：[furniture_search/examples/http_client_examples.md](furniture_search/examples/http_client_examples.md)

## 📦 SDK 特性

- **三路混合搜索**：图像向量 + 文本向量 + BM25 全文
- **RRF 融合排序**：智能融合多路搜索结果
- **AI 描述生成**：自动生成商品描述
- **HTTP API 模式**：支持任意语言调用
- **零依赖部署**：无需安装向量库 SDK

## 🔧 项目结构

```
f_search_demo/
├── furniture_search/        # SDK 包（可复用）
│   ├── client.py           # Python 客户端
│   ├── server.py           # HTTP 服务
│   ├── core/               # 核心服务
│   └── examples/           # 使用示例
├── src/                    # 原始项目
│   ├── api/                # FastAPI 路由
│   ├── encoders/           # 编码器
│   ├── generators/         # 生成器
│   ├── storage/            # 存储客户端
│   └── search/             # 搜索服务
├── scripts/                # 初始化脚本
├── frontend/               # Vue 3 前端
└── README.md               # 本文件
```

## 📖 相关文档

- [原项目 README](README.md): 家具搜索系统的完整说明
- [SDK README](furniture_search/README.md): SDK 详细文档
- [快速开始](furniture_search/QUICKSTART.md): 5 分钟快速上手
- [其他语言调用示例](furniture_search/examples/http_client_examples.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License
