# 家具搜索 SDK - 使用总结

## 📦 SDK 创建完成

我已经为你创建了一个可复用的家具搜索 SDK，支持在其他项目中集成家具相似搜索功能。

## 📁 项目结构

```
f_search_demo/
├── furniture_search/        # SDK 包（新增）
│   ├── client.py           # Python 客户端
│   ├── server.py           # HTTP 服务（供其他语言调用）
│   ├── config.py           # 配置管理
│   ├── core/               # 核心服务
│   │   ├── hybrid_search.py
│   │   ├── encoders/
│   │   ├── generators/
│   │   └── storage/
│   ├── examples/           # 使用示例
│   │   ├── python_usage.py
│   │   ├── http_client_examples.md  # Java/Go/JS 等语言示例
│   │   └── test_sdk.py     # 测试脚本
│   ├── README.md           # SDK 详细文档
│   ├── QUICKSTART.md       # 快速开始指南
│   ├── pyproject.toml      # 包配置
│   ├── .env.example        # 环境变量示例
│   ├── Dockerfile          # Docker 配置
│   └── start.sh            # 启动脚本
├── src/                    # 原始项目（保持不变）
├── scripts/                # 初始化脚本
├── frontend/               # Vue 3 前端
└── SDK_README.md           # SDK 概述（新增）
```

## 🚀 快速使用

### 1. 安装 SDK

```bash
make install
# 或
cd furniture_search && pip install -e .
```

### 2. 配置环境变量

```bash
cd furniture_search
cp .env.example .env
# 编辑 .env 填入实际的 API 密钥
```

### 3. 启动 HTTP 服务（模式 2 - 推荐）

```bash
# 使用 Makefile
make start

# 或使用启动脚本
cd furniture_search && ./start.sh

# 或直接运行
python -m furniture_search.server
```

### 4. 调用 API

#### Python 示例

```python
import requests
import base64

# 读取图片
with open('design.jpg', 'rb') as f:
    image_bytes = f.read()
image_base64 = base64.b64encode(image_bytes).decode()

# 调用 API
response = requests.post(
    'http://localhost:8000/search',
    json={
        'image_base64': image_base64,
        'top_k': 20,
        'category_hint': 'sofa'
    }
)

result = response.json()
print(result['data']['query_description'])
```

#### Java 示例

```java
import java.net.http.HttpClient;
import java.util.Base64;

// 参见 furniture_search/examples/http_client_examples.md
```

#### Go 示例

```go
import "net/http"

// 参见 furniture_search/examples/http_client_examples.md
```

#### JavaScript/Node.js 示例

```javascript
const axios = require('axios');
const fs = require('fs');

const imageBase64 = fs.readFileSync('design.jpg').toString('base64');

axios.post('http://localhost:8000/search', {
  image_base64: imageBase64,
  top_k: 20
}).then(response => {
  console.log(response.data);
});
```

#### cURL 示例

```bash
IMAGE_BASE64=$(base64 -i design.jpg)
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_BASE64\", \"top_k\": 20}"
```

## 📖 API 文档

### POST /search

**请求体：**

```json
{
  "image_base64": "base64_encoded_image_data",
  "top_k": 20,
  "category_hint": "sofa"
}
```

**响应体：**

```json
{
  "success": true,
  "data": {
    "query_description": "现代简约风格的三人沙发...",
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

## 🔧 Makefile 命令

```bash
make help      # 显示帮助信息
make install   # 安装 SDK
make test      # 运行测试
make start     # 启动服务
make dev       # 开发模式启动
make docker-build   # 构建 Docker 镜像
make docker-run     # 运行 Docker 容器
make clean     # 清理临时文件
make format    # 格式化代码
make lint      # 代码检查
```

## 🐳 Docker 部署

```bash
# 构建镜像
make docker-build

# 运行容器
make docker-run

# 或手动运行
docker run -d \
  -p 8000:8000 \
  --env-file furniture_search/.env \
  furniture-search-sdk
```

## 📚 文档

- **SDK 概述**: [SDK_README.md](SDK_README.md)
- **SDK 详细文档**: [furniture_search/README.md](furniture_search/README.md)
- **快速开始**: [furniture_search/QUICKSTART.md](furniture_search/QUICKSTART.md)
- **其他语言调用示例**: [furniture_search/examples/http_client_examples.md](furniture_search/examples/http_client_examples.md)
- **Python 直接使用**: [furniture_search/examples/python_usage.py](furniture_search/examples/python_usage.py)

## ✨ 特性

- ✅ 支持任意语言通过 HTTP API 调用
- ✅ Python 项目可直接使用（无需 HTTP）
- ✅ 三路混合搜索（图像向量 + 文本向量 + BM25）
- ✅ RRF 融合排序
- ✅ AI 描述生成
- ✅ Docker 支持
- ✅ 完整的使用示例（Python/Java/Go/JavaScript/cURL）

## 🎯 下一步

1. 配置环境变量（复制 `.env.example` 到 `.env`）
2. 运行测试：`make test`
3. 启动服务：`make start`
4. 访问 API 文档：http://localhost:8000/docs
5. 根据你的语言选择相应的调用示例

## 💡 提示

- SDK 服务独立运行，不影响原项目
- 原项目的 `src/` 目录保持不变
- `scripts/` 中的初始化脚本仍需在原项目目录下运行
- 查看 API 文档（http://localhost:8000/docs）获取更多细节
