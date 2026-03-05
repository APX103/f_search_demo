# 快速开始

## 5 分钟启动服务

### 1. 安装依赖

```bash
cd furniture_search
pip install -e .
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入实际的 API 密钥
```

### 3. 测试 SDK

```bash
python examples/test_sdk.py
```

### 4. 启动服务

```bash
# 方式 1: 使用启动脚本
./start.sh

# 方式 2: 直接运行
python -m furniture_search.server

# 方式 3: 使用 uvicorn
uvicorn furniture_search.server:app --reload
```

### 5. 调用 API

```bash
# 使用 cURL
IMAGE_BASE64=$(base64 -i path/to/your/image.jpg)
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64\": \"$IMAGE_BASE64\",
    \"top_k\": 20
  }"
```

访问 `http://localhost:8000/docs` 查看交互式 API 文档。

## Docker 部署

```bash
# 构建镜像
cd furniture_search
docker build -t furniture-search-sdk .

# 运行容器
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  furniture-search-sdk
```

## 其他语言调用

查看 [examples/http_client_examples.md](examples/http_client_examples.md) 获取更多语言的调用示例：

- ✅ Python
- ✅ Java
- ✅ Go
- ✅ JavaScript/Node.js
- ✅ cURL

## 常见问题

### 1. 如何获取 API 密钥？

- **阿里云 DashScope**: https://dashscope.console.aliyun.com/
- **智谱 AI**: https://open.bigmodel.cn/
- **Zilliz Cloud**: https://cloud.zilliz.com/

### 2. 如何初始化数据库？

使用原项目的初始化脚本：

```bash
cd ..  # 回到项目根目录
python scripts/init_collection.py
python scripts/batch_import.py -i data/products.csv -f csv
```

### 3. 如何查看日志？

服务启动后会输出日志到控制台，可以使用日志工具：

```bash
# 启动服务并记录日志
python -m furniture_search.server 2>&1 | tee search.log

# 或者使用 systemd 管理（生产环境推荐）
```

### 4. 如何扩展服务？

修改 `furniture_search/server.py` 或 `furniture_search/client.py`。

### 5. 如何配置搜索参数？

在 `.env` 文件中修改：

```bash
SEARCH_RRF_K=60                    # RRF 参数
SEARCH_CANDIDATE_MULTIPLIER=3      # 候选数量倍数
```

## 性能优化建议

1. **增加并发**: 使用多 worker 启动

```bash
uvicorn furniture_search.server:app --workers 4
```

2. **使用 CDN**: 对图片 URL 使用 CDN 加速

3. **缓存结果**: 对相同图片使用 Redis 缓存

4. **异步调用**: 使用异步 HTTP 客户端调用 API

## 下一步

- 查看 [README.md](README.md) 了解更多功能
- 查看 [examples/http_client_examples.md](examples/http_client_examples.md) 学习如何用其他语言调用
- 查看 [examples/python_usage.py](examples/python_usage.py) 学习如何在 Python 中直接使用 SDK
