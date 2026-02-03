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
