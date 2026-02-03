# 家具图像混合搜索系统设计文档

> 创建时间：2026-02-03  
> 状态：设计完成，待实施

## 1. 概述

### 1.1 业务背景

用户使用 AI 生成家具设计效果图，当用户对某件家具设计满意时，希望能在家具库中找到外观最相近的真实商品。

### 1.2 核心挑战

| 挑战 | 描述 |
|------|------|
| Domain Gap | AI 渲染图 vs 真实商品照片，视觉风格差异大 |
| 语义匹配 | 用户期望不是像素级相似，而是"看起来像" |
| 多维度特征 | 颜色、材质、风格、造型、尺寸都可能影响匹配 |

### 1.3 解决方案概述

采用 **三路混合搜索** 架构：
1. **图像 embedding 搜索**：捕捉视觉细节
2. **文本 embedding 搜索**：捕捉语义概念（通过图→文转换弥合 domain gap）
3. **关键词 BM25 搜索**：捕捉精确特征词

三路结果通过 **RRF (Reciprocal Rank Fusion)** 融合排序。

### 1.4 技术选型

| 组件 | 选择 | 说明 |
|------|------|------|
| **开发语言** | Python 3.10+ | 主要开发语言 |
| **Web 框架** | FastAPI | 高性能异步 API 框架 |
| **向量数据库** | Zilliz Cloud | Milvus 托管服务，免运维 |
| **描述生成 (VLM)** | 智谱 GLM-4.6V-Flash | 高性价比多模态模型 |
| **文本 Embedding** | 智谱 Embedding-3 | 2048 维，中文优化 |
| **图像 Embedding** | 阿里云 Multimodal-Embedding | 1024 维，跨模态统一向量 |
| **图像 Embedding (备选)** | OpenCLIP 自部署 | 无 API 调用成本，需 GPU |

### 1.5 系统定位

本系统是一个 **Python 后端服务**，提供：
- **搜索 API**：接收用户上传的 AI 设计图，返回最相似的真实商品列表
- **数据导入脚本**：批量导入商品数据到向量数据库

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              客户端                                      │
│                         (AI 生成的家具设计图)                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           API Gateway                                    │
│                         POST /search/image                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Search Service                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Input Processing                              │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │   │
│  │  │  Image Encoder  │  │  VLM 描述生成    │  │  Text Encoder   │  │   │
│  │  │  (阿里云/CLIP)  │  │  (GLM-4.6V-Flash)│  │  (智谱Embedding)│  │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │   │
│  │           │                    │                    │           │   │
│  │           ▼                    ▼                    ▼           │   │
│  │      image_emb            description           text_emb        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Parallel Search                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│  │  │ Image Vector│  │ Text Vector │  │   BM25      │              │   │
│  │  │   Search    │  │   Search    │  │   Search    │              │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │   │
│  │         │                │                │                      │   │
│  │         └────────────────┼────────────────┘                      │   │
│  │                          ▼                                       │   │
│  │                 ┌─────────────────┐                              │   │
│  │                 │   RRF Fusion    │                              │   │
│  │                 └────────┬────────┘                              │   │
│  │                          │                                       │   │
│  └──────────────────────────┼──────────────────────────────────────┘   │
│                             ▼                                          │
│                      Top K Results                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Milvus Cloud (Zilliz Cloud)                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Collection: products                          │   │
│  │  ┌─────────┬──────────┬─────────────────┬───────────────────┐   │   │
│  │  │   id    │ category │ description_*   │ image/text_emb    │   │   │
│  │  └─────────┴──────────┴─────────────────┴───────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
离线数据处理（商品入库）:
商品图片 ──→ CLIP ──→ image_embedding ──┐
    │                                    │
    └──→ VLM ──→ description_ai ──→ Text Encoder ──→ text_embedding
                      │
                      └──→ 存入 Milvus

在线搜索:
用户设计图 ──→ CLIP ──→ query_image_emb ──→ Image Vector Search ──┐
    │                                                              │
    └──→ VLM ──→ query_description ──→ Text Encoder ──→ Text Vector Search ──┼──→ RRF ──→ Results
                      │                                            │
                      └──→ BM25 Search ────────────────────────────┘
```

---

## 3. 数据模型

### 3.1 Milvus Collection Schema

```python
from pymilvus import CollectionSchema, FieldSchema, DataType

fields = [
    # 主键
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    
    # 商品基础信息
    FieldSchema(name="product_code", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
    
    # 描述字段（支持 BM25 搜索）
    FieldSchema(name="description_human", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="description_ai", dtype=DataType.VARCHAR, max_length=4096),
    
    # 向量字段
    FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),  # 阿里云 Multimodal-Embedding
    FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=2048),  # 智谱 Embedding-3
    
    # 元数据
    FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="created_at", dtype=DataType.INT64),
]

schema = CollectionSchema(fields=fields, description="Furniture product collection")
```

### 3.2 索引配置

```python
# 图像向量索引
image_index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 16,
        "efConstruction": 256
    }
}

# 文本向量索引
text_index_params = {
    "index_type": "HNSW", 
    "metric_type": "COSINE",
    "params": {
        "M": 16,
        "efConstruction": 256
    }
}

# BM25 全文索引（Milvus 2.4+）
# 在 description_ai 和 description_human 字段上启用
```

### 3.3 Zilliz Cloud 配置

使用 Zilliz Cloud 托管服务，无需自建 Milvus 集群。

**环境变量配置**：

```bash
# .env

# Zilliz Cloud (Milvus)
ZILLIZ_CLOUD_URI=https://xxx.api.gcp-us-west1.zillizcloud.com
ZILLIZ_CLOUD_TOKEN=your_api_token
ZILLIZ_CLOUD_COLLECTION=products

# 智谱 AI（描述生成 + 文本 Embedding）
ZHIPU_API_KEY=your_zhipu_api_key

# 阿里云（图像 Embedding）
ALIYUN_DASHSCOPE_API_KEY=your_aliyun_api_key
```

**连接示例**：

```python
from pymilvus import MilvusClient

def get_milvus_client():
    return MilvusClient(
        uri=os.getenv("ZILLIZ_CLOUD_URI"),
        token=os.getenv("ZILLIZ_CLOUD_TOKEN")
    )
```

**云服务优势**：
- 免运维，自动扩缩容
- 内置高可用与备份
- 按用量计费，适合初期验证

### 3.4 商品数据格式

#### 3.4.1 输入数据格式

商品数据通过 CSV 或 JSON 格式导入，支持两种方式：

**方式一：CSV 格式**

```csv
product_code,category,description_human,image_path
SF-001,sofa,"三人位布艺沙发，北欧风格，灰色",/data/images/SF-001.jpg
TB-002,table,"实木餐桌，现代简约，原木色",/data/images/TB-002.jpg
CH-003,chair,"休闲单椅，轻奢风格，深蓝色皮质",/data/images/CH-003.jpg
```

**方式二：JSON 格式**

```json
{
  "products": [
    {
      "product_code": "SF-001",
      "category": "sofa",
      "description_human": "三人位布艺沙发，北欧风格，灰色",
      "image_path": "/data/images/SF-001.jpg"
    },
    {
      "product_code": "TB-002",
      "category": "table",
      "description_human": "实木餐桌，现代简约，原木色",
      "image_path": "/data/images/TB-002.jpg"
    }
  ]
}
```

**方式三：目录结构自动解析**

```
data/
├── sofa/
│   ├── SF-001.jpg
│   ├── SF-002.jpg
│   └── ...
├── table/
│   ├── TB-001.jpg
│   └── ...
└── chair/
    ├── CH-001.jpg
    └── ...
```

目录名作为 `category`，文件名作为 `product_code`，`description_human` 留空。

#### 3.4.2 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `product_code` | 是 | 商品唯一编码 |
| `category` | 是 | 商品品类（如 sofa, table, chair） |
| `description_human` | 否 | 人工编写的描述，可留空 |
| `image_path` | 是 | 商品图片路径（本地路径或 URL） |

### 3.5 数据导入脚本

#### 3.5.1 脚本功能

```
scripts/
├── init_collection.py      # 初始化 Zilliz Cloud collection
├── batch_import.py         # 批量导入商品数据
└── validate_data.py        # 验证数据格式
```

#### 3.5.2 初始化 Collection 脚本

```python
# scripts/init_collection.py
"""初始化 Zilliz Cloud Collection"""

import os
from pymilvus import MilvusClient, DataType

def create_collection():
    client = MilvusClient(
        uri=os.getenv("ZILLIZ_CLOUD_URI"),
        token=os.getenv("ZILLIZ_CLOUD_TOKEN")
    )
    
    collection_name = os.getenv("ZILLIZ_CLOUD_COLLECTION", "products")
    
    # 如果已存在则跳过
    if client.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists, skipping...")
        return
    
    # 创建 schema
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("product_code", DataType.VARCHAR, max_length=64)
    schema.add_field("category", DataType.VARCHAR, max_length=128)
    schema.add_field("description_human", DataType.VARCHAR, max_length=2048)
    schema.add_field("description_ai", DataType.VARCHAR, max_length=4096)
    schema.add_field("image_embedding", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("text_embedding", DataType.FLOAT_VECTOR, dim=2048)
    schema.add_field("image_url", DataType.VARCHAR, max_length=512)
    schema.add_field("created_at", DataType.INT64)
    
    # 创建索引
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
    
    # 创建 collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    
    print(f"Collection '{collection_name}' created successfully!")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    create_collection()
```

#### 3.5.3 批量导入脚本

```python
# scripts/batch_import.py
"""批量导入商品数据"""

import os
import json
import csv
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class ProductInput:
    product_code: str
    category: str
    image_path: str
    description_human: Optional[str] = None

class BatchImporter:
    def __init__(
        self,
        milvus_client,
        description_generator,
        image_encoder,
        text_encoder,
        batch_size: int = 10
    ):
        self.milvus = milvus_client
        self.desc_gen = description_generator
        self.img_encoder = image_encoder
        self.text_encoder = text_encoder
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
        """从目录结构导入（目录名=category，文件名=product_code）"""
        products = []
        data_path = Path(data_dir)
        
        for category_dir in data_path.iterdir():
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            
            for image_file in category_dir.glob('*.jpg'):
                products.append(ProductInput(
                    product_code=image_file.stem,
                    category=category,
                    image_path=str(image_file),
                    description_human=''
                ))
            
            for image_file in category_dir.glob('*.png'):
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
        
        for i in tqdm(range(0, len(products), self.batch_size)):
            batch = products[i:i + self.batch_size]
            records = await self._process_batch(batch)
            
            # 插入到 Milvus
            self.milvus.insert(
                collection_name=os.getenv("ZILLIZ_CLOUD_COLLECTION"),
                data=records
            )
        
        print(f"Successfully imported {len(products)} products!")
    
    async def _process_batch(self, batch: List[ProductInput]) -> List[Dict]:
        """处理一批商品：生成描述和 embedding"""
        import time
        
        records = []
        
        # 并行处理每个商品
        tasks = [self._process_single(p) for p in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for product, result in zip(batch, results):
            if isinstance(result, Exception):
                print(f"Error processing {product.product_code}: {result}")
                continue
            records.append(result)
        
        return records
    
    async def _process_single(self, product: ProductInput) -> Dict:
        """处理单个商品"""
        import time
        
        # 读取图片
        with open(product.image_path, 'rb') as f:
            image_bytes = f.read()
        
        # 并行生成描述和图像 embedding
        desc_task = self.desc_gen.generate(image_bytes)
        img_emb_task = asyncio.to_thread(self.img_encoder.encode, image_bytes)
        
        description_ai, image_embedding = await asyncio.gather(desc_task, img_emb_task)
        
        # 生成文本 embedding
        text_embedding = self.text_encoder.encode(description_ai)
        
        return {
            "product_code": product.product_code,
            "category": product.category,
            "description_human": product.description_human or "",
            "description_ai": description_ai,
            "image_embedding": image_embedding.tolist(),
            "text_embedding": text_embedding.tolist(),
            "image_url": product.image_path,  # 实际部署时替换为 CDN URL
            "created_at": int(time.time())
        }

# CLI 入口
async def main():
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='批量导入商品数据')
    parser.add_argument('--input', '-i', required=True, help='输入文件或目录路径')
    parser.add_argument('--format', '-f', choices=['csv', 'json', 'dir'], default='csv',
                        help='输入格式: csv, json, dir')
    parser.add_argument('--batch-size', '-b', type=int, default=10, help='批处理大小')
    args = parser.parse_args()
    
    # 初始化组件
    from src.storage.milvus_client import get_milvus_client
    from src.generators.description import ZhipuDescriptionGenerator
    from src.encoders.image_encoder import AliyunImageEncoder
    from src.encoders.text_encoder import ZhipuTextEncoder
    
    importer = BatchImporter(
        milvus_client=get_milvus_client(),
        description_generator=ZhipuDescriptionGenerator(os.getenv("ZHIPU_API_KEY")),
        image_encoder=AliyunImageEncoder(os.getenv("ALIYUN_DASHSCOPE_API_KEY")),
        text_encoder=ZhipuTextEncoder(os.getenv("ZHIPU_API_KEY")),
        batch_size=args.batch_size
    )
    
    if args.format == 'csv':
        await importer.import_from_csv(args.input)
    elif args.format == 'json':
        await importer.import_from_json(args.input)
    else:
        await importer.import_from_directory(args.input)

if __name__ == "__main__":
    asyncio.run(main())
```

#### 3.5.4 使用示例

```bash
# 初始化 collection
python scripts/init_collection.py

# 从 CSV 导入
python scripts/batch_import.py -i data/products.csv -f csv

# 从 JSON 导入
python scripts/batch_import.py -i data/products.json -f json

# 从目录结构导入
python scripts/batch_import.py -i data/images/ -f dir --batch-size 20
```

---

## 4. 核心组件设计

### 4.1 统一描述生成 Prompt

**关键原则**：商品图片和用户设计图必须使用**完全相同**的 prompt，确保生成的描述在同一语义空间。

```python
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
"""
```

### 4.2 描述生成服务（智谱 GLM-4.6V-Flash）

```python
from abc import ABC, abstractmethod
from typing import Optional
import httpx

class DescriptionGenerator(ABC):
    @abstractmethod
    async def generate(self, image_bytes: bytes) -> str:
        pass

class ZhipuDescriptionGenerator(DescriptionGenerator):
    """智谱 GLM-4.6V-Flash 描述生成器"""
    
    def __init__(self, api_key: str, model: str = "glm-4v-flash"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient()
    
    async def generate(self, image_bytes: bytes) -> str:
        import base64
        image_b64 = base64.b64encode(image_bytes).decode()
        
        response = await self.client.post(
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
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
                "max_tokens": 500
            },
            timeout=30.0
        )
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
```

### 4.3 Embedding 服务

```python
from abc import ABC, abstractmethod
import numpy as np
import httpx

class ImageEncoder(ABC):
    @abstractmethod
    def encode(self, image_bytes: bytes) -> np.ndarray:
        pass

class TextEncoder(ABC):
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        pass

# 阿里云 Multimodal-Embedding 图像编码器
class AliyunImageEncoder(ImageEncoder):
    """阿里云多模态向量服务 - 图像编码器"""
    
    def __init__(self, api_key: str, model: str = "multimodal-embedding-v1"):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
    
    def encode(self, image_bytes: bytes) -> np.ndarray:
        import base64
        image_b64 = base64.b64encode(image_bytes).decode()
        
        response = httpx.post(
            self.endpoint,
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
            },
            timeout=30.0
        )
        
        result = response.json()
        return np.array(result["output"]["embeddings"][0]["embedding"])

# 智谱 Embedding-3 文本编码器
class ZhipuTextEncoder(TextEncoder):
    """智谱 Embedding-3 文本编码器"""
    
    def __init__(self, api_key: str, model: str = "embedding-3"):
        self.api_key = api_key
        self.model = model
    
    def encode(self, text: str) -> np.ndarray:
        response = httpx.post(
            "https://open.bigmodel.cn/api/paas/v4/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "input": text
            },
            timeout=30.0
        )
        
        result = response.json()
        return np.array(result["data"][0]["embedding"])

# 备选：OpenCLIP 自部署图像编码器
class OpenCLIPImageEncoder(ImageEncoder):
    """OpenCLIP 自部署图像编码器（备选方案）"""
    
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai"):
        import open_clip
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
    
    def encode(self, image_bytes: bytes) -> np.ndarray:
        from PIL import Image
        import io
        import torch
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy().flatten()
```

### 4.4 混合搜索服务

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import asyncio

@dataclass
class SearchResult:
    product_id: int
    product_code: str
    category: str
    description_ai: str
    image_url: str
    score: float
    # 调试用：各路搜索的排名
    debug_ranks: Optional[Dict[str, int]] = None

@dataclass
class SearchConfig:
    # 各路搜索的权重
    weight_image: float = 0.35
    weight_text_vector: float = 0.40
    weight_bm25: float = 0.25
    
    # RRF 参数
    rrf_k: int = 60
    
    # 搜索参数
    candidate_multiplier: int = 3  # 每路搜索返回 top_k * multiplier 条
    
    # Category 软过滤加权
    category_boost: float = 1.2  # 匹配 category 时的加权系数

class HybridSearchService:
    def __init__(
        self,
        milvus_client,
        image_encoder: ImageEncoder,
        text_encoder: TextEncoder,
        description_generator: DescriptionGenerator,
        config: SearchConfig = None
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
    ) -> List[SearchResult]:
        """
        执行三路混合搜索
        
        Args:
            image_bytes: 输入图片的二进制数据
            top_k: 返回结果数量
            category_hint: 可选的品类提示，用于软过滤加权
        
        Returns:
            按相关性排序的搜索结果列表
        """
        # 1. 并行处理输入
        image_emb_task = asyncio.to_thread(self.image_encoder.encode, image_bytes)
        description_task = self.desc_generator.generate(image_bytes)
        
        image_emb, description = await asyncio.gather(image_emb_task, description_task)
        text_emb = self.text_encoder.encode(description)
        
        # 2. 并行执行三路搜索
        candidate_limit = top_k * self.config.candidate_multiplier
        
        image_search_task = self._search_by_image(image_emb, candidate_limit)
        text_vector_task = self._search_by_text_vector(text_emb, candidate_limit)
        bm25_task = self._search_by_bm25(description, candidate_limit)
        
        image_results, text_vector_results, bm25_results = await asyncio.gather(
            image_search_task, text_vector_task, bm25_task
        )
        
        # 3. RRF 融合
        fused_results = self._rrf_fusion(
            image_results=image_results,
            text_vector_results=text_vector_results,
            bm25_results=bm25_results,
            category_hint=category_hint
        )
        
        # 4. 返回 Top K
        return fused_results[:top_k]
    
    async def _search_by_image(self, embedding, limit: int) -> List[Dict]:
        """图像向量搜索"""
        results = self.milvus.search(
            collection_name="products",
            data=[embedding.tolist()],
            anns_field="image_embedding",
            limit=limit,
            output_fields=["product_code", "category", "description_ai", "image_url"]
        )
        return self._parse_milvus_results(results)
    
    async def _search_by_text_vector(self, embedding, limit: int) -> List[Dict]:
        """文本向量搜索"""
        results = self.milvus.search(
            collection_name="products",
            data=[embedding.tolist()],
            anns_field="text_embedding",
            limit=limit,
            output_fields=["product_code", "category", "description_ai", "image_url"]
        )
        return self._parse_milvus_results(results)
    
    async def _search_by_bm25(self, query_text: str, limit: int) -> List[Dict]:
        """BM25 全文搜索"""
        # Milvus 2.4+ 支持 BM25
        results = self.milvus.search(
            collection_name="products",
            data=[query_text],
            anns_field="description_ai",  # 或使用复合字段
            limit=limit,
            output_fields=["product_code", "category", "description_ai", "image_url"],
            search_params={"metric_type": "BM25"}
        )
        return self._parse_milvus_results(results)
    
    def _parse_milvus_results(self, results) -> List[Dict]:
        """解析 Milvus 返回结果"""
        parsed = []
        for hits in results:
            for rank, hit in enumerate(hits, 1):
                parsed.append({
                    "id": hit.id,
                    "rank": rank,
                    "score": hit.score,
                    "product_code": hit.entity.get("product_code"),
                    "category": hit.entity.get("category"),
                    "description_ai": hit.entity.get("description_ai"),
                    "image_url": hit.entity.get("image_url")
                })
        return parsed
    
    def _rrf_fusion(
        self,
        image_results: List[Dict],
        text_vector_results: List[Dict],
        bm25_results: List[Dict],
        category_hint: Optional[str] = None
    ) -> List[SearchResult]:
        """RRF 融合排序"""
        
        # 建立 ID -> 排名 的映射
        def build_rank_map(results: List[Dict]) -> Dict[int, int]:
            return {r["id"]: r["rank"] for r in results}
        
        image_ranks = build_rank_map(image_results)
        text_vector_ranks = build_rank_map(text_vector_results)
        bm25_ranks = build_rank_map(bm25_results)
        
        # 收集所有候选 ID 及其元数据
        all_candidates = {}
        for results in [image_results, text_vector_results, bm25_results]:
            for r in results:
                if r["id"] not in all_candidates:
                    all_candidates[r["id"]] = r
        
        # 计算 RRF 分数
        k = self.config.rrf_k
        fused_scores = []
        
        for product_id, meta in all_candidates.items():
            score = 0.0
            ranks = {}
            
            # 图像搜索贡献
            if product_id in image_ranks:
                ranks["image"] = image_ranks[product_id]
                score += self.config.weight_image / (k + image_ranks[product_id])
            
            # 文本向量搜索贡献
            if product_id in text_vector_ranks:
                ranks["text_vector"] = text_vector_ranks[product_id]
                score += self.config.weight_text_vector / (k + text_vector_ranks[product_id])
            
            # BM25 搜索贡献
            if product_id in bm25_ranks:
                ranks["bm25"] = bm25_ranks[product_id]
                score += self.config.weight_bm25 / (k + bm25_ranks[product_id])
            
            # Category 软过滤加权
            if category_hint and meta.get("category"):
                if self._category_match(category_hint, meta["category"]):
                    score *= self.config.category_boost
            
            fused_scores.append(SearchResult(
                product_id=product_id,
                product_code=meta["product_code"],
                category=meta["category"],
                description_ai=meta["description_ai"],
                image_url=meta["image_url"],
                score=score,
                debug_ranks=ranks
            ))
        
        # 按分数降序排序
        fused_scores.sort(key=lambda x: x.score, reverse=True)
        return fused_scores
    
    def _category_match(self, hint: str, actual: str) -> bool:
        """
        判断 category 是否匹配（支持模糊匹配）
        可以扩展为更复杂的语义匹配
        """
        hint_lower = hint.lower()
        actual_lower = actual.lower()
        return hint_lower in actual_lower or actual_lower in hint_lower
```

---

## 5. API 设计

### 5.1 搜索接口

```yaml
POST /api/v1/search/image
Content-Type: multipart/form-data

Request:
  - image: file (required) - 输入图片
  - top_k: int (optional, default=20) - 返回结果数量
  - category_hint: string (optional) - 品类提示

Response:
  {
    "success": true,
    "data": {
      "query_description": "...",  # 生成的描述，便于调试
      "results": [
        {
          "product_id": 12345,
          "product_code": "SF-001",
          "category": "sofa",
          "description_ai": "[COLOR] Dark gray...",
          "image_url": "https://...",
          "score": 0.0234,
          "rank": 1
        },
        ...
      ]
    },
    "meta": {
      "took_ms": 523,
      "total_candidates": 180
    }
  }
```

### 5.2 商品入库接口

```yaml
POST /api/v1/products
Content-Type: multipart/form-data

Request:
  - image: file (required) - 商品图片
  - product_code: string (required) - 商品编码
  - category: string (required) - 商品品类
  - description_human: string (optional) - 人工描述

Response:
  {
    "success": true,
    "data": {
      "product_id": 12345,
      "description_ai": "...",  # 自动生成的描述
    }
  }
```

---

## 6. 性能优化

### 6.1 延迟优化

| 环节 | 优化策略 |
|------|----------|
| VLM 描述生成 | GLM-4.6V-Flash 已是高性价比选择，可缓存常见图片类型的描述模板 |
| 三路搜索 | 并行执行，不串行等待 |
| Embedding 计算 | 图像 embedding 和文本 embedding 并行计算 |
| Milvus 索引 | 使用 HNSW 索引，ef_search 根据延迟要求调整 |

### 6.2 预期延迟分解

```
GLM-4.6V-Flash 描述生成:  ~600ms (可并行)
阿里云图像 Embedding:     ~100ms (可并行)
智谱文本 Embedding:       ~100ms (依赖描述生成)
Milvus 搜索x3:            ~50ms  (并行)
RRF 融合:                 ~10ms
──────────────────────────────
总延迟:                   ~800ms (P95)
```

### 6.3 吞吐量优化

- **批量入库**：商品入库时批量处理 embedding 计算
- **异步队列**：入库请求放入消息队列，异步处理
- **GPU 资源池**：CLIP 编码器使用 GPU 资源池

---

## 7. 评估与调优

### 7.1 评估指标

| 指标 | 定义 | 目标 |
|------|------|------|
| **Recall@K** | Top K 结果中包含相关商品的比例 | Recall@20 ≥ 0.8 |
| **MRR** | 第一个相关结果的排名倒数平均 | MRR ≥ 0.5 |
| **NDCG@K** | 考虑排名位置的相关性得分 | NDCG@20 ≥ 0.7 |

### 7.2 评估数据集构建

```python
# 评估数据集格式
evaluation_set = [
    {
        "query_image": "path/to/ai_design_1.jpg",
        "relevant_products": ["SF-001", "SF-002"],  # 人工标注的相关商品
        "highly_relevant": ["SF-001"]  # 最相关的商品
    },
    ...
]
```

**标注建议**：
- 每个查询图片标注 3-5 个相关商品
- 区分"高度相关"和"一般相关"
- 样本量建议 ≥ 200 条

### 7.3 调参策略

```python
# 需要调优的参数
tuning_params = {
    # RRF 权重
    "weight_image": [0.2, 0.3, 0.35, 0.4],
    "weight_text_vector": [0.3, 0.4, 0.45, 0.5],
    "weight_bm25": [0.15, 0.2, 0.25, 0.3],
    
    # RRF k 参数
    "rrf_k": [20, 40, 60, 80],
    
    # Category 加权
    "category_boost": [1.0, 1.1, 1.2, 1.3]
}

# 使用网格搜索或贝叶斯优化
```

### 7.4 A/B 测试

上线后通过 A/B 测试验证：
- 对照组：纯向量搜索
- 实验组：三路混合搜索

跟踪指标：
- 用户点击率 (CTR)
- 用户满意度（如有反馈机制）
- 搜索结果后的转化率

---

## 8. 错误处理

### 8.1 降级策略

```python
class SearchServiceWithFallback(HybridSearchService):
    async def search(self, image_bytes: bytes, top_k: int = 20, **kwargs):
        try:
            return await super().search(image_bytes, top_k, **kwargs)
        except VLMTimeoutError:
            # VLM 超时：降级为纯图像搜索
            logger.warning("VLM timeout, falling back to image-only search")
            return await self._fallback_image_only_search(image_bytes, top_k)
        except MilvusConnectionError:
            # Milvus 连接失败：返回缓存结果或错误
            logger.error("Milvus connection failed")
            raise ServiceUnavailableError("Search service temporarily unavailable")
```

### 8.2 错误码定义

| 错误码 | 含义 | 处理建议 |
|--------|------|----------|
| 4001 | 图片格式不支持 | 返回支持的格式列表 |
| 4002 | 图片过大 | 提示压缩或限制尺寸 |
| 5001 | VLM 服务不可用 | 自动降级 + 告警 |
| 5002 | Milvus 服务不可用 | 返回错误 + 告警 |
| 5003 | Embedding 服务不可用 | 自动降级 + 告警 |

---

## 9. 监控与告警

### 9.1 关键监控指标

```yaml
# Prometheus metrics
search_request_total: Counter  # 搜索请求总数
search_latency_seconds: Histogram  # 搜索延迟分布
search_error_total: Counter  # 按错误类型分类

vlm_latency_seconds: Histogram  # VLM 生成延迟
embedding_latency_seconds: Histogram  # Embedding 计算延迟
milvus_search_latency_seconds: Histogram  # Milvus 搜索延迟

# 业务指标
search_result_count: Histogram  # 返回结果数量分布
search_empty_result_total: Counter  # 空结果请求数
```

### 9.2 告警规则

```yaml
alerts:
  - name: HighSearchLatency
    condition: search_latency_seconds_p95 > 2s
    severity: warning
    
  - name: VLMHighErrorRate
    condition: vlm_error_rate > 5%
    severity: critical
    
  - name: MilvusConnectionFailed
    condition: milvus_connection_errors > 0
    severity: critical
```

---

## 10. 项目结构

```
furniture_search_demo/
├── docs/
│   └── plans/
│       └── 2026-02-03-furniture-hybrid-search-design.md  # 本文档
├── src/
│   ├── __init__.py
│   ├── config.py                 # 配置管理
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic 数据模型
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── image_encoder.py      # 阿里云/OpenCLIP 图像编码器
│   │   └── text_encoder.py       # 智谱文本编码器
│   ├── generators/
│   │   ├── __init__.py
│   │   └── description.py        # 智谱 GLM-4.6V-Flash 描述生成
│   ├── search/
│   │   ├── __init__.py
│   │   ├── hybrid_search.py      # 混合搜索核心逻辑
│   │   └── fusion.py             # RRF 融合算法
│   ├── storage/
│   │   ├── __init__.py
│   │   └── milvus_client.py      # Zilliz Cloud 客户端封装
│   └── api/
│       ├── __init__.py
│       ├── main.py               # FastAPI 应用入口
│       ├── deps.py               # 依赖注入
│       └── routes/
│           ├── __init__.py
│           ├── search.py         # 搜索接口
│           └── health.py         # 健康检查接口
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # pytest fixtures
│   ├── test_encoders.py
│   ├── test_search.py
│   └── test_api.py
├── scripts/
│   ├── init_collection.py        # 初始化 Zilliz Cloud collection
│   ├── batch_import.py           # 批量导入商品
│   ├── validate_data.py          # 验证数据格式
│   └── evaluate.py               # 搜索效果评估脚本
├── data/
│   └── sample/                   # 示例数据
│       ├── products.csv
│       └── images/
├── .env.example                  # 环境变量模板
├── requirements.txt              # Python 依赖
├── pyproject.toml                # 项目配置
├── Dockerfile                    # Docker 镜像构建
└── README.md                     # 项目说明
```

### 10.1 requirements.txt

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

# CLI
tqdm>=4.66.0

# Optional: OpenCLIP (if self-hosting image encoder)
# open-clip-torch>=2.24.0
# torch>=2.2.0
```

### 10.2 FastAPI 应用入口

```python
# src/api/main.py
"""FastAPI 应用入口"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.routes import search, health
from src.api.deps import init_services, cleanup_services

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化服务
    await init_services()
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
    allow_origins=["*"],  # 生产环境应限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, tags=["Health"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
```

### 10.3 启动命令

```bash
# 开发环境
uvicorn src.api.main:app --reload --port 8000

# 生产环境
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker
docker build -t furniture-search .
docker run -p 8000:8000 --env-file .env furniture-search
```

---

## 11. 实施计划概要

### Phase 1: 基础搭建
- [ ] Zilliz Cloud 账号配置与 collection 创建
- [ ] 阿里云 Multimodal-Embedding 图像编码器集成
- [ ] 智谱 GLM-4.6V-Flash 描述生成集成
- [ ] 智谱 Embedding-3 文本编码器集成

### Phase 2: 核心功能
- [ ] 商品入库流程（图片 → 描述 → embedding → 存储）
- [ ] 三路搜索实现
- [ ] RRF 融合算法实现
- [ ] API 接口开发

### Phase 3: 优化与评估
- [ ] 构建评估数据集
- [ ] 参数调优
- [ ] 性能优化
- [ ] 错误处理与降级

### Phase 4: 上线准备
- [ ] 监控与告警配置
- [ ] A/B 测试框架
- [ ] 文档完善
- [ ] 部署上线

---

## 12. 参考资料

### 论文
- [Hybrid Search: Balancing the Blend](https://arxiv.org/html/2508.01405) - 混合搜索权衡分析
- [FashionKLIP](https://aclanthology.org/2023.acl-industry.16/) - 电商图文检索
- [CommerceMM](https://arxiv.org/abs/2202.07247) - 阿里巴巴电商多模态框架

### 技术文档
- [Milvus Hybrid Search](https://milvus.io/docs/hybrid_search.md)
- [Zilliz Cloud Documentation](https://docs.zilliz.com/)
- [Azure AI Search: Hybrid Retrieval](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/azure-ai-search-outperforming-vector-search-with-hybrid-retrieval-and-reranking/3929167)

### 模型与 API
- [智谱 GLM-4V 系列](https://open.bigmodel.cn/dev/howuse/glm-4v)
- [智谱 Embedding-3](https://open.bigmodel.cn/dev/howuse/embedding)
- [阿里云 Multimodal-Embedding](https://www.alibabacloud.com/help/zh/model-studio/multimodal-embedding-api-reference)
- [OpenCLIP](https://github.com/mlfoundations/open_clip) (备选自部署方案)
