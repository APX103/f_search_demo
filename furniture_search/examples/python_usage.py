"""示例：如何使用 FurnitureSearch SDK"""

import asyncio
import base64
from pathlib import Path

from furniture_search import FurnitureSearchClient, SearchConfig


async def search_example():
    """直接在 Python 项目中使用 SDK"""
    
    # 初始化客户端
    client = FurnitureSearchClient(
        zilliz_endpoint="https://your-cluster.cloud.zilliz.com",
        zilliz_token="your-api-token",
        aliyun_api_key="your-aliyun-key",
        zhipu_api_key="your-zhipu-key",
        collection_name="furniture_products",
        config=SearchConfig(rrf_k=60, candidate_multiplier=3)
    )
    
    try:
        # 读取图片文件
        image_path = Path("path/to/your/design.jpg")
        image_bytes = image_path.read_bytes()
        
        # 执行搜索
        results, query_description = await client.search(
            image_bytes=image_bytes,
            top_k=20,
            category_hint="sofa"
        )
        
        # 显示结果
        print(f"Generated description: {query_description}")
        print(f"Top results:")
        for result in results[:5]:
            print(f"  - {result.name} (score: {result.score:.4f})")
            
    finally:
        await client.close()


async def http_server_example():
    """启动 HTTP 服务器供其他语言调用"""
    
    print("Starting HTTP server at http://localhost:8000")
    print("API endpoint: POST /search")
    print("""
Example request:
curl -X POST "http://localhost:8000/search" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_base64": "base64_encoded_image_data",
    "top_k": 20,
    "category_hint": "sofa"
  }'
""")


if __name__ == "__main__":
    print("=== SDK 使用示例 ===")
    print()
    print("1. 直接 Python 使用示例:")
    asyncio.run(search_example())
    print()
    print("2. HTTP 服务器启动:")
    asyncio.run(http_server_example())
