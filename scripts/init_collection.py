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
