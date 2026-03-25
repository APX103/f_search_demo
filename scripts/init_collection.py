# scripts/init_collection.py
"""初始化 Zilliz Cloud Collection (via REST API)"""

import os
import sys

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv

COLLECTION_SCHEMA = {
    "collectionName": "products",
    "schema": {
        "autoId": True,
        "enabledDynamicField": False,
        "fields": [
            {"fieldName": "id", "dataType": "Int64", "isPrimary": True, "autoID": True},
            {"fieldName": "sku", "dataType": "VarChar", "elementTypeParams": {"max_length": "64"}},
            {"fieldName": "name", "dataType": "VarChar", "elementTypeParams": {"max_length": "256"}},
            {"fieldName": "category", "dataType": "VarChar", "elementTypeParams": {"max_length": "128"}},
            {"fieldName": "price", "dataType": "VarChar", "elementTypeParams": {"max_length": "64"}},
            {"fieldName": "discription", "dataType": "VarChar", "elementTypeParams": {"max_length": "2048"}},
            {"fieldName": "LLMDescription", "dataType": "VarChar", "elementTypeParams": {"max_length": "4096"}, "enableAnalyzer": True, "analyzerParams": {"type": "chinese"}},
            {"fieldName": "imageVector", "dataType": "FloatVector", "elementTypeParams": {"dim": "1024"}},
            {"fieldName": "vector", "dataType": "FloatVector", "elementTypeParams": {"dim": "2048"}},
            {"fieldName": "url", "dataType": "VarChar", "elementTypeParams": {"max_length": "512"}},
            {"fieldName": "imageUrl", "dataType": "VarChar", "elementTypeParams": {"max_length": "512"}},
        ]
    },
    "indexParams": [
        {"fieldName": "imageVector", "indexType": "HNSW", "metricType": "COSINE", "params": {"M": 16, "efConstruction": 256}},
        {"fieldName": "vector", "indexType": "HNSW", "metricType": "COSINE", "params": {"M": 16, "efConstruction": 256}},
    ]
}


def create_collection():
    """创建 Zilliz Cloud Collection via REST API"""
    load_dotenv()

    uri = os.getenv("ZILLIZ_CLOUD_URI")
    token = os.getenv("ZILLIZ_CLOUD_TOKEN")
    collection_name = os.getenv("ZILLIZ_CLOUD_COLLECTION", "products")

    if not uri or not token:
        print("Error: ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_TOKEN must be set")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    def api_post(endpoint: str, payload: dict) -> dict:
        url = f"{uri.rstrip('/')}/{endpoint.lstrip('/')}"
        resp = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    # 检查是否已存在
    describe = api_post("v2/vectordb/collections/describe", {"collectionName": collection_name})
    if describe.get("code") == 0:
        print(f"Collection '{collection_name}' already exists.")
        response = input("Do you want to drop and recreate it? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        api_post("v2/vectordb/collections/drop", {"collectionName": collection_name})
        print(f"Dropped existing collection '{collection_name}'")

    # 创建 collection
    body = dict(COLLECTION_SCHEMA)
    body["collectionName"] = collection_name

    result = api_post("v2/vectordb/collections/create", body)
    if result.get("code") != 0:
        print(f"Error creating collection: {result}")
        sys.exit(1)

    print(f"Collection '{collection_name}' created successfully!")

    # 显示 collection 信息
    info = api_post("v2/vectordb/collections/describe", {"collectionName": collection_name})
    data = info.get("data", info)
    print(f"\nCollection info:")
    print(f"  - Name: {data.get('collectionName', collection_name)}")
    fields = data.get("fields", [])
    print(f"  - Fields: {len(fields)}")
    for field in fields:
        print(f"    - {field.get('fieldName')}: {field.get('dataType')}")


if __name__ == "__main__":
    create_collection()
