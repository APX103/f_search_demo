# src/storage/zilliz_client.py
"""Zilliz Cloud REST API 客户端"""

from typing import List, Dict, Any, Optional
import httpx


class ZillizClient:
    """Zilliz Cloud REST API 客户端"""
    
    def __init__(self, endpoint: str, token: str, collection_name: str):
        """
        Args:
            endpoint: Zilliz Cloud 集群 endpoint (如 https://xxx.cloud.zilliz.com)
            token: API Token
            collection_name: 集合名称
        """
        self.endpoint = endpoint.rstrip('/')
        self.token = token
        self.collection_name = collection_name
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    async def hybrid_search(
        self,
        vector_searches: List[Dict[str, Any]],
        limit: int = 20,
        rrf_k: int = 60,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        多路向量混合搜索 + RRF 融合
        
        Args:
            vector_searches: 向量搜索列表，每项包含:
                - data: 向量数据 (List[float])
                - anns_field: 向量字段名 (如 "vector", "imageVector")
                - limit: 每路返回数量
            limit: 最终返回数量
            rrf_k: RRF 参数 k
            output_fields: 输出字段列表
        
        Returns:
            搜索结果列表
        """
        # 构建搜索请求
        search_list = []
        for search in vector_searches:
            search_list.append({
                "data": [search["data"]],  # API 要求二维数组
                "annsField": search["anns_field"],
                "limit": search.get("limit", limit),
                "outputFields": output_fields or ["*"]
            })
        
        payload = {
            "collectionName": self.collection_name,
            "search": search_list,
            "rerank": {
                "strategy": "rrf",
                "params": {"k": rrf_k}
            },
            "limit": limit,
            "outputFields": output_fields or ["*"]
        }
        
        url = f"{self.endpoint}/v2/vectordb/entities/hybrid_search"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
        
        if result.get("code") != 0:
            raise Exception(f"Zilliz API error: {result.get('message', 'Unknown error')}")
        
        return self._parse_results(result.get("data", []))
    
    async def search(
        self,
        vector: List[float],
        anns_field: str,
        limit: int = 20,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        单路向量搜索
        
        Args:
            vector: 查询向量
            anns_field: 向量字段名
            limit: 返回数量
            output_fields: 输出字段列表
        
        Returns:
            搜索结果列表
        """
        payload = {
            "collectionName": self.collection_name,
            "data": [vector],
            "annsField": anns_field,
            "limit": limit,
            "outputFields": output_fields or ["*"]
        }
        
        url = f"{self.endpoint}/v2/vectordb/entities/search"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
        
        if result.get("code") != 0:
            raise Exception(f"Zilliz API error: {result.get('message', 'Unknown error')}")
        
        return self._parse_results(result.get("data", []))
    
    async def insert(self, data: List[Dict[str, Any]]) -> Dict:
        """
        插入数据
        
        Args:
            data: 数据列表
        
        Returns:
            插入结果
        """
        payload = {
            "collectionName": self.collection_name,
            "data": data
        }
        
        url = f"{self.endpoint}/v2/vectordb/entities/insert"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
        
        if result.get("code") != 0:
            raise Exception(f"Zilliz API error: {result.get('message', 'Unknown error')}")
        
        return result
    
    def _parse_results(self, data: List[Dict]) -> List[Dict]:
        """解析搜索结果"""
        results = []
        for rank, item in enumerate(data, 1):
            result = {
                "id": item.get("primary_key") or item.get("id"),
                "primary_key": item.get("primary_key") or item.get("id"),
                "rank": rank,
                "score": item.get("distance", 0),
            }
            # 添加其他字段
            for key, value in item.items():
                if key not in ("id", "distance", "primary_key"):
                    result[key] = value
            results.append(result)
        return results
