# src/storage/milvus_client.py
"""Zilliz Cloud (Milvus) 客户端封装"""

from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker


class MilvusClientWrapper:
    """Milvus 客户端封装类"""
    
    def __init__(self, uri: str, token: str, collection_name: str):
        self.client = MilvusClient(uri=uri, token=token)
        self.collection_name = collection_name
        self.uri = uri
        self.token = token
    
    def has_collection(self) -> bool:
        """检查 collection 是否存在"""
        return self.client.has_collection(self.collection_name)
    
    def create_collection(self, schema, index_params) -> None:
        """创建 collection"""
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
    
    def insert(self, data: List[Dict[str, Any]]) -> None:
        """插入数据"""
        self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
    
    def hybrid_search(
        self,
        search_requests: List[Dict],
        output_fields: List[str],
        limit: int,
        rrf_k: int = 60
    ) -> List[Dict]:
        """
        执行混合搜索（多路搜索 + RRF 融合）
        
        Args:
            search_requests: 搜索请求列表，每个包含:
                - field_name: 搜索字段
                - data: 搜索数据（向量或文本）
                - search_type: "vector" 或 "text"
                - limit: 每路返回数量
            output_fields: 输出字段
            limit: 最终返回数量
            rrf_k: RRF 参数 k
        
        Returns:
            融合后的搜索结果
        """
        # 构建 AnnSearchRequest 列表
        reqs = []
        for req in search_requests:
            if req["search_type"] == "vector":
                search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
            else:  # text/BM25
                search_params = {"metric_type": "BM25"}
            
            ann_req = AnnSearchRequest(
                data=[req["data"]],
                anns_field=req["field_name"],
                param=search_params,
                limit=req.get("limit", limit)
            )
            reqs.append(ann_req)
        
        # 使用 RRF Ranker 进行融合
        ranker = RRFRanker(k=rrf_k)
        
        # 执行混合搜索
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=limit,
            output_fields=output_fields
        )
        
        return self._parse_hybrid_results(results)
    
    def search_by_vector(
        self,
        field_name: str,
        vector: List[float],
        limit: int,
        output_fields: List[str],
        search_params: Optional[Dict] = None
    ) -> List[Dict]:
        """单路向量搜索（保留用于简单场景）"""
        params = search_params or {"metric_type": "COSINE", "params": {"ef": 64}}
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field=field_name,
            limit=limit,
            output_fields=output_fields,
            search_params=params
        )
        
        return self._parse_results(results)
    
    def search_by_text(
        self,
        field_name: str,
        query_text: str,
        limit: int,
        output_fields: List[str]
    ) -> List[Dict]:
        """单路 BM25 全文搜索（保留用于简单场景）"""
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_text],
            anns_field=field_name,
            limit=limit,
            output_fields=output_fields,
            search_params={"metric_type": "BM25"}
        )
        
        return self._parse_results(results)
    
    def _parse_results(self, results) -> List[Dict]:
        """解析单路搜索结果"""
        parsed = []
        for hits in results:
            for rank, hit in enumerate(hits, 1):
                parsed.append({
                    "id": hit["id"],
                    "rank": rank,
                    "score": hit["distance"],
                    **hit["entity"]
                })
        return parsed
    
    def _parse_hybrid_results(self, results) -> List[Dict]:
        """解析混合搜索结果"""
        parsed = []
        for rank, hit in enumerate(results[0], 1):
            parsed.append({
                "id": hit["id"],
                "rank": rank,
                "score": hit["distance"],
                **hit["entity"]
            })
        return parsed
    
    def close(self) -> None:
        """关闭连接"""
        self.client.close()


def get_collection_schema(client: MilvusClient):
    """
    获取 collection schema
    
    字段说明（与 Zilliz Cloud 现有 schema 对齐）：
    - vector: 文本向量 (2048维，智谱 Embedding-3)
    - imageVector: 图像向量 (1024维，阿里云 Multimodal-Embedding)
    - name: 商品名称
    - url: 商品页面 URL
    - imageUrl: 商品图片 URL
    - price: 价格
    - description: 人工描述
    - sku: 商品 SKU
    - LLMDescription: AI 生成的描述（用于 BM25 搜索）
    - category: 商品类别
    """
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("sku", DataType.VARCHAR, max_length=64)
    schema.add_field("name", DataType.VARCHAR, max_length=128)
    schema.add_field("category", DataType.VARCHAR, max_length=64)
    schema.add_field("price", DataType.VARCHAR, max_length=64)
    schema.add_field("description", DataType.VARCHAR, max_length=2048)
    schema.add_field("LLMDescription", DataType.VARCHAR, max_length=4096,
                     enable_analyzer=True, analyzer_params={"type": "chinese"})
    schema.add_field("imageVector", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2048)
    schema.add_field("url", DataType.VARCHAR, max_length=512)
    schema.add_field("imageUrl", DataType.VARCHAR, max_length=512)
    
    return schema


def get_index_params(client: MilvusClient):
    """获取索引参数"""
    index_params = client.prepare_index_params()
    
    index_params.add_index(
        field_name="imageVector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256}
    )
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256}
    )
    index_params.add_index(
        field_name="LLMDescription",
        index_type="AUTOINDEX",
        index_name="LLMDescription_bm25"
    )
    
    return index_params
