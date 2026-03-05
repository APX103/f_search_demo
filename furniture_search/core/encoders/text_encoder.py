# src/encoders/text_encoder.py
"""文本编码器实现"""

from typing import Protocol
import numpy as np
import httpx


class TextEncoder(Protocol):
    """文本编码器协议"""
    
    async def encode(self, text: str) -> np.ndarray:
        """将文本编码为向量"""
        ...


class ZhipuTextEncoder:
    """智谱 Embedding-3 文本编码器"""
    
    ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    
    def __init__(
        self,
        api_key: str,
        model: str = "embedding-3",
        timeout: float = 30.0
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端（懒加载）"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def encode(self, text: str) -> np.ndarray:
        """
        将文本编码为 2048 维向量
        
        Args:
            text: 输入文本
        
        Returns:
            2048 维 numpy 数组
        """
        client = await self._get_client()
        response = await client.post(
            self.ENDPOINT,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "input": text
            }
        )
        
        response.raise_for_status()
        result = response.json()
        
        embedding = result["data"][0]["embedding"]
        return np.array(embedding, dtype=np.float32)
    
    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
