# src/encoders/image_encoder.py
"""图像编码器实现"""

from abc import ABC, abstractmethod
from typing import Protocol
import base64
import numpy as np
import httpx


class ImageEncoder(Protocol):
    """图像编码器协议"""
    
    async def encode(self, image_bytes: bytes) -> np.ndarray:
        """将图片编码为向量"""
        ...


class AliyunImageEncoder:
    """阿里云多模态向量服务 - 图像编码器"""
    
    ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
    
    def __init__(
        self,
        api_key: str,
        model: str = "multimodal-embedding-v1",
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
    
    async def encode(self, image_bytes: bytes) -> np.ndarray:
        """
        将图片编码为 1024 维向量
        
        Args:
            image_bytes: 图片二进制数据
        
        Returns:
            1024 维 numpy 数组
        """
        image_b64 = base64.b64encode(image_bytes).decode()
        
        client = await self._get_client()
        response = await client.post(
            self.ENDPOINT,
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
            }
        )
        
        response.raise_for_status()
        result = response.json()
        
        embedding = result["output"]["embeddings"][0]["embedding"]
        return np.array(embedding, dtype=np.float32)
    
    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
