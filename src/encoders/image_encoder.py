# src/encoders/image_encoder.py
"""图像编码器实现"""

from abc import ABC, abstractmethod
from typing import Protocol
import base64
import io
import numpy as np
import httpx
from PIL import Image


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
    
    def _compress_image(self, image_bytes: bytes, max_size_kb: int = 2800) -> bytes:
        """压缩图片到指定大小以下"""
        # 如果已经小于限制，直接返回
        if len(image_bytes) <= max_size_kb * 1024:
            return image_bytes
        
        img = Image.open(io.BytesIO(image_bytes))
        
        # 转换为 RGB（去除 alpha 通道）
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # 逐步降低质量直到满足大小要求
        for quality in [85, 70, 55, 40, 25]:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            if buffer.tell() <= max_size_kb * 1024:
                return buffer.getvalue()
        
        # 如果质量降低还不够，缩小尺寸
        width, height = img.size
        for scale in [0.75, 0.5, 0.35, 0.25]:
            new_size = (int(width * scale), int(height * scale))
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            resized.save(buffer, format='JPEG', quality=50, optimize=True)
            if buffer.tell() <= max_size_kb * 1024:
                return buffer.getvalue()
        
        return buffer.getvalue()
    
    async def encode(self, image_bytes: bytes) -> np.ndarray:
        """
        将图片编码为 1024 维向量
        
        Args:
            image_bytes: 图片二进制数据
        
        Returns:
            1024 维 numpy 数组
        """
        # 压缩图片（阿里云限制 3MB）
        image_bytes = self._compress_image(image_bytes)
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
        
        if response.status_code != 200:
            print(f"[AliyunImageEncoder] Error {response.status_code}: {response.text}")
            response.raise_for_status()
        result = response.json()
        
        embedding = result["output"]["embeddings"][0]["embedding"]
        return np.array(embedding, dtype=np.float32)
    
    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
