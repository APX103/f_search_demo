# src/generators/description.py
"""描述生成服务"""

from typing import Protocol
import base64
import hashlib
import httpx
from collections import OrderedDict


UNIFIED_DESCRIPTION_PROMPT = """
Analyze this furniture image and generate a structured description.
For each attribute, provide both English and Chinese.

Output format (use exactly these labels):
[COLOR] Primary and secondary colors, e.g., dark gray / 深灰色, oak with white accents / 白边橡木
[MATERIAL] Visible materials, e.g., fabric / 布艺, leather / 皮革, solid wood / 实木, metal frame / 金属框架
[STYLE] Design style, e.g., modern minimalist / 现代简约, Scandinavian / 北欧风, industrial / 工业风
[SHAPE] Form and structure, e.g., L-shaped / L型, with armrests / 有扶手, tapered legs / 锥形桌腿
[SIZE] Apparent scale, e.g., 3-seater / 三人位, compact / 紧凑型, oversized / 加大款
[SUMMARY] One fluent sentence describing this furniture piece

Rules:
- Describe only what is visible, do not assume
- Use common, searchable terms
- Be consistent: same furniture should produce similar descriptions
- If uncertain, omit rather than guess
- Always include both English and Chinese for each attribute value
""".strip()


class DescriptionGenerator(Protocol):
    """描述生成器协议"""
    
    async def generate(self, image_bytes: bytes) -> str:
        """为图片生成结构化描述"""
        ...


class ZhipuDescriptionGenerator:
    """智谱 GLM-4.6V-Flash 描述生成器"""
    
    ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    
    def __init__(
        self,
        api_key: str,
        model: str = "glm-4v-flash",
        timeout: float = 20.0,
        max_tokens: int = 500,
        cache_size: int = 128
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_size = cache_size
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端（懒加载）"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def generate(self, image_bytes: bytes) -> str:
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        
        cached = self._cache.get(image_hash)
        if cached is not None:
            self._cache.move_to_end(image_hash)
            return cached
        
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
                "max_tokens": self.max_tokens
            }
        )
        
        response.raise_for_status()
        result = response.json()
        description = result["choices"][0]["message"]["content"]
        
        self._cache[image_hash] = description
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        
        return description
    
    def clear_cache(self) -> None:
        self._cache.clear()
    
    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._cache.clear()
