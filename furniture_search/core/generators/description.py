# src/generators/description.py
"""描述生成服务"""

from typing import Protocol
import base64
import httpx


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
        timeout: float = 30.0,
        max_tokens: int = 500
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端（懒加载）"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def generate(self, image_bytes: bytes) -> str:
        """
        为图片生成结构化描述
        
        Args:
            image_bytes: 图片二进制数据
        
        Returns:
            结构化描述文本
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
        
        return result["choices"][0]["message"]["content"]
    
    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
