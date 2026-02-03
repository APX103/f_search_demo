# tests/test_image_encoder.py
"""图像编码器测试"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.mark.asyncio
async def test_aliyun_image_encoder_encode():
    """测试阿里云图像编码器"""
    from src.encoders.image_encoder import AliyunImageEncoder
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "output": {
            "embeddings": [
                {"embedding": [0.1] * 1024}
            ]
        }
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        encoder = AliyunImageEncoder(api_key="test_key")
        
        # 创建测试图片数据（最小有效 JPEG）
        image_bytes = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        embedding = await encoder.encode(image_bytes)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)
        mock_post.assert_called_once()


def test_image_encoder_protocol():
    """测试图像编码器遵循协议"""
    from src.encoders.image_encoder import ImageEncoder, AliyunImageEncoder
    
    # 验证 AliyunImageEncoder 实现了 ImageEncoder 协议
    encoder = AliyunImageEncoder(api_key="test_key")
    assert hasattr(encoder, "encode")
