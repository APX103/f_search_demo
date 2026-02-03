# tests/test_text_encoder.py
"""文本编码器测试"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.mark.asyncio
async def test_zhipu_text_encoder_encode():
    """测试智谱文本编码器"""
    from src.encoders.text_encoder import ZhipuTextEncoder
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1] * 2048}
        ]
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        encoder = ZhipuTextEncoder(api_key="test_key")
        
        text = "三人位布艺沙发，北欧风格"
        embedding = await encoder.encode(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (2048,)
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_zhipu_text_encoder_sends_correct_payload():
    """测试智谱文本编码器发送正确的请求"""
    from src.encoders.text_encoder import ZhipuTextEncoder
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1] * 2048}]
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        encoder = ZhipuTextEncoder(api_key="test_key")
        await encoder.encode("test text")
        
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "embedding-3"
        assert call_kwargs["json"]["input"] == "test text"
