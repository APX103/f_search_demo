# tests/test_description_generator.py
"""描述生成器测试"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.mark.asyncio
async def test_zhipu_description_generator():
    """测试智谱描述生成器"""
    from src.generators.description import ZhipuDescriptionGenerator
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "[COLOR] Dark gray\n[MATERIAL] Fabric\n[STYLE] Modern"
                }
            }
        ]
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        generator = ZhipuDescriptionGenerator(api_key="test_key")
        
        image_bytes = b'\xff\xd8\xff\xe0\x00\x10JFIF'
        description = await generator.generate(image_bytes)
        
        assert "[COLOR]" in description
        assert "Dark gray" in description
        mock_post.assert_called_once()


def test_unified_description_prompt_content():
    """测试统一描述 prompt 包含必要标签"""
    from src.generators.description import UNIFIED_DESCRIPTION_PROMPT
    
    assert "[COLOR]" in UNIFIED_DESCRIPTION_PROMPT
    assert "[MATERIAL]" in UNIFIED_DESCRIPTION_PROMPT
    assert "[STYLE]" in UNIFIED_DESCRIPTION_PROMPT
    assert "[SHAPE]" in UNIFIED_DESCRIPTION_PROMPT
    assert "[SIZE]" in UNIFIED_DESCRIPTION_PROMPT
    assert "[SUMMARY]" in UNIFIED_DESCRIPTION_PROMPT


@pytest.mark.asyncio
async def test_generator_uses_correct_model():
    """测试生成器使用正确的模型"""
    from src.generators.description import ZhipuDescriptionGenerator
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "test"}}]
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        generator = ZhipuDescriptionGenerator(api_key="test_key")
        await generator.generate(b'\xff\xd8\xff\xe0')
        
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "glm-4v-flash"
