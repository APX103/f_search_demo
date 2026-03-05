"""测试 SDK 是否正常工作"""

import asyncio
import base64
import sys
from pathlib import Path

# 添加 SDK 到路径
sys.path.insert(0, str(Path(__file__).parent))

from furniture_search import FurnitureSearchClient, SearchConfig
from furniture_search.config import get_settings


async def test_health_check():
    """测试健康检查"""
    print("测试 1: 健康检查")
    print("=" * 50)
    
    settings = get_settings()
    print(f"✅ Zilliz URI: {settings.zilliz_cloud_uri}")
    print(f"✅ 集合名称: {settings.zilliz_cloud_collection}")
    print()


async def test_client_init():
    """测试客户端初始化"""
    print("测试 2: 客户端初始化")
    print("=" * 50)
    
    settings = get_settings()
    
    client = FurnitureSearchClient(
        zilliz_endpoint=settings.zilliz_cloud_uri,
        zilliz_token=settings.zilliz_cloud_token,
        aliyun_api_key=settings.aliyun_dashscope_api_key,
        zhipu_api_key=settings.zhipu_api_key,
        collection_name=settings.zilliz_cloud_collection,
        config=SearchConfig(rrf_k=60, candidate_multiplier=3)
    )
    
    await client.close()
    print("✅ 客户端初始化成功")
    print()


async def test_base64_decode():
    """测试 base64 解码"""
    print("测试 3: Base64 解码")
    print("=" * 50)
    
    # 创建一个简单的测试图片（1x1 像素的 PNG）
    test_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\x0d\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    test_base64 = base64.b64encode(test_image_data).decode()
    
    # 尝试解码
    decoded = base64.b64decode(test_base64)
    assert decoded == test_image_data
    print("✅ Base64 编码/解码正常")
    print()


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("Furniture Search SDK - 测试套件")
    print("=" * 50 + "\n")
    
    try:
        await test_health_check()
        await test_client_init()
        await test_base64_decode()
        
        print("=" * 50)
        print("✅ 所有测试通过！")
        print("=" * 50)
        print("\n下一步：")
        print("  1. 启动 HTTP 服务：python -m furniture_search.server")
        print("  2. 访问 API 文档：http://localhost:8000/docs")
        print("  3. 使用其他语言调用：查看 examples/http_client_examples.md")
        
    except Exception as e:
        print("\n" + "=" * 50)
        print("❌ 测试失败")
        print("=" * 50)
        print(f"\n错误信息: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
