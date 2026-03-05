#!/bin/bash

# 家具搜索 SDK - 快速启动脚本

set -e

echo "=== 家具搜索 SDK 启动脚本 ==="
echo

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "⚠️  警告: 未找到 .env 文件"
    echo "请复制 .env.example 并配置环境变量："
    echo "  cp .env.example .env"
    echo "  # 编辑 .env 填入实际的 API 密钥"
    echo
    exit 1
fi

# 加载环境变量
export $(cat .env | grep -v '^#' | xargs)

# 检查必要的环境变量
required_vars=(
    "ZILLIZ_CLOUD_URI"
    "ZILLIZ_CLOUD_TOKEN"
    "ALIYUN_DASHSCOPE_API_KEY"
    "ZHIPU_API_KEY"
)

missing_vars=()
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "❌ 错误: 以下环境变量未配置："
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo
    echo "请在 .env 文件中配置这些变量"
    exit 1
fi

echo "✅ 环境变量检查通过"
echo
echo "配置信息："
echo "  - Zilliz 集群: $ZILLIZ_CLOUD_URI"
echo "  - 集合名称: $ZILLIZ_CLOUD_COLLECTION"
echo
echo "🚀 启动服务..."
echo

# 启动服务
python -m furniture_search.server
