# Furniture Search SDK Makefile

.PHONY: help install test start dev docker-build docker-run clean

help: ## 显示帮助信息
	@echo "Furniture Search SDK - 可用命令："
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## 安装 SDK
	cd furniture_search && pip install -e .

test: ## 运行测试
	cd furniture_search && python examples/test_sdk.py

start: ## 启动服务
	cd furniture_search && ./start.sh

dev: ## 开发模式启动服务
	cd furniture_search && python -m furniture_search.server

docker-build: ## 构建 Docker 镜像
	cd furniture_search && docker build -t furniture-search-sdk .

docker-run: ## 运行 Docker 容器
	docker run -d -p 8000:8000 --env-file furniture_search/.env furniture-search-sdk

clean: ## 清理临时文件
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov

format: ## 格式化代码
	cd furniture_search && ruff format .
	cd furniture_search && ruff check --fix .

lint: ## 代码检查
	cd furniture_search && ruff check .
