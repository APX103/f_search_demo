# 图像搜索性能优化分析

> 日期：2026-03-09
> 背景：单次搜索请求耗时约 10 秒，需要分析瓶颈并提出优化方案

## 一、当前搜索流水线

```
时间线 ──────────────────────────────────────────────────────────►

Step 1 (并行):
  ├── 图像编码 (阿里云 Multimodal-Embedding API)    ~1-2s
  └── VLM 描述生成 (智谱 GLM-4V-Flash)              ~3-6s  ← 最大瓶颈
                                                         │
Step 2 (串行，依赖 VLM 结果):                             ▼
  └── 文本编码 (智谱 Embedding-3)                    ~0.5-1s

Step 3 (串行):
  └── Zilliz Cloud 混合搜索 (REST API)               ~0.5-1s（含新建 TCP/TLS 连接）

总计约: 5-9s (理论) + 网络波动 ≈ 10s
```

### 关键路径分析

```
                    ┌──────────────────┐
                    │  用户上传图片      │
                    └────────┬─────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
    ┌───────────────────┐    ┌───────────────────┐
    │ 图像编码 ~1-2s     │    │ VLM 描述 ~3-6s    │ ← 瓶颈：关键路径上最慢
    │ (阿里云 API)       │    │ (智谱 GLM-4V)     │
    └───────────────────┘    └────────┬──────────┘
                                      │
                                      ▼
                             ┌───────────────────┐
                             │ 文本编码 ~0.5-1s   │ ← 必须等 VLM 完成
                             │ (智谱 Embedding-3) │
                             └────────┬──────────┘
                                      │
                              ┌───────┴────────┐
                              │                │
                              ▼                ▼
                    ┌───────────────────────────────┐
                    │ Zilliz 混合搜索 ~0.5-1s        │
                    │ (每次新建 httpx 连接)           │ ← 无连接复用
                    └───────────────────────────────┘
```

### 瓶颈总结

| 环节 | 耗时估算 | 瓶颈原因 |
|------|---------|---------|
| VLM 描述生成 (GLM-4V-Flash) | 3-6s | LLM 推理慢，占总耗时 50%+ |
| 文本编码 | 0.5-1s | 串行依赖 VLM 输出，无法并行 |
| Zilliz 搜索 | 0.5-1s | 每次请求新建 `httpx.AsyncClient`，TCP/TLS 握手开销 |
| 图片压缩 (`_compress_image`) | 0.1-0.5s | CPU 密集型 PIL 操作阻塞事件循环 |
| 无缓存 | — | 相同图片重复搜索仍走完整流水线 |

## 二、优化方案

### P0：立即见效的快速优化

#### 2.1 添加分步耗时日志（定位瓶颈）

**文件**: `src/search/hybrid_search.py`

在 `search()` 方法中加入分步计时，用实际数据确认瓶颈分布：

```python
import time, logging
logger = logging.getLogger(__name__)

async def search(self, image_bytes, top_k=20, category_hint=None):
    t0 = time.time()

    # Step 1: 并行
    image_emb, description = await asyncio.gather(
        self.image_encoder.encode(image_bytes),
        self.desc_generator.generate(image_bytes)
    )
    logger.info(f"Step1 (image_encode + VLM): {time.time()-t0:.2f}s")

    # Step 2: 文本编码
    t1 = time.time()
    text_emb = await self.text_encoder.encode(description)
    logger.info(f"Step2 (text_encode): {time.time()-t1:.2f}s")

    # Step 3: Zilliz 搜索
    t2 = time.time()
    results = await self.zilliz.hybrid_search(...)
    logger.info(f"Step3 (zilliz_search): {time.time()-t2:.2f}s")
    logger.info(f"Total search: {time.time()-t0:.2f}s")
```

**预期收益**: 无直接性能提升，但为后续优化提供精确数据支撑。

#### 2.2 ZillizClient 连接池复用（预计省 0.3-0.5s）

**文件**: `src/storage/zilliz_client.py`

**问题**: 当前每次搜索都 `async with httpx.AsyncClient(timeout=30.0) as client`，新建连接包含 TCP + TLS 握手。

**方案**: 改为与 ImageEncoder / TextEncoder 一致的懒加载复用模式：

```python
class ZillizClient:
    def __init__(self, endpoint, token, collection_name):
        ...
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def hybrid_search(self, ...):
        client = await self._get_client()
        response = await client.post(url, headers=self.headers, json=payload)
        ...

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None
```

#### 2.3 提供"快速搜索"模式 — 跳过 VLM（预计省 4-7s）

**这是单项投入产出比最高的优化。**

VLM 的作用是生成文字描述再转文本向量做语义搜索，但图像向量搜索本身就能返回不错的结果。提供两种模式：

- **快速模式**: 仅图像向量搜索，~1-2s 完成
- **精确模式**: 保持现有三路混合搜索，~10s

```python
async def search(self, image_bytes, top_k=20, fast_mode=False, category_hint=None):
    image_emb = await self.image_encoder.encode(image_bytes)

    if fast_mode:
        results = await self.zilliz.search(
            vector=image_emb.tolist(),
            anns_field="imageVector",
            limit=top_k,
            output_fields=self.OUTPUT_FIELDS
        )
        return results, ""

    # 完整混合搜索流程...
    description = await self.desc_generator.generate(image_bytes)
    text_emb = await self.text_encoder.encode(description)
    ...
```

同时在前端增加一个开关让用户选择搜索模式。

### P1：中等投入，效果显著

#### 2.4 结果缓存（重复图片秒回）

**问题**: 相同图片的重复搜索每次都走完整流水线。

**方案**: 用图片内容哈希做 key，缓存搜索结果：

```python
import hashlib
from cachetools import TTLCache

class HybridSearchService:
    def __init__(self, ...):
        ...
        self._cache = TTLCache(maxsize=256, ttl=600)  # 10 分钟过期

    async def search(self, image_bytes, top_k=20, ...):
        cache_key = hashlib.md5(image_bytes).hexdigest() + f":{top_k}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # ... 正常搜索 ...

        self._cache[cache_key] = (results, description)
        return results, description
```

#### 2.5 前端预压缩图片（预计省 0.5-1s）

**问题**: 用户上传的原始大图需要在后端做 PIL 压缩，且大图上传本身也慢。

**方案**: 在前端用 Canvas API 压缩到 800px 宽度、JPEG 质量 80%：

```javascript
function compressImage(file, maxWidth = 800, quality = 0.8) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const ratio = Math.min(maxWidth / img.width, 1);
            canvas.width = img.width * ratio;
            canvas.height = img.height * ratio;
            canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(resolve, 'image/jpeg', quality);
        };
        img.src = URL.createObjectURL(file);
    });
}
```

#### 2.6 图片压缩放入线程池（预计省 0.1-0.3s）

**文件**: `src/encoders/image_encoder.py`

**问题**: `_compress_image()` 是 CPU 密集型的 PIL 操作，在 async 事件循环中执行会阻塞其他协程。

**方案**:

```python
async def encode(self, image_bytes: bytes) -> np.ndarray:
    image_bytes = await asyncio.to_thread(self._compress_image, image_bytes)
    ...
```

### P2：架构级优化（长期）

#### 2.7 替换 VLM 为本地 CLIP 模型（预计总延迟降到 1-2s）

**这是最根本的优化方案。**

当前架构：图片 → VLM 生成文字(3-6s) → 文本编码(0.5-1s) → 文本向量，绕了一大圈。

可以改用 CLIP 模型（如 `chinese-clip` 或 OpenAI CLIP）在本地同时生成图像向量和文本空间对齐的向量：

```
当前:  图片 → [VLM 生成文字 3-6s] → [文本编码 1s] → 文本向量
优化:  图片 → [本地 CLIP ~0.1s] → 图像向量 + 文本向量（同时输出）
```

**代价**: 需要重新对全部商品数据做 embedding 导入，但长期收益巨大。

#### 2.8 VLM 流式输出 + 提前文本编码

如果必须保留 VLM，可以用流式（streaming）返回。VLM 输出到前几个标签时就提前启动文本编码，不需要等待全部生成完成，预计省 1-2s。

#### 2.9 HTTP/2 多路复用

让所有 httpx 客户端启用 HTTP/2，利用连接多路复用减少开销：

```python
self._client = httpx.AsyncClient(timeout=30.0, http2=True)
```

需额外安装 `httpcore[http2]`。

## 三、方案对比总结

| 方案 | 预计节省 | 实现难度 | 优先级 | 涉及文件 |
|------|---------|---------|--------|---------|
| 添加分步耗时日志 | — (诊断) | 低 | **P0** | `hybrid_search.py` |
| Zilliz 连接池复用 | 0.3-0.5s | 低 | **P0** | `zilliz_client.py` |
| 快速搜索模式(跳过 VLM) | 4-7s | 中 | **P0** | `hybrid_search.py`, `search.py`, `index.html` |
| 结果缓存 | 100%(重复请求) | 低 | **P1** | `hybrid_search.py` |
| 前端预压缩图片 | 0.5-1s | 低 | **P1** | `frontend/index.html` |
| 图片压缩放线程池 | 0.1-0.3s | 低 | **P1** | `image_encoder.py` |
| 替换为本地 CLIP 模型 | 5-8s | 高 | **P2** | 全量重构 encoders + 重新导入数据 |
| VLM 流式 + 提前编码 | 1-2s | 中 | **P2** | `description.py`, `hybrid_search.py` |
| HTTP/2 多路复用 | 0.1-0.3s | 低 | **P2** | 所有 encoder + zilliz_client |

## 四、推荐实施路径

```
阶段 1 (1 天)
  ├── [P0] 添加分步耗时日志 → 确认实际瓶颈分布
  ├── [P0] Zilliz 连接池复用
  └── [P0] 快速搜索模式
  预期效果: 快速模式 ~1-2s，精确模式 ~8s

阶段 2 (2-3 天)
  ├── [P1] 结果缓存
  ├── [P1] 前端预压缩图片
  └── [P1] 图片压缩放线程池
  预期效果: 精确模式 ~6-7s，重复请求秒回

阶段 3 (1-2 周，长期)
  ├── [P2] 评估 CLIP 模型替换 VLM
  └── [P2] HTTP/2 + 流式优化
  预期效果: 所有请求 ~1-2s
```
