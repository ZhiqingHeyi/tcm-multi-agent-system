"""本地 GPU 向量化服务（Apple Silicon / MPS）。

**为什么必须跑在宿主机，而不是容器里：**
Docker Desktop on macOS 无法把 Metal GPU 透传进容器（不同于 Linux 的 --gpus）。
因此 GPU 加速的向量化只能跑在宿主机上，再通过 OpenAI 兼容的 /v1/embeddings
契约暴露给容器内的后端。好处是后端的 embeddings.py 一行都不用改——它本来就是
"远程优先"设计，这里只是把"远程"换成了"本机"。

**为什么指令前缀放在客户端而不是这里：**
BGE / E5 这类模型要求给 query 加指令前缀。如果在本服务里加，就必须给请求体
塞一个非标准的 input_type 字段，换成正牌供应商就会 400。把前缀交给客户端处理，
本服务就是一个纯编码器，契约 100% 兼容 OpenAI，任何供应商都能无缝替换。

启动（建议用 hf-mirror 加速模型下载）：
    HF_ENDPOINT=https://hf-mirror.com ./.venv-embed/bin/python -m tools.embed_server

自检：
    ./.venv-embed/bin/python -m tools.embed_server --check

接口：
    GET  /health          → 模型、维度、设备、是否预热
    GET  /v1/models       → 兼容 OpenAI 客户端探测
    POST /v1/embeddings   → {"model": "...", "input": ["文本", ...]}
"""

from __future__ import annotations

import argparse
import os
import threading
import time

from pydantic import BaseModel

# 必须在 import torch 之前设置：MPS 有部分算子未实现，允许自动回落到 CPU 执行，
# 否则遇到不支持的操作会直接抛异常中断入库。
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEFAULT_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
DEFAULT_BATCH = int(os.environ.get("EMBED_BATCH", "16"))
MAX_SEQ_LEN = int(os.environ.get("EMBED_MAX_SEQ", "1024"))
# fp16 在 MPS 上通常能带来接近翻倍的吞吐；向量质量影响可忽略，
# 但若出现 NaN 或明显掉点就退回 float32（用 EMBED_DTYPE=float32）。
DTYPE = os.environ.get("EMBED_DTYPE", "float32")
HOST = os.environ.get("EMBED_HOST", "127.0.0.1")
PORT = int(os.environ.get("EMBED_PORT", "8100"))


class EmbeddingRequest(BaseModel):
    """必须定义在模块级。

    若定义在 create_app() 内部，配合 from __future__ import annotations，
    FastAPI 无法在模块全局命名空间解析 "EmbeddingRequest" 这个注解字符串，
    会静默退化为"把它当成查询参数"，请求体被忽略并返回 422
    （detail 里 loc 是 ["query","request"]）。这是极难猜到的排查点。
    """

    model: str | None = None
    input: str | list[str]

_model = None
_torch = None
_lock = threading.Lock()
_device = "cpu"
_dim = 0


def _pick_device(torch_module) -> str:
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    """惰性加载模型并预热。首次调用会触发模型下载（约 2.3GB）。"""
    global _model, _torch, _device, _dim
    with _lock:
        if _model is not None:
            return _model
        import torch
        from sentence_transformers import SentenceTransformer

        _torch = torch
        _device = _pick_device(torch)
        started = time.time()
        model = SentenceTransformer(DEFAULT_MODEL, device=_device)
        # 截断长度直接决定单块编码成本。实测语料块长 p99=767、最长 806 字，
        # 1024 已完全覆盖，同时避免超长块把 GPU 显存吃满。
        model.max_seq_length = min(MAX_SEQ_LEN, model.max_seq_length or MAX_SEQ_LEN)
        if DTYPE in ("float16", "fp16") and _device == "mps":
            model = model.half()
        probe = model.encode(["预热"], batch_size=1, normalize_embeddings=True)
        if probe.dtype.name == "float16" and not all(v == v for v in probe.flatten().tolist()):
            raise RuntimeError("fp16 编码出现 NaN，请用 EMBED_DTYPE=float32 重启")
        _dim = int(probe.shape[1])
        _model = model
        print(
            f"[embed-server] 模型 {DEFAULT_MODEL} 已加载 | 设备 {_device} | 维度 {_dim} | "
            f"max_seq {model.max_seq_length} | 耗时 {time.time() - started:.1f}s",
            flush=True,
        )
        return _model


def encode(texts: list[str], batch_size: int = DEFAULT_BATCH) -> list[list[float]]:
    model = load_model()
    with _lock:
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # 归一化后余弦相似度等价于点积，与 pgvector 的 vector_cosine_ops 对齐
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return [vector.tolist() for vector in vectors]


def create_app():
    from fastapi import FastAPI

    app = FastAPI(title="Local Embedding Service (MPS)", version="1.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": DEFAULT_MODEL,
            "device": _device,
            "dim": _dim,
            "loaded": _model is not None,
            "max_seq_length": getattr(_model, "max_seq_length", None),
        }

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": DEFAULT_MODEL, "object": "model", "owned_by": "local-mps"}]}

    @app.post("/v1/embeddings")
    def embeddings(request: EmbeddingRequest) -> dict:
        # 同步 def：FastAPI 会自动丢进线程池，避免阻塞事件循环
        texts = [request.input] if isinstance(request.input, str) else list(request.input)
        if not texts:
            texts = [""]
        started = time.time()
        vectors = encode(texts)
        return {
            "object": "list",
            "model": DEFAULT_MODEL,
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ],
            "usage": {
                "prompt_tokens": sum(len(text) for text in texts),
                "total_tokens": sum(len(text) for text in texts),
                "elapsed_ms": round((time.time() - started) * 1000, 1),
            },
        }

    return app


# 语义自检：查询用口语化改写，刻意不含方名，正确答案是典籍条文。
# 这类模型（BGE 系）的向量空间是各向异性的，绝对余弦值都挤在 0.5~0.7 的窄带内，
# 所以判据只能是**排序**，不能是绝对阈值。把正确答案与干扰项放一起排序，
# 看正确项能否排到第一——这才是"模型有没有语义区分度"的有效检验。
CHECK_CASES: tuple[tuple[str, str], ...] = (
    ("吹了风以后怕冷，身上微微出汗，后脖子发紧，该用什么方",
     "太阳病，头痛发热，汗出恶风，桂枝汤主之。"),
    ("老人手脚冰凉，精神萎靡，一天到晚总想睡，需要急救回阳",
     "少阴病，脉微细，但欲寐，四逆汤主之。"),
    ("白天动不动就出汗，说话没力气，整个人提不起劲，中气不足",
     "脾胃气虚，元气不足，补中益气汤主之，黄芪为君。"),
    ("春天流行的外感，嗓子痒轻微咳嗽，有点怕风，不算重",
     "太阴风温，但咳，身不甚热，微渴者，辛凉轻剂桑菊饮主之。"),
    ("一侧手脚突然不听使唤，说话含糊，经络被瘀血堵住",
     "气血凝滞，经络不通，活络效灵丹主之，当归丹参乳香没药。"),
)


def run_check() -> None:
    started = time.time()
    load_model()
    print(f"[自检] 设备={_device} 维度={_dim} 加载耗时={time.time() - started:.1f}s")

    import math

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / norm

    queries = [item[0] for item in CHECK_CASES]
    passages = [item[1] for item in CHECK_CASES]
    encoded_started = time.time()
    query_vectors = encode(queries, batch_size=8)
    passage_vectors = encode(passages, batch_size=8)
    elapsed = time.time() - encoded_started

    print(f"[自检] 编码 {len(queries) * 2} 条，耗时 {elapsed:.2f}s（{len(queries) * 2 / max(elapsed, 1e-6):.1f} 条/秒）")
    print("[自检] 排序检验（正确项应排第一）")
    top1 = 0
    for index, (query, _) in enumerate(CHECK_CASES):
        scores = [cosine(query_vectors[index], passage) for passage in passage_vectors]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        hit = order[0] == index
        top1 += int(hit)
        margin = scores[index] - max(s for j, s in enumerate(scores) if j != index)
        print(
            f"  [{'命中' if hit else '未中'}] {query[:22]}… → 预测第 {order.index(index) + 1} 位，"
            f"正确项得分 {scores[index]:.4f}，与最佳干扰项差距 {margin:+.4f}"
        )
    print(f"[自检] Top-1 准确率 {top1}/{len(CHECK_CASES)}")
    print("[自检] 判据：应全部命中。若普遍未命中，检查是否漏加查询指令前缀或维度被截断")


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 MPS 向量化服务")
    parser.add_argument("--check", action="store_true", help="自检模型与语义区分度后退出")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    if args.check:
        run_check()
        return

    import uvicorn

    load_model()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
