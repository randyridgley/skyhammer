"""Embeddings (OpenAI-shaped) and rerank over Bedrock models.

    LLM("titan-embed-text-v2").embeddings.create(input=["a", "b"], dimensions=256)
    LLM("embed-v4").embeddings.create(input="query", input_type="search_query")
    LLM("rerank-v3-5").rerank("capital of France", ["Berlin...", "Paris..."], top_n=1)

Each model family has its own InvokeModel body; this module hides that. Token counts come
from Bedrock's ``x-amzn-bedrock-input-token-count`` header, so costs are exact.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from skyhammer.errors import SkyhammerError

_COHERE_BATCH = 96
_NOVA_PURPOSE = {
    "search_document": "GENERIC_INDEX",
    "search_query": "GENERIC_RETRIEVAL",
    "classification": "CLASSIFICATION",
    "clustering": "CLUSTERING",
}
_RERANK_DOCS_PER_SEARCH_UNIT = 100


@dataclass
class Embedding:
    index: int
    embedding: list[float]
    object: str = "embedding"


@dataclass
class EmbeddingUsage:
    prompt_tokens: int
    total_tokens: int
    cost_usd: float | None = None


@dataclass
class EmbeddingResponse:
    data: list[Embedding]
    model: str
    usage: EmbeddingUsage
    object: str = "list"


@dataclass
class RerankResult:
    index: int
    relevance_score: float
    document: str | None = None


@dataclass
class RerankUsage:
    search_units: int
    cost_usd: float | None = None


@dataclass
class RerankResponse:
    results: list[RerankResult]
    model: str
    usage: RerankUsage = field(default_factory=lambda: RerankUsage(0))


def _family(model_id: str) -> str:
    base = (
        model_id.split(".", 1)[1]
        if model_id.startswith(("us.", "eu.", "apac.", "global."))
        else model_id
    )
    if base.startswith("amazon.titan-embed-text-v2"):
        return "titan-v2"
    if base.startswith("amazon.titan-embed"):
        return "titan"
    if base.startswith("cohere.embed"):
        return "cohere"
    if base.startswith("amazon.nova-2-multimodal-embeddings"):
        return "nova"
    raise SkyhammerError(
        f"{model_id} isn't a supported text embedding model yet "
        "(Titan, Cohere Embed, Nova Multimodal Embeddings are)."
    )


class Embeddings:
    def __init__(
        self,
        runtime: Any,
        model_id: str,
        on_usage: Callable[[int], float | None] | None = None,
        before_call: Callable[[], None] | None = None,
        max_workers: int = 8,
    ) -> None:
        self.runtime = runtime
        self.model_id = model_id
        self._on_usage = on_usage
        self._before_call = before_call
        self.max_workers = max_workers

    def _invoke(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        resp = self.runtime.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        tokens = int(
            resp.get("ResponseMetadata", {})
            .get("HTTPHeaders", {})
            .get("x-amzn-bedrock-input-token-count", 0)
        )
        return json.loads(resp["body"].read()), tokens

    def create(
        self,
        *,
        input: str | list[str],
        dimensions: int | None = None,
        input_type: str = "search_document",
        normalize: bool = True,
        **_: Any,
    ) -> EmbeddingResponse:
        """``input_type``: search_document | search_query | classification | clustering."""
        if self._before_call:
            self._before_call()
        texts = [input] if isinstance(input, str) else list(input)
        family = _family(self.model_id)
        vectors: list[list[float]] = []
        tokens = 0
        if family == "cohere":
            for i in range(0, len(texts), _COHERE_BATCH):
                body: dict[str, Any] = {
                    "texts": texts[i : i + _COHERE_BATCH],
                    "input_type": input_type,
                    "embedding_types": ["float"],
                }
                if dimensions:
                    body["output_dimension"] = dimensions
                out, n = self._invoke(body)
                embs = out["embeddings"]
                vectors += embs["float"] if isinstance(embs, dict) else embs
                tokens += n
        else:

            def one(text: str) -> tuple[list[float], int]:
                if family == "titan-v2":
                    body = {"inputText": text, "normalize": normalize}
                    if dimensions:
                        body["dimensions"] = dimensions
                elif family == "titan":
                    body = {"inputText": text}
                else:  # nova
                    body = {
                        "taskType": "SINGLE_EMBEDDING",
                        "singleEmbeddingParams": {
                            "embeddingPurpose": _NOVA_PURPOSE.get(input_type, "GENERIC_INDEX"),
                            "embeddingDimension": dimensions or 1024,
                            "text": {"truncationMode": "END", "value": text},
                        },
                    }
                out, n = self._invoke(body)
                vec = out["embeddings"][0]["embedding"] if family == "nova" else out["embedding"]
                return vec, n

            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(texts) or 1)) as ex:
                for vec, n in ex.map(one, texts):  # map preserves input order
                    vectors.append(vec)
                    tokens += n
        usage = EmbeddingUsage(tokens, tokens)
        if self._on_usage:
            usage.cost_usd = self._on_usage(tokens)
        return EmbeddingResponse(
            [Embedding(i, v) for i, v in enumerate(vectors)], self.model_id, usage
        )


def rerank(
    agent_runtime: Any,
    region: str,
    model_id: str,
    query: str,
    documents: list[str],
    *,
    top_n: int | None = None,
    return_documents: bool = False,
    price_per_search_unit: float | None = None,
) -> RerankResponse:
    if not documents:
        return RerankResponse([], model_id)
    resp = agent_runtime.rerank(
        queries=[{"type": "TEXT", "textQuery": {"text": query}}],
        sources=[
            {
                "type": "INLINE",
                "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": d}},
            }
            for d in documents
        ],
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "numberOfResults": top_n or len(documents),
                "modelConfiguration": {
                    "modelArn": f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
                },
            },
        },
    )
    units = math.ceil(len(documents) / _RERANK_DOCS_PER_SEARCH_UNIT)
    results = [
        RerankResult(
            r["index"], r["relevanceScore"], documents[r["index"]] if return_documents else None
        )
        for r in resp.get("results", [])
    ]
    cost = units * price_per_search_unit if price_per_search_unit is not None else None
    return RerankResponse(results, model_id, RerankUsage(units, cost))
