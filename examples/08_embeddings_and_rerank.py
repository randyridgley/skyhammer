"""Embeddings for semantic search, then rerank the hits.

Cost: a fraction of a cent. Run: uv run python examples/08_embeddings_and_rerank.py
"""

import math

from skyhammer import LLM

docs = [
    "To reset your password, open Settings > Security.",
    "Invoices are emailed on the first business day of each month.",
    "Two-factor authentication can be enabled from your profile page.",
    "Refunds are processed within five business days.",
]
query = "How do I change my password?"

embedder = LLM("titan-embed-text-v2")  # also: embed-v4 (Cohere), nova-2-multimodal-embeddings
doc_vecs = embedder.embeddings.create(input=docs, dimensions=256).data
q_vec = (
    embedder.embeddings.create(input=query, input_type="search_query", dimensions=256)
    .data[0]
    .embedding
)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


top = sorted(range(len(docs)), key=lambda i: cosine(q_vec, doc_vecs[i].embedding), reverse=True)
print("embedding search:", [docs[i][:40] for i in top[:3]])

ranked = LLM("rerank-v3-5").rerank(
    query, [docs[i] for i in top[:3]], top_n=2, return_documents=True
)
for r in ranked.results:
    print(f"rerank {r.relevance_score:.3f}: {r.document}")
print("rerank cost: $", ranked.usage.cost_usd)
