import io
import json
from typing import Any

import boto3
import pytest
from botocore.validate import validate_parameters

from skyhammer.embeddings import Embeddings, rerank
from skyhammer.errors import SkyhammerError


class FakeRuntime:
    def __init__(self) -> None:
        self.bodies: list[tuple[str, dict[str, Any]]] = []

    def invoke_model(self, modelId: str, body: str, **_: Any) -> dict[str, Any]:
        b = json.loads(body)
        self.bodies.append((modelId, b))
        n = len(b.get("texts", [None]))
        if "texts" in b:
            out: dict[str, Any] = {"embeddings": {"float": [[0.1, 0.2]] * n}}
        elif "taskType" in b:
            out = {
                "embeddings": [
                    {
                        "embeddingType": "TEXT",
                        "embedding": [len(b["singleEmbeddingParams"]["text"]["value"])],
                    }
                ]
            }
        else:
            out = {"embedding": [float(len(b["inputText"]))]}
        return {
            "body": io.BytesIO(json.dumps(out).encode()),
            "ResponseMetadata": {"HTTPHeaders": {"x-amzn-bedrock-input-token-count": "3"}},
        }


def test_titan_v2_per_text_preserves_order_and_counts_tokens() -> None:
    rt = FakeRuntime()
    costs: list[int] = []
    resp = Embeddings(
        rt, "amazon.titan-embed-text-v2:0", on_usage=lambda t: costs.append(t) or 0.5
    ).create(input=["a", "bbb", "cc"], dimensions=256)
    assert [e.embedding for e in resp.data] == [[1.0], [3.0], [2.0]]  # input order kept
    assert rt.bodies[0][1] == {"inputText": "a", "normalize": True, "dimensions": 256}
    assert resp.usage.total_tokens == 9 and costs == [9] and resp.usage.cost_usd == 0.5


def test_cohere_batches_of_96_with_input_type_and_dimensions() -> None:
    rt = FakeRuntime()
    resp = Embeddings(rt, "us.cohere.embed-v4:0").create(
        input=[f"t{i}" for i in range(100)], input_type="search_query", dimensions=512
    )
    assert len(rt.bodies) == 2 and len(rt.bodies[0][1]["texts"]) == 96
    assert rt.bodies[0][1]["input_type"] == "search_query"
    assert rt.bodies[0][1]["output_dimension"] == 512
    assert len(resp.data) == 100 and resp.data[99].index == 99


def test_nova_purpose_mapping_and_unsupported_family() -> None:
    rt = FakeRuntime()
    Embeddings(rt, "amazon.nova-2-multimodal-embeddings-v1:0").create(
        input="hi", input_type="search_query"
    )
    params = rt.bodies[0][1]["singleEmbeddingParams"]
    assert (
        params["embeddingPurpose"] == "GENERIC_RETRIEVAL" and params["embeddingDimension"] == 1024
    )
    with pytest.raises(SkyhammerError, match="supported text embedding"):
        Embeddings(rt, "twelvelabs.marengo-embed-3-0-v1:0").create(input="x")


class FakeAgentRuntime:
    def __init__(self) -> None:
        self.req: dict[str, Any] = {}

    def rerank(self, **req: Any) -> dict[str, Any]:
        client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
        validate_parameters(req, client.meta.service_model.operation_model("Rerank").input_shape)
        self.req = req
        return {
            "results": [{"index": 1, "relevanceScore": 0.9}, {"index": 0, "relevanceScore": 0.1}]
        }


def test_rerank_request_is_schema_valid_and_costed_per_search_unit() -> None:
    ar = FakeAgentRuntime()
    docs = ["berlin", "paris"] + [f"d{i}" for i in range(150)]
    resp = rerank(
        ar,
        "us-east-1",
        "cohere.rerank-v3-5:0",
        "capital of france",
        docs,
        top_n=2,
        return_documents=True,
        price_per_search_unit=0.002,
    )
    cfg = ar.req["rerankingConfiguration"]["bedrockRerankingConfiguration"]
    assert cfg["modelConfiguration"]["modelArn"] == (
        "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"
    )
    assert cfg["numberOfResults"] == 2
    assert [(r.index, r.document) for r in resp.results] == [(1, "paris"), (0, "berlin")]
    assert resp.usage.search_units == 2 and resp.usage.cost_usd == pytest.approx(0.004)
