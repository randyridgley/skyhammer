"""Serving metrics for any model: traffic, latency percentiles, tokens and errors.

Read from CloudWatch, so it covers every caller (other processes, the OpenAI client,
batch). Shared serverless models report account-wide numbers; fine-tuned models and
dedicated endpoints have their own. Cost: free. Run: uv run python examples/27_serving_metrics.py
CLI: skyhammer metrics nova-micro --hours 24
"""

from skyhammer import LLM

llm = LLM("nova-micro")
llm.chat.completions.create(messages=[{"role": "user", "content": "Say hi"}], max_tokens=10)

m = llm.metrics(hours=24)  # CloudWatch lags ~1-2 min, so the call above may not show yet
print(f"{m.model_id}: {m.invocations:,} calls in {m.hours:g}h, error rate {m.error_rate:.1%}")
print(f"latency p50 {m.latency_ms_p50 or 0:,.0f} ms, p99 {m.latency_ms_p99 or 0:,.0f} ms")
print(f"tokens in/out: {m.input_tokens:,} / {m.output_tokens:,}; throttles: {m.throttles}")
# Same for your models: LLM.get("support-qwen-v1").metrics()  (Bedrock or SageMaker endpoint)
