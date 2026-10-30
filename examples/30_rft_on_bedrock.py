"""Reinforcement fine-tuning on Bedrock: Nova 2 Lite, and open-weight models cross-region.

Write the reward as a Python function; Skyhammer deploys it as a Lambda in the training
region and trains on prompts plus reference answers (100 to 20,000 rows). Two Bedrock paths:

- Nova 2 Lite trains with Bedrock model customization in us-east-1, then deploys on demand.
- Qwen3-32B and GPT-OSS-20B train through Bedrock's OpenAI-compatible fine-tuning API
  (bedrock-mantle), which runs only in us-west-2. From any home region, pass
  ``backend="mantle"`` and Skyhammer routes there; the fine-tuned model is then called on
  demand with the same chat API. (Without it, these models train on SageMaker at home.)

Bedrock RFT is billed by training time with no runtime cap: Nova 2 Lite is $80/hour
(Qwen3-32B isn't in the AWS Price List yet), so keep prompts and epochs small and use
``job.cancel()`` to stop early. Calls to the reward Lambda run as you, so you need
``lambda:InvokeFunction`` on ``skyhammer-grader-*``.
Run: uv run python examples/30_rft_on_bedrock.py [nova|qwen]
"""

import sys

from skyhammer import LLM, Dataset, grader, reference_answer


@grader
def correct_sum(sample: dict, output: str) -> float:
    """1.0 if the expected number appears in the answer (standard library only)."""
    expected = reference_answer(sample).strip()
    return 1.0 if expected and expected in output else 0.0


problems = Dataset.from_records(
    [
        {
            "messages": [
                {"role": "user", "content": f"What is {a} + {b}? Answer with the number."}
            ],
            "answer": str(a + b),  # sent to the reward function as reference_answer
        }
        for a in range(10, 20)
        for b in range(10, 20)
    ],
    name="sums",
)

if sys.argv[1:] == ["qwen"]:
    # Trains in us-west-2; ~90 of 100 prompts train, under the default batch of 128.
    base, extra = LLM("qwen3-32b"), {"backend": "mantle", "batch_size": 16}
else:
    base, extra = LLM("nova-2-lite"), {}

request, estimate = base.plan("rft", problems, output_model="sums-rft", grader=correct_sum, **extra)
print(request.route.describe(), "|", estimate)

job = base.create_rft_job(problems, grader=correct_sum, output_model="sums-rft", epochs=1, **extra)
job.wait(on_status=lambda st: print(st.state.value, st.message or ""))
print("\n".join(job.logs(limit=10)))  # reward and training metrics

ft = job.output_llm.deploy()  # Nova: on-demand deployment; mantle models are already live
print(
    ft.chat.completions.create(
        messages=[{"role": "user", "content": "What is 17 + 25? Answer with the number."}],
        max_tokens=300,
    )
    .choices[0]
    .message.content
)
ft.delete()
correct_sum.delete(region=request.route.region)  # the reward Lambda, in the training region
