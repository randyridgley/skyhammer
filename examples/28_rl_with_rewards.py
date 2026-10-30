"""Reinforcement learning with your own reward function (RLVR) on an open-weight model.

Write the reward as a plain Python function; Skyhammer deploys it as a Lambda, registers it
with SageMaker, trains, and imports the result into Bedrock. RL is billed by the hour, so
every job has a runtime cap (max_hours) and the estimate is the most it can cost.
Cost: ~$80/hour of training, capped at max_hours (the run below took ~5 min of training);
~30 min end to end. Run: uv run python examples/28_rl_with_rewards.py
"""

from skyhammer import LLM, Dataset, grader, reference_answer


@grader
def correct_sum(sample: dict, output: str) -> float:
    """1.0 if the expected number appears in the answer. Runs inside Lambda, so use only
    the standard library (``reference_answer`` is bundled with it)."""
    expected = reference_answer(sample).strip()
    return 1.0 if expected and expected in output else 0.0


print(correct_sum({"answer": "7"}, "3 + 4 = 7"))  # test the reward locally first: 1.0

# No rule to check? Use an LLM judge as the reward (RLAIF). Skyhammer deploys it as a Lambda
# that asks the judge model on Bedrock to score each answer 1-5 against your rubric:
#   reward = skyhammer.Grader.llm_judge("nova-pro", "Reward correct, concise, polite answers.")
#   LLM("qwen3-0.6b").create_rft_job(prompts, grader=reward, output_model="polite-rl")

problems = Dataset.from_records(
    [
        {
            "messages": [
                {"role": "user", "content": f"What is {a} + {b}? Answer with the number."}
            ],
            "answer": str(a + b),  # the reference the reward function checks against
        }
        for a in range(20)
        for b in range(20)
    ],
    name="sums",
)

job = LLM("qwen3-0.6b").create_rft_job(
    problems,
    grader=correct_sum,
    output_model="sums-rl",
    epochs=1,
    max_hours=1,  # hard stop; the estimate (and max_cost_usd) uses this cap
    max_cost_usd=100,
)
job.wait(on_status=lambda st: print(" ", st.state.value, st.message or ""))
tuned = job.output_llm.deploy()
reply = tuned.chat.completions.create(
    messages=[{"role": "user", "content": "What is 3 + 4? Answer with the number."}],
    max_tokens=200,
)
print(reply.choices[0].message.content)
tuned.delete()
correct_sum.delete()  # the reward Lambda
