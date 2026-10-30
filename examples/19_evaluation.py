"""Evaluate and compare models with built-in, Python, LLM-judge or Lambda graders.

Cost: a fraction of a cent. Run: uv run python examples/19_evaluation.py
CLI: skyhammer eval compare nova-micro qwen3-32b --dataset eval.jsonl --grader contains
"""

from skyhammer import LLM, Grader, grader
from skyhammer.evaluation import evaluate

records = [
    {"messages": [{"role": "user", "content": "Capital of France? One word."}], "answer": "Paris"},
    {"messages": [{"role": "user", "content": "12 * 12? Just the number."}], "answer": "144"},
    {"messages": [{"role": "user", "content": "Red planet? One word."}], "answer": "Mars"},
]


@grader
def short_answer(sample: dict, output: str) -> float:
    """Your own scoring logic; must return a score in [0, 1]."""
    return 1.0 if len(output.split()) <= 3 else 0.0


# A model as the judge: scores 1-5 against your rubric (and the reference answer), as 0-1.
judge = Grader.llm_judge("nova-pro", rubric="Correct, and no longer than needed.")

graders = [Grader.builtin("contains"), Grader.builtin("exact_match"), short_answer, judge]
# Grader.from_lambda("arn:aws:lambda:...")  -> your Lambda gets {"sample", "output"}

for name in ["nova-micro", "qwen3-32b", "claude-haiku-4-5"]:  # e.g. base vs your fine-tune
    result = evaluate(LLM(name), records, graders, max_tokens=20)
    print(f"{name:18}", {k: round(v, 2) for k, v in result.means.items()})

# Deploy a Python grader as a Lambda (idempotent; free when idle). RFT jobs do this
# automatically; the same function scores RL rollouts and evals.
deployed = short_answer.deploy()
print("lambda:", deployed.lambda_arn, deployed(records[0], "Paris"))
short_answer.delete()
