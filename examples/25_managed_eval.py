"""Managed evaluation on Bedrock: a judge model scores your model's answers (LLM as a judge).

Use it for formal reports at scale (up to 1,000 prompts per job, built-in metrics such as
correctness, completeness, helpfulness, faithfulness, harmfulness). For quick local
iteration with your own graders, see 19_evaluation.py.
Cost: a few cents (the model's answers + the judge's tokens); ~10-30 min.
Run: uv run python examples/25_managed_eval.py
CLI: skyhammer eval judge nova-micro --dataset eval.jsonl --metric correctness --judge nova-pro
"""

from skyhammer import LLM

records = [
    {"messages": [{"role": "user", "content": "Capital of France?"}], "answer": "Paris"},
    {"messages": [{"role": "user", "content": "What is 12 * 12?"}], "answer": "144"},
    {"messages": [{"role": "user", "content": "Who wrote Hamlet?"}], "answer": "Shakespeare"},
]

job = LLM("nova-micro").create_eval_job(  # works the same for your fine-tuned models
    records, metrics=["correctness", "completeness", "helpfulness"], judge="nova-pro"
)
print("evaluation job:", job.name)  # reattach later with EvalJob.get(job.name)
job.wait(on_status=lambda st: print(" ", st.status))

report = job.results()
print({metric: round(score, 2) for metric, score in report.means.items()})
for r in report.records:
    print(f"{r.prompt!r} -> {r.response!r}: {r.scores}")
    print("   judge:", r.explanations.get("correctness", "")[:120])
