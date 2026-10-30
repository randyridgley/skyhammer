# Skyhammer: project instructions

*Forge, tune and call models on AWS, without the heavy lifting.* Python SDK + `skyhammer`
CLI over Amazon Bedrock, SageMaker AI and related AWS services. See ROADMAP.md (scope and
status), DESIGN.md (architecture), examples/ (usage).

## Every feature ships with an example (required)

When you add or complete a user-facing feature:

1. Add or extend a runnable Python example in `examples/` showing the simplest real use
   (`examples/NN_short_name.py`). Keep it short, put the cost and time in the docstring
   when it touches billable AWS resources, and clean up anything it creates.
2. List the example in the index table in `examples/README.md`, including every roadmap
   ID it demonstrates (for example `SI-7`).
3. A feature is only ✅ in ROADMAP.md when it has an example. `tests/unit/test_examples.py`
   enforces this: every ✅ user-facing ID (DX, DA, TR, EV, SI, SV, OP, SE) must appear in
   `examples/README.md`, and every example must compile and use only public imports.

## Other conventions

- Verify AWS behavior live before marking roadmap items ✅ (🔴 items), and record real
  responses as test fixtures. Unit-test fakes have hidden real bugs several times.
- Never touch AWS resources without the `skyhammer:name` tag; this account hosts other
  projects. Confirm with the user before creating new billable resources.
- Checks before finishing: `uv run ruff check . && uv run ruff format --check . &&
  uv run mypy src && uv run pytest -q && scripts/docs.sh`.
- Docs are generated (`scripts/gen_docs.py`); never hand-edit `build/docs`. A feature's docs
  page comes from its example's docstring and the `examples/README.md` row, and its API
  page from docstrings, so write those for readers.
