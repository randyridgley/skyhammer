"""Every completed user-facing feature must have a runnable example (see CLAUDE.md)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
INDEX = (EXAMPLES / "README.md").read_text()
USER_FACING = ("DX", "DA", "TR", "EV", "SI", "SV", "OP", "SE")


def done_feature_ids() -> list[str]:
    ids = []
    for line in (ROOT / "ROADMAP.md").read_text().splitlines():
        m = re.match(r"^\| ((?:" + "|".join(USER_FACING) + r")-\d+) \|", line)
        if m and " ✅ " in line:
            ids.append(m.group(1))
    return ids


@pytest.mark.parametrize("feature", done_feature_ids())
def test_every_completed_feature_has_an_example(feature: str) -> None:
    assert re.search(rf"\b{re.escape(feature)}\b", INDEX), (
        f"{feature} is ✅ in ROADMAP.md but no example in examples/README.md covers it. "
        "Add an examples/NN_*.py and list the ID in its Covers column."
    )


EXAMPLE_FILES = sorted(p for p in EXAMPLES.iterdir() if p.suffix in (".py", ".yaml"))


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_every_example_is_indexed(path: Path) -> None:
    assert f"]({path.name})" in INDEX, f"{path.name} is missing from examples/README.md"


@pytest.mark.parametrize(
    "path", [p for p in EXAMPLE_FILES if p.suffix == ".py"], ids=lambda p: p.name
)
def test_examples_compile_and_use_public_api(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    assert ast.get_docstring(tree), f"{path.name} needs a docstring (what, cost, how to run)"
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("skyhammer"):
            assert not any(part.startswith("_") for part in node.module.split(".")), node.module
            for alias in node.names:
                assert not alias.name.startswith("_"), f"{path.name} imports private {alias.name}"
                module = __import__(node.module, fromlist=[alias.name])
                assert hasattr(module, alias.name), f"{node.module}.{alias.name} doesn't exist"


def test_yaml_examples_are_valid_specs() -> None:
    from skyhammer.spec import load_spec

    for path in EXAMPLES.glob("*.yaml"):
        load_spec(path)


def test_roadmap_ids_are_unique() -> None:
    ids = re.findall(r"^\| ([A-Z]{2}-\d+) \|", (ROOT / "ROADMAP.md").read_text(), flags=re.M)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f"duplicate roadmap IDs: {dupes}"
