"""MkDocs hook: regenerate the docs pages before every build, so `scripts/docs.sh serve`
picks up changes to src/, examples/ and the project docs as you edit them.

MkDocs runs in its own uvx environment, so generate in the project environment instead.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def on_pre_build(config: Any) -> None:
    subprocess.run(
        ["uv", "run", "--quiet", "python", "scripts/gen_docs.py", config["docs_dir"]],
        cwd=ROOT,
        check=True,
    )
