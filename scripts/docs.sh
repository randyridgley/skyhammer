#!/usr/bin/env bash
# Build the Skyhammer docs site from the SDK source, examples/ and the project docs.
#
#   scripts/docs.sh            # build static HTML into build/docs/site
#   scripts/docs.sh serve      # live preview at http://127.0.0.1:8000, rebuilt as you edit
#   scripts/docs.sh generate   # only write the Markdown + mkdocs.yml (build/docs/src)
#
# Pages are generated, not hand-written: edit docstrings, examples/ or README/ROADMAP/DESIGN
# and re-run. MkDocs runs through uvx, so it isn't a project dependency.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="build/docs"
CMD="${1:-build}"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

uv run --quiet python scripts/gen_docs.py "$OUT/src"

mkdocs() {
  NO_MKDOCS_2_WARNING=1 uvx --quiet --from 'mkdocs<2' \
    --with 'mkdocs-material>=9.5' --with 'mkdocstrings[python]>=0.26' --with ruff \
    mkdocs "$@" -f "$OUT/mkdocs.yml"
}

case "$CMD" in
  generate) ;;
  build) mkdocs build --strict -d "$PWD/$OUT/site" && echo "open $OUT/site/index.html" ;;
  serve) mkdocs serve "${@:2}" ;;  # e.g. scripts/docs.sh serve -a 127.0.0.1:8001
  *) echo "usage: $0 [build|serve|generate]" >&2; exit 2 ;;
esac
