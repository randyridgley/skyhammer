"""Token counting: the model's own tokenizer when available, a 4-chars-per-token estimate otherwise.

Real counts need ``pip install 'skyhammer[tokenizers]'`` and a public Hugging Face tokenizer
for the model (``ModelEntry.hf_repo``). Gated repos (Llama) also need ``HF_TOKEN``. Nova
has no public tokenizer, so its counts are always estimates.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import cache

CHARS_PER_TOKEN = 4

log = logging.getLogger("skyhammer")

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@cache
def tokenizer_counter(hf_repo: str | None) -> TokenCounter | None:
    """A counter backed by ``hf_repo``'s tokenizer, or None (not installed, gated, offline)."""
    if not hf_repo:
        return None
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError:
        log.debug("tokenizers not installed; estimating token counts")
        return None
    try:
        tok = Tokenizer.from_file(hf_hub_download(hf_repo, "tokenizer.json"))
    except Exception as e:  # gated repo without HF_TOKEN, offline, no tokenizer.json
        log.info("Using estimated token counts; couldn't load %s tokenizer (%s)", hf_repo, e)
        return None

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)

    return count
