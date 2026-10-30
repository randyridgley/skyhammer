"""Import Hugging Face weights you already have (in S3) and serve them on Bedrock.

Pre-flight checks (architecture, 128K context, size, files) run before the import job.
Cost: Bedrock import storage ($1.95/month) + on-demand usage; ~10-20 min.
Run: uv run python examples/22_bring_your_own_weights.py s3://my-bucket/my-model/
"""

import sys

from skyhammer import LLM

ft = LLM.import_weights(sys.argv[1], name="my-imported-model")
print(
    ft.chat.completions.create(messages=[{"role": "user", "content": "Hi!"}], max_tokens=40)
    .choices[0]
    .message.content
)
ft.delete()
