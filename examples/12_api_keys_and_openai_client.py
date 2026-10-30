"""No AWS profile? Use a Bedrock API key, or the official OpenAI SDK against Bedrock.

Requires: pip install 'skyhammer[openai]'. Cost: a fraction of a cent.
Run: uv run python examples/12_api_keys_and_openai_client.py
"""

import skyhammer

# A short-term (<= 12h) Bedrock API key derived from your current AWS credentials. Hand it
# to an app as SKYHAMMER_API_KEY (or AWS_BEARER_TOKEN_BEDROCK); no AWS profile needed there.
key = skyhammer.generate_api_key(hours=1)
print("key:", key[:28] + "...")

# The OpenAI SDK, pointed at Bedrock's OpenAI-compatible endpoint.
client = skyhammer.openai_client(api_key=key)
r = client.chat.completions.create(
    model="openai.gpt-oss-20b-1:0",
    messages=[{"role": "user", "content": "Say hello in five words."}],
    max_tokens=200,
)
print(r.choices[0].message.content)
