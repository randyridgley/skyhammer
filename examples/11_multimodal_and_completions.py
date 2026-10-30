"""Images and documents in chat, plus the classic completions API.

Cost: a fraction of a cent. Run: uv run python examples/11_multimodal_and_completions.py
"""

import base64
import struct
import zlib

from skyhammer import LLM


def red_square_png(size: int = 8) -> bytes:
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


image = base64.b64encode(red_square_png()).decode()
r = LLM("claude-haiku-4-5").chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image? One word."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
            ],
        }
    ],
    max_tokens=10,
)
print("image:", r.choices[0].message.content)

doc = base64.b64encode(b"Q3 revenue grew 12% to $4.2M; churn fell to 2.1%.").decode()
r = LLM("nova-lite").chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What was Q3 revenue?"},
                {
                    "type": "file",
                    "file": {
                        "file_data": f"data:text/plain;base64,{doc}",
                        "filename": "q3-report.txt",
                    },
                },
            ],
        }
    ],
    max_tokens=30,
)
print("document:", r.choices[0].message.content)

c = LLM("nova-micro").completions.create(prompt="Name three primary colors:", max_tokens=20)
print("completion:", c.choices[0].text)
