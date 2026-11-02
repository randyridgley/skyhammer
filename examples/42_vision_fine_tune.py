"""Vision fine-tuning: teach Nova Lite to read your images, then ask it about a new one.

Rows hold images the same way ``chat`` does: ``image_url`` content parts whose URL is a
local path (relative to the JSONL file), a base64 ``data:`` URL or an ``s3://`` URI.
Validation checks every image against the model's limits (format, size, count per row,
one kind of media per dataset) before anything is uploaded. On upload the images go to
the Skyhammer bucket next to the dataset, content-addressed so each uploads once, and
the rows are rewritten to the S3 references Bedrock expects. Video works the same way
with ``video_url`` parts (one video per row, up to 90 seconds).

Nova Lite, Nova Pro and Nova 2 Lite take images and video; Nova Micro is text only.

Cost: a few cents of training (32x32 images, 48 rows) + on-demand inference; ~45 min,
mostly Bedrock job startup. Run: uv run python examples/42_vision_fine_tune.py
"""

import base64
import json
import struct
import tempfile
import zlib
from pathlib import Path

from skyhammer import LLM, Dataset


def png(rgb: tuple[int, int, int], size: int = 32) -> bytes:
    """A solid-color PNG (stand-in for your photos)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\0" + bytes(rgb) * size for _ in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


# Paint-chip codes: the model has to learn your labels, not just name the color.
CHIPS = {"RX-1": (200, 30, 30), "GN-4": (30, 160, 60), "BL-7": (40, 70, 200)}
SYSTEM = "You read paint chips. Answer with the chip code only."

folder = Path(tempfile.mkdtemp())
rows = []
for code, (r, g, b) in CHIPS.items():
    for shade in range(16):
        name = f"{code}-{shade}.png"
        (folder / name).write_bytes(png((r + shade, g + shade, b + shade)))
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Which chip is this?"},
                            {"type": "image_url", "image_url": {"url": name}},
                        ],
                    },
                    {"role": "assistant", "content": code},
                ]
            }
        )
(folder / "chips.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

ds = Dataset.from_file(folder / "chips.jsonl", name="paint-chips")
job = LLM("nova-lite").create_sft_job(ds, output_model="paint-chips-nova", epochs=3)
print("training", job.name)
job.wait(on_status=lambda st: print(" ", st.state.value))

tuned = job.output_llm.deploy()
new_chip = base64.b64encode(png((215, 45, 45))).decode()  # a red it hasn't seen
reply = tuned.chat.completions.create(
    messages=[
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Which chip is this?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{new_chip}"}},
            ],
        },
    ],
    max_tokens=10,
)
print(reply.choices[0].message.content)  # RX-1
tuned.delete()
