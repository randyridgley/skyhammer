"""DA-10 / TR-20: images and video in training data, staged to S3 for Nova on Bedrock."""

import base64
import json
from pathlib import Path
from typing import Any

import boto3
import pytest
from moto import mock_aws

from skyhammer import Dataset
from skyhammer.backends.bedrock import BedrockNativeBackend
from skyhammer.catalog import Technique, get_model
from skyhammer.errors import DatasetValidationError, SkyhammerError
from skyhammer.session import Session

# A real 1x1 PNG.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def row(*media: dict[str, Any], answer: str = "red", text: str = "What color is it?") -> dict:
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": text}, *media]},
            {"role": "assistant", "content": answer},
        ]
    }


def image(url: str, **extra: Any) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url, **extra}}


def video(url: str) -> dict[str, Any]:
    return {"type": "video_url", "video_url": {"url": url}}


@pytest.fixture
def photos(tmp_path: Path) -> Path:
    """A JSONL file whose rows point at images next to it (relative paths)."""
    (tmp_path / "img").mkdir()
    for i in range(10):
        (tmp_path / "img" / f"{i}.png").write_bytes(PNG + bytes([i]))  # distinct contents
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(row(image(f"img/{i}.png"))) for i in range(10)) + "\n")
    return path


def errors(ds: Dataset, model: str = "nova-lite", technique: Technique = Technique.SFT) -> str:
    report = ds.validate(get_model(model), technique)
    return report.summary(limit=50)


def test_local_images_validate_for_nova_lite(photos: Path) -> None:
    ds = Dataset.from_file(photos)
    report = ds.validate(get_model("nova-lite"), Technique.SFT)
    assert report.ok, report.summary()
    assert "token counts cover text only" in report.warnings[0].message


def test_image_only_turns_are_not_empty() -> None:
    rows = [
        {
            "messages": [
                {"role": "human", "content": [image("data:image/png;base64," + _b64())]},
                {"role": "gpt", "content": "red"},
            ]
        }
    ] * 8
    ds = Dataset.from_records(rows, "inline")
    ds.prepare(get_model("nova-lite"), Technique.SFT, auto_fix=True)  # roles renamed, image kept
    assert ds.records()[0]["messages"][0]["content"][0]["type"] == "image_url"


def _b64(data: bytes = PNG) -> str:
    return base64.b64encode(data).decode()


def test_media_limits(tmp_path: Path) -> None:
    def one(*media: dict[str, Any]) -> Dataset:
        return Dataset.from_records([row(*media)] * 8, "x")

    png = tmp_path / "a.png"
    png.write_bytes(PNG)
    big = tmp_path / "big.png"
    with big.open("wb") as f:
        f.truncate(11 * 2**20)
    (tmp_path / "a.webp").write_bytes(b"RIFF\0\0\0\0WEBPVP8 ")

    assert "text only" in errors(one(image(str(png))), "nova-micro")
    assert "text only" in errors(one(image(str(png))), "qwen3-0.6b")
    assert "0 errors" in errors(one(image(str(png))))
    assert "format unknown isn't supported" in errors(one(image(str(tmp_path / "a.bmp"))))
    assert "file not found" in errors(one(image(str(tmp_path / "missing.png"))))
    assert "over the 10 MB image limit" in errors(one(image(str(big))))
    assert "11 images in one row; at most 10" in errors(one(*[image(str(png))] * 11))
    assert "download it first" in errors(one(image("https://example.com/a.png")))
    # WebP: Nova Lite yes, Nova 2 Lite no.
    webp = one(image(str(tmp_path / "a.webp")))
    assert "0 errors" in errors(webp, "nova-lite")
    assert "format webp isn't supported (png, jpeg, gif)" in errors(webp, "nova-2-lite")
    # The extension can be overridden when it's misleading.
    (tmp_path / "photo.bin").write_bytes(PNG)
    assert "0 errors" in errors(one(image(str(tmp_path / "photo.bin"), format="png")))
    # S3 media isn't downloaded to check; the format still is.
    assert "0 errors" in errors(one(image("s3://elsewhere/pics/a.jpg")))
    assert "for supervised fine-tuning (sft) only" in errors(
        Dataset.from_records(
            [{"messages": [{"role": "user", "content": [image(str(png))]}]}] * 100, "rft"
        ),
        "nova-2-lite",
        Technique.RFT,
    )
    in_answer = {
        "messages": [
            {"role": "user", "content": "draw it"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}, image(str(png))]},
        ]
    }
    assert "media goes in user messages" in errors(Dataset.from_records([in_answer] * 8, "x"))


def test_one_kind_of_media_per_dataset(tmp_path: Path) -> None:
    rows = [row(image("s3://b/a.png"))] * 8 + [row(video("s3://b/clip.mp4"))]
    report = Dataset.from_records(rows, "mixed").validate(get_model("nova-lite"), Technique.SFT)
    assert "mixes images and video (images from line 1, videos from line 9)" in report.summary()
    assert not report.fixable
    with pytest.raises(DatasetValidationError):
        Dataset.from_records(rows, "mixed").prepare(
            get_model("nova-lite"), Technique.SFT, auto_fix=True
        )


def test_changing_an_image_changes_the_version(photos: Path) -> None:
    before = Dataset.from_file(photos).sha256
    (photos.parent / "img" / "3.png").write_bytes(PNG + b"edited")
    assert Dataset.from_file(photos).sha256 != before


def test_upload_stages_media_next_to_the_dataset(photos: Path, session: Session) -> None:
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket = session.settings.bucket or ""
        s3.create_bucket(Bucket=bucket)
        s3.create_bucket(Bucket="elsewhere")
        s3.put_object(Bucket="elsewhere", Key="pics/far.jpg", Body=b"jpeg bytes")
        s3.put_object(Bucket=bucket, Key="mine/near.gif", Body=b"gif bytes")

        records = [json.loads(line) for line in photos.read_text().splitlines()]
        records += [
            row(image("s3://elsewhere/pics/far.jpg")),
            row(image(f"s3://{bucket}/mine/near.gif")),
            row(image("data:image/png;base64," + _b64(PNG + b"inline"))),
        ]
        ds = Dataset(photos.stem, list(enumerate(records, 1)), source=str(photos))
        ds.prepare(get_model("nova-lite"), Technique.SFT)
        backend = BedrockNativeBackend(session)
        uri = ds.upload(backend, Technique.SFT)

        key = uri.split("/", 3)[3]
        lines = [
            json.loads(x) for x in s3.get_object(Bucket=bucket, Key=key)["Body"].read().splitlines()
        ]
        blocks = [line["messages"][0]["content"] for line in lines]
        assert blocks[0][0] == {"text": "What color is it?"}
        img = blocks[0][1]["image"]
        assert img["format"] == "png"
        assert img["source"]["s3Location"]["bucketOwner"] == session.account_id
        staged = img["source"]["s3Location"]["uri"]
        assert staged.startswith(f"s3://{bucket}/media/") and staged.endswith(".png")
        assert (
            s3.get_object(Bucket=bucket, Key=staged.split("/", 3)[3])["Body"].read() == PNG + b"\0"
        )
        # Another bucket's object is copied in; one already in the bucket stays put.
        far = blocks[10][1]["image"]
        assert far["format"] == "jpeg" and "/media/s3/" in far["source"]["s3Location"]["uri"]
        assert (
            blocks[11][1]["image"]["source"]["s3Location"]["uri"] == f"s3://{bucket}/mine/near.gif"
        )
        assert blocks[12][1]["image"]["source"]["s3Location"]["uri"].startswith(
            f"s3://{bucket}/media/"
        )
        assert lines[0]["messages"][1]["content"] == [{"text": "red"}]

        media = s3.list_objects_v2(Bucket=bucket, Prefix="media/")["KeyCount"]
        assert media == 12  # 10 local + 1 copied + 1 inline, each once
        assert ds.upload(backend, Technique.SFT) == uri  # idempotent
        assert s3.list_objects_v2(Bucket=bucket, Prefix="media/")["KeyCount"] == media
        assert ds.records()[0]["messages"][0]["content"][1]["image_url"]["url"] == "img/0.png"


def test_rows_cant_pass_off_other_files_as_media(tmp_path: Path) -> None:
    secret = tmp_path / "credentials"
    secret.write_text("[default]\naws_secret_access_key = abc\n")
    disguised = Dataset.from_records([row(image(str(secret), format="png"))] * 8, "x")
    assert "contents aren't a png image" in errors(disguised)
    fake_inline = "data:image/png;base64," + _b64(b"not an image at all")
    assert "contents aren't a png image" in errors(
        Dataset.from_records([row(image(fake_inline))] * 8, "x")
    )


def test_remote_datasets_cant_reach_local_files_or_other_buckets(tmp_path: Path) -> None:
    png = tmp_path / "a.png"
    png.write_bytes(PNG)

    def remote(source: str, *media: dict[str, Any]) -> Dataset:
        return Dataset("shared", list(enumerate([row(*media)] * 8, 1)), source=source)

    from_s3 = "s3://data-bucket/sets/train.jsonl"
    assert "can't use local files" in errors(remote(from_s3, image(str(png))))
    assert "can't use local files" in errors(remote(from_s3, image(f"file://{png}")))
    assert "can only use media in its own bucket" in errors(
        remote(from_s3, image("s3://payroll/scan.png"))
    )
    assert "0 errors" in errors(remote(from_s3, image("s3://data-bucket/img/a.png")))
    assert "can only use media in its own bucket" in errors(
        remote("hf://someone/vqa", image("s3://data-bucket/img/a.png"))
    )
    inline = image("data:image/png;base64," + _b64())
    assert "0 errors" in errors(remote("hf://someone/vqa", inline))
    # Mixing keeps the restriction; local parts keep their own (relative) images.
    (tmp_path / "local.jsonl").write_text(json.dumps(row(image("a.png"))) + "\n")
    local = Dataset.from_file(tmp_path / "local.jsonl")
    mixed = Dataset.mix({remote("hf://someone/vqa", inline): 1, local: 1}, n=16, repeat=True)
    assert mixed.source == "hf://someone/vqa"
    assert "can't use local files" in errors(mixed)
    assert "0 errors" in errors(Dataset.mix({local: 1}, n=8, repeat=True))


def test_upload_rechecks_files_changed_after_validation(photos: Path, session: Session) -> None:
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(
            Bucket=session.settings.bucket or ""
        )
        ds = Dataset.from_file(photos)
        ds.prepare(get_model("nova-lite"), Technique.SFT)
        (photos.parent / "img" / "4.png").write_text("swapped for something else")
        with pytest.raises(SkyhammerError, match="contents aren't a png image"):
            ds.upload(BedrockNativeBackend(session), Technique.SFT)
