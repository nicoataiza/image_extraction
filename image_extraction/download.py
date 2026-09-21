"""Download a pinned dataset snapshot without duplicating it in a blob cache."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from .dataset import IMAGE_EXTENSIONS

DATASET_ID = "Smily6820/car-images"
DATASET_REVISION = "0dca057af12e29f55febea065eb96b43ab5d18f9"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"


def write_json(path: Path, value: dict) -> None:
    """Publish a complete JSON file, preserving the previous one on write failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def download_dataset(
    destination: str | Path = "data/car-images",
    *,
    revision: str = DATASET_REVISION,
    workers: int = 4,
) -> dict:
    """Download/resume one revision and preserve its provenance.

    Rerun the same command after interruption. Use a separate directory for a
    different revision to avoid mixing old and new files. Completion here means
    files were downloaded, not that their image contents passed inspection.
    """
    if workers <= 0:
        raise ValueError("workers must be positive")
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Set defaults before importing the Hub client, including Xet's cache path.
    os.environ.setdefault("HF_HOME", str(destination / ".cache" / "huggingface"))
    os.environ.setdefault("HF_XET_CACHE", str(destination / ".cache" / "xet"))
    from huggingface_hub import HfApi, snapshot_download

    metadata_path = destination / "dataset-source.json"
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    info = HfApi().dataset_info(DATASET_ID, revision=revision, files_metadata=True)
    if previous and (previous["repo_id"] != DATASET_ID or previous["revision"] != info.sha):
        raise ValueError("Destination contains another dataset revision; use a new directory")

    image_files = sorted(
        entry.rfilename for entry in info.siblings
        if Path(entry.rfilename).suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError("The dataset revision contains no supported image files")
    selected = image_files + [entry.rfilename for entry in info.siblings if entry.rfilename == "README.md"]
    metadata = {
        "repo_id": DATASET_ID,
        "url": DATASET_URL,
        "revision": info.sha,
        "license": "cc-by-4.0",
        "image_count": len(image_files),
        "image_files": image_files,
        "status": "downloading",
    }
    write_json(metadata_path, metadata)
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=info.sha,
        local_dir=destination,
        allow_patterns=selected,
        max_workers=workers,
    )
    expected_sizes = {entry.rfilename: entry.size for entry in info.siblings}
    for name in selected:
        path = destination / name
        if not path.is_file():
            raise ValueError(f"Download incomplete: missing {name}")
        if expected_sizes[name] is not None and path.stat().st_size != expected_sizes[name]:
            raise ValueError(f"Download size mismatch: {name}")
    metadata.update(
        status="downloaded",
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        image_bytes=sum((destination / name).stat().st_size for name in image_files),
    )
    write_json(metadata_path, metadata)
    return metadata
