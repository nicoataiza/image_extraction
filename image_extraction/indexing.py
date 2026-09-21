"""Resumable descriptor checkpoints and portable, verified FAISS artifacts."""

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import sys
import time

import numpy as np
import PIL

from .dataset import ImageDataset, ImageLoadError
from .descriptors import SpatialDescriptor
from .download import write_json
from .search import ExactSearch, require_faiss

INDEX_SCHEMA = 1


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(dataset: ImageDataset) -> list[dict]:
    result = []
    for path in dataset.paths:
        stat = path.stat()
        result.append({"relative_path": path.relative_to(dataset.root).as_posix(),
                       "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return result


def ensure_separate(first: Path, second: Path) -> None:
    if first == second or first in second.parents or second in first.parents:
        raise ValueError(f"Directories must be separate, not nested: {first} and {second}")


def build_index(images, index, *, search_device="auto", batch_size=64, rebuild=False,
                descriptor=None) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    descriptor = descriptor or SpatialDescriptor()
    dataset = ImageDataset(images)
    destination = Path(index).expanduser().resolve()
    ensure_separate(dataset.root, destination)
    faiss = require_faiss()
    try:
        from filelock import FileLock, Timeout
    except ImportError as error:
        raise ValueError("Install .[retrieval] or .[video] for indexing dependencies") from error
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(destination / ".index.lock"), timeout=0):
            return _build(dataset, destination, descriptor, faiss, search_device, batch_size, rebuild)
    except Timeout as error:
        raise ValueError("Another process is building this index") from error


def _build(dataset, destination, descriptor, faiss, search_device, batch_size, rebuild):
    started = time.perf_counter()
    manifest_path = destination / "manifest.json"
    database_path = destination / "entries.sqlite"
    if not database_path.exists() and any(p.name != ".index.lock" for p in destination.iterdir()):
        raise ValueError("Index directory is not empty and has no descriptor checkpoint; use a new directory")
    entries = inventory(dataset)
    fingerprint = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    settings = {"schema_version": INDEX_SCHEMA, "images_root": str(dataset.root),
                "descriptor": descriptor.metadata(), "inventory_sha256": fingerprint,
                "extraction_versions": {"numpy": np.__version__, "pillow": PIL.__version__}}
    with closing(sqlite3.connect(database_path)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)")
        db.execute("""CREATE TABLE IF NOT EXISTS entries (
            path TEXT PRIMARY KEY, metadata TEXT NOT NULL, vector BLOB, error TEXT)""")
        previous = db.execute("SELECT value FROM settings WHERE id=1").fetchone()
        if previous and json.loads(previous[0]) != settings and not rebuild:
            raise ValueError("Index source, descriptor, or extraction versions changed; use --rebuild or a new index directory")
        # Check backend before invalidating a completed artifact.
        backend = ExactSearch(faiss.IndexFlatIP(descriptor.dimension), search_device)
        manifest = {**settings, "status": "indexing", "candidate_images": len(entries),
                    "started_at": datetime.now(timezone.utc).isoformat()}
        write_json(manifest_path, manifest)
        if rebuild or previous is None:
            with db:
                db.execute("DELETE FROM entries")
                db.execute("INSERT OR REPLACE INTO settings VALUES (1, ?)", (json.dumps(settings),))
        reused = extracted = 0
        extraction_seconds = 0.0
        for number, entry in enumerate(entries):
            cached = db.execute("SELECT metadata FROM entries WHERE path=?", (entry["relative_path"],)).fetchone()
            if cached:
                saved = json.loads(cached[0])
                if any(saved.get(key) != value for key, value in entry.items()):
                    raise ValueError("Descriptor checkpoint does not match source inventory; use --rebuild")
                reused += 1
                continue
            tick = time.perf_counter()
            vector, failure = None, None
            metadata = dict(entry)
            try:
                sample = dataset[number]
            except ImageLoadError as error:
                failure = error.reason
                print(str(error), file=sys.stderr)
            else:
                try:
                    vector = descriptor.extract(sample.image)
                    if (vector.shape != (descriptor.dimension,) or vector.dtype != np.float32
                            or not np.isfinite(vector).all()):
                        raise ValueError("Descriptor returned invalid data")
                    norm = float(np.linalg.norm(vector))
                    if norm != 0 and not np.isclose(norm, 1, atol=1e-5):
                        raise ValueError("Descriptor is not normalized")
                    metadata.update(width=sample.source_size[0], height=sample.source_size[1], zero_vector=norm == 0)
                finally:
                    sample.close()
            stat = dataset.paths[number].stat()
            if (stat.st_size, stat.st_mtime_ns) != (entry["size_bytes"], entry["mtime_ns"]):
                raise ValueError("An image changed during extraction; rerun with --rebuild")
            # One transaction per file: interruption loses at most the current image.
            with db:
                db.execute("INSERT INTO entries VALUES (?, ?, ?, ?)", (
                    entry["relative_path"], json.dumps(metadata),
                    None if vector is None else vector.tobytes(), failure))
            extraction_seconds += time.perf_counter() - tick
            extracted += 1
            if (number + 1) % 100 == 0:
                print(f"Indexed {number + 1}/{len(entries)} candidates", file=sys.stderr)
        if inventory(ImageDataset(dataset.root)) != entries:
            raise ValueError("Image collection changed during indexing; rerun with --rebuild")
        add_seconds = 0.0
        usable = invalid = zero_vectors = 0
        batch = []
        images_tmp, errors_tmp = destination / "images.jsonl.tmp", destination / "errors.jsonl.tmp"
        with images_tmp.open("w", encoding="utf-8") as image_stream, errors_tmp.open("w", encoding="utf-8") as error_stream:
            for metadata_json, blob, error in db.execute("SELECT metadata, vector, error FROM entries ORDER BY path"):
                metadata = json.loads(metadata_json)
                if error is not None:
                    invalid += 1
                    error_stream.write(json.dumps({**metadata, "error": error}) + "\n")
                    continue
                if blob is None or len(blob) != descriptor.dimension * 4:
                    raise ValueError("Invalid descriptor checkpoint; use --rebuild")
                image_stream.write(json.dumps({"id": usable, **metadata}) + "\n")
                usable += 1
                zero_vectors += metadata["zero_vector"]
                batch.append(np.frombuffer(blob, dtype=np.float32))
                if len(batch) >= batch_size:
                    tick = time.perf_counter()
                    backend.add(np.stack(batch))
                    add_seconds += time.perf_counter() - tick
                    batch.clear()
            if batch:
                tick = time.perf_counter()
                backend.add(np.stack(batch))
                add_seconds += time.perf_counter() - tick
        images_tmp.replace(destination / "images.jsonl")
        errors_tmp.replace(destination / "errors.jsonl")
        manifest.update(usable_images=usable, invalid_images=invalid, zero_vectors=zero_vectors,
                        reused_candidates=reused, processed_candidates=extracted)
        if not usable:
            write_json(manifest_path, {**manifest, "status": "failed", "error": "No usable images"})
            raise ValueError("No usable images; see errors.jsonl")
        tick = time.perf_counter()
        temporary = destination / "index.faiss.tmp"
        faiss.write_index(backend.cpu_index(), str(temporary))
        temporary.replace(destination / "index.faiss")
        save_seconds = time.perf_counter() - tick
        manifest.update(status="complete", search=backend.metadata(),
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        python_version=platform.python_version(), batch_size=batch_size,
                        timings_seconds={"extraction_and_checkpoint": extraction_seconds,
                                         "vector_index_add": add_seconds, "index_save": save_seconds,
                                         "total": time.perf_counter() - started},
                        descriptor_bytes=usable * descriptor.dimension * 4,
                        files={name: {"sha256": file_hash(destination / name),
                                      "bytes": (destination / name).stat().st_size}
                               for name in ("index.faiss", "images.jsonl", "errors.jsonl")})
        write_json(manifest_path, manifest)
        return manifest


def load_index(index, *, search_device="auto"):
    destination = Path(index).expanduser().resolve()
    if not (destination / "manifest.json").is_file():
        raise ValueError(f"No index manifest found in {destination}; run index first")
    try:
        from filelock import FileLock, Timeout
    except ImportError as error:
        raise ValueError("Install .[retrieval] or .[video] for indexing dependencies") from error
    try:
        # Hold the build lock until all artifacts have been verified and loaded.
        with FileLock(str(destination / ".index.lock"), timeout=0):
            return _load_index(destination, search_device)
    except Timeout as error:
        raise ValueError("Index is currently being rebuilt; retry after indexing completes") from error


def _load_index(destination, search_device):
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest.get("schema_version") != INDEX_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("Index is incompatible or incomplete; build/resume it before querying")
    try:
        descriptor = SpatialDescriptor(**manifest["descriptor"]["parameters"])
        if descriptor.metadata() != manifest["descriptor"]:
            raise ValueError("Index descriptor is incompatible; rebuild the index")
        for name in ("index.faiss", "images.jsonl", "errors.jsonl"):
            if file_hash(destination / name) != manifest["files"][name]["sha256"]:
                raise ValueError(f"Index artifact checksum mismatch: {name}; rerun index")
        images = [json.loads(line) for line in (destination / "images.jsonl").read_text().splitlines()]
        for number, entry in enumerate(images):
            path = Path(entry["relative_path"])
            if entry["id"] != number or path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid image mapping in index")
        faiss = require_faiss()
        cpu = faiss.read_index(str(destination / "index.faiss"))
        if (cpu.d != descriptor.dimension or cpu.ntotal != len(images) or not images
                or len(images) != manifest["usable_images"] or cpu.metric_type != faiss.METRIC_INNER_PRODUCT
                or not isinstance(cpu, faiss.IndexFlat)):
            raise ValueError("FAISS index does not match its metadata")
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid index manifest or image mapping") from error
    return manifest, images, descriptor, ExactSearch(cpu, search_device)
