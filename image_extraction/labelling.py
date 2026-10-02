"""Zero-shot, unreviewed image annotations from frozen SigLIP 2 vectors."""

from collections import Counter
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps

from .dataset import ImageDataset
from .download import write_json
from .indexing import ensure_separate, file_hash, load_index
from .semantic import DEFAULT_CACHE, MODEL_ID, MODEL_REVISION, SemanticDescriptor

# Broad scene descriptions compete separately from the source's fine part taxonomy.
VIEWS = {
    "front_exterior": ["the front exterior of a car", "a car viewed from the front corner"],
    "rear_exterior": ["the rear exterior of a car", "a car viewed from the rear corner"],
    "side_exterior": ["the side profile of a car", "a car viewed from the side showing its doors"],
    "engine_bay": ["a car engine bay under the open bonnet", "an engine installed under the hood of a car"],
    "dashboard": ["a car dashboard and steering wheel", "the front cabin of a car showing the dashboard"],
    "instrument_cluster": ["a close-up of a car speedometer and odometer", "a car instrument cluster showing gauges and warning lights"],
    "front_seats": ["the front seats inside a car", "a car front seat viewed through an open door"],
    "rear_seats": ["the back seats inside a car", "the rear passenger seating area of a car"],
    "door_interior": ["the inside trim panel of a car door", "a car door interior with its handle and window controls"],
    "cargo_area": ["the open boot cargo compartment of a car", "the inside of a car trunk or SUV cargo area"],
    "ute_tray": ["the cargo tray of a pickup truck", "the open load bed of a ute"],
    "wheel_tyre": ["a close-up of a car wheel and tyre", "a car alloy wheel rim and tire"],
    "headlight": ["a close-up of a car headlight", "the front lamp of a car"],
    "taillight": ["a close-up of a car rear tail light", "the rear lamp of a car"],
    "exterior_mirror": ["a close-up of a car side mirror", "a car door mirror"],
    "roof": ["the roof of a car viewed from above", "a car roof and sunroof"],
    "underbody": ["the underside of a car on a hoist", "a car underbody showing suspension and exhaust"],
    "identification_plate": ["a close-up of a vehicle identification plate with printed text", "a car VIN sticker or compliance label"],
    "registration_plate": ["a close-up of a car registration number plate", "a vehicle license plate"],
    "infotainment_controls": ["a close-up of a car radio and climate controls", "a car infotainment screen in the centre dashboard"],
    "gear_shifter_console": ["a car gear shift lever and centre console", "a close-up of the transmission shifter inside a car"],
    "steering_wheel": ["a close-up of a car steering wheel", "a steering wheel inside a vehicle"],
    "interior_headliner": ["the interior roof lining of a car", "a car headliner and interior ceiling lights"],
    "loose_part": ["a removed car part on a workshop floor", "an isolated automotive spare part"],
    "document": ["a document with printed text", "a vehicle paperwork sheet or handwritten notes"],
    "other_unclear": ["a blurry unrecognizable photograph", "an unrelated scene without a clearly visible car part"],
}


def make_taxonomy(source):
    labels = [{"id": key, "name": key.replace("_", " "), "family": "view",
               "prompts": [f"this is a photo of {text}." for text in descriptions]}
              for key, descriptions in VIEWS.items()]
    parts = {}
    skipped = 0
    with Path(source).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"part_type_code", "part_type_name"} <= set(reader.fieldnames or []):
            raise ValueError("Part CSV needs part_type_code and part_type_name columns")
        for row in reader:
            code, name = row["part_type_code"].strip(), row["part_type_name"].strip()
            if not code or not name:
                skipped += 1
                continue
            if code in parts and parts[code] != name:
                raise ValueError(f"Conflicting names for part code {code}")
            parts[code] = name
    if len(parts) < 3:
        raise ValueError("At least three named part types are required")
    for code, name in sorted(parts.items()):
        readable = name.lower().replace("a c ", "air conditioning ")
        readable = readable.replace("instrumentcluster", "instrument cluster")
        labels.append({"id": code, "name": name, "family": "part", "prompts": [
            f"this is a photo of a car {readable}.",
            f"this is a close-up photo of an automotive {readable}."]})
    return labels, skipped


def encode_prompts(labels, *, device="auto", cache_dir=None, batch_size=32):
    """The pinned text tower's projected pooler output matches stored vision vectors."""
    if device not in ("auto", "cpu", "cuda") or batch_size <= 0:
        raise ValueError("Invalid device or text batch size")
    import torch
    from transformers import AutoTokenizer, Siglip2TextModel

    actual, fallback = "cpu", None
    if device != "cpu":
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch reports CUDA unavailable")
            probe = torch.ones((2, 2), device="cuda")
            _ = probe @ probe
            torch.cuda.synchronize()
            actual = "cuda"
        except RuntimeError as error:
            if device == "cuda":
                raise ValueError(f"CUDA text encoding unavailable: {error}") from error
            fallback = str(error)
            print(f"Text encoding using CPU: {error}", file=sys.stderr)
    common = {"revision": MODEL_REVISION, "cache_dir": str(cache_dir or DEFAULT_CACHE),
              "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **common)
    model, loading = Siglip2TextModel.from_pretrained(
        MODEL_ID, dtype=torch.float32, use_safetensors=True, attn_implementation="sdpa",
        output_loading_info=True, **common)
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Text encoder weights did not load completely")
    model.eval().to(actual)
    prompts = [prompt.lower() for label in labels for prompt in label["prompts"]]
    chunks = []
    for start in range(0, len(prompts), batch_size):
        # SigLIP 2 is trained on lowercase text padded to exactly 64 tokens.
        inputs = tokenizer(prompts[start:start + batch_size], padding="max_length",
                           max_length=64, truncation=True, return_tensors="pt").to(actual)
        with torch.inference_mode():
            vectors = model(**inputs).pooler_output.float()
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
            chunks.append(vectors.cpu().numpy())
    vectors = np.concatenate(chunks)
    validate_vectors(vectors, len(prompts), 768)
    return vectors, {"requested_device": device, "device": actual, "fallback_reason": fallback,
                     "tokenizer": type(tokenizer).__name__, "padding": "max_length",
                     "max_length": 64, "lowercase": True, "dtype": "float32"}


def validate_vectors(vectors, count, dimension):
    if (vectors.shape != (count, dimension) or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
        raise ValueError("Expected finite nonzero unit embedding vectors")


def score_labels(images, prompts, labels):
    """Mean prompt cosine per label; avoid favouring labels with more prompts."""
    similarities = images @ prompts.T
    columns, offset = [], 0
    for label in labels:
        count = len(label["prompts"])
        if not count:
            raise ValueError("Every label needs at least one prompt")
        columns.append(similarities[:, offset:offset + count].mean(axis=1))
        offset += count
    if offset != len(prompts):
        raise ValueError("Prompt/vector mapping differs")
    scores = np.stack(columns, axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite label scores")
    return scores


def rank_scores(scores, labels, family):
    columns = [i for i, label in enumerate(labels) if label["family"] == family]
    if len(columns) < 3:
        raise ValueError("Need at least three labels per family")
    # Six decimals and taxonomy order resolve ties reproducibly on saved scores.
    subset = np.round(scores[:, columns], 6)
    order = np.argsort(-subset, axis=1, kind="stable")[:, :3]
    return np.asarray(columns)[order], np.take_along_axis(subset, order, axis=1)


def export_predictions(output, images, scores, labels, *, min_score, min_margin):
    view_ids, view_scores = rank_scores(scores, labels, "view")
    part_ids, part_scores = rank_scores(scores, labels, "part")
    fields = ["image_path", "stock_number", "semantic_label", "suggested_view", "view_cosine",
              "view_margin", "view_alternatives", "suggested_part_code", "suggested_part_type",
              "part_cosine", "part_margin", "part_alternatives", "review_priority",
              "annotation_status", "reviewed_label", "review_notes"]
    counts, priorities = Counter(), Counter()
    with (output / "semantic-labels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, entry in enumerate(images):
            view, part = labels[view_ids[i, 0]], labels[part_ids[i, 0]]
            margin = round(float(view_scores[i, 0] - view_scores[i, 1]), 6)
            priority = ("low_similarity" if view_scores[i, 0] < min_score else
                        "ambiguous" if margin < min_margin else
                        "other_unclear" if view["id"] == "other_unclear" else "routine")
            accepted = view["id"] if priority == "routine" else "uncertain"
            counts[accepted] += 1
            priorities[priority] += 1
            def alternatives(ids, values):
                return json.dumps([{"id": labels[j]["id"], "name": labels[j]["name"],
                                    "cosine": float(v)} for j, v in zip(ids, values)])
            writer.writerow({"image_path": entry["relative_path"],
                "stock_number": Path(entry["relative_path"]).parts[0],
                "semantic_label": accepted, "suggested_view": view["id"],
                "view_cosine": f"{view_scores[i, 0]:.6f}", "view_margin": f"{margin:.6f}",
                "view_alternatives": alternatives(view_ids[i], view_scores[i]),
                "suggested_part_code": part["id"], "suggested_part_type": part["name"],
                "part_cosine": f"{part_scores[i, 0]:.6f}",
                "part_margin": f"{part_scores[i, 0] - part_scores[i, 1]:.6f}",
                "part_alternatives": alternatives(part_ids[i], part_scores[i]),
                "review_priority": priority, "annotation_status": "unreviewed",
                "reviewed_label": "", "review_notes": ""})
    return dict(counts), dict(priorities), view_ids, view_scores, part_ids


def render_review(output, root, images, labels, view_ids, view_scores, part_ids):
    """A deterministic sample: high-score and ambiguous examples per winning view."""
    thumbnails = output / "thumbnails"
    thumbnails.mkdir()
    sections, sample_ids, failures = [], [], []
    for label_id, label in enumerate(labels):
        if label["family"] != "view":
            continue
        members = np.flatnonzero(view_ids[:, 0] == label_id)
        if not len(members):
            continue
        strongest = sorted(members, key=lambda i: (-view_scores[i, 0], int(i)))
        ambiguous = sorted(members, key=lambda i: (view_scores[i, 0] - view_scores[i, 1], int(i)))
        # Avoid filling the preview with the same image repeated under stock folders.
        picks, seen_names = [], set()
        for pool, count in ((strongest, 3), (ambiguous, 3)):
            added = 0
            for i in pool:
                name = Path(images[i]["relative_path"]).name
                if name in seen_names:
                    continue
                picks.append(int(i)); seen_names.add(name); added += 1
                if added == count:
                    break
        cards = []
        for i in picks:
            entry = images[i]
            target = thumbnails / f"{i}.jpg"
            try:
                with Image.open(root / entry["relative_path"]) as image:
                    with ImageOps.exif_transpose(image) as oriented:
                        oriented.thumbnail((420, 300))
                        with oriented.convert("RGB") as rgb:
                            rgb.save(target, quality=85)
            except (OSError, ValueError) as error:
                failures.append({"image_path": entry["relative_path"], "error": str(error)})
                continue
            sample_ids.append(i)
            runner = labels[view_ids[i, 1]]["name"]
            part = labels[part_ids[i, 0]]["name"]
            cards.append(f'<article><a href="{html.escape((root / entry["relative_path"]).as_uri())}">'
                f'<img loading="lazy" src="thumbnails/{i}.jpg"></a>'
                f'<p>{html.escape(entry["relative_path"])}</p>'
                f'<p>View: {view_scores[i, 0]:.3f}; runner-up: {html.escape(runner)} '
                f'({view_scores[i, 1]:.3f})</p><p>Part suggestion: {html.escape(part)}</p></article>')
        sections.append(f'<h2>{html.escape(label["name"])} ({len(members):,} top matches)</h2>'
                        '<div class="grid">' + ''.join(cards) + '</div>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Semantic annotation review</title>
<style>body{font:16px system-ui;margin:30px;background:#f4f6f8;color:#17202a}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}article{background:white;padding:12px;border-radius:8px}img{width:100%;height:230px;object-fit:contain}article p{overflow-wrap:anywhere;font-size:13px}</style>
<h1>Semantic annotation review</h1><p>All predictions are unreviewed. Scores are cosine similarities, not probabilities.
Parts are suggestions only; a whole-vehicle photo can contain multiple parts. Left/right, hidden parts and exact stock identities are not verified.</p>
<p>Each view shows up to three strongest matches followed by three smallest-margin matches, with repeated filenames suppressed.
This is a diagnostic sample, not a measured accuracy result.</p><p><a href="semantic-labels.csv">Download annotations CSV</a> · <a href="summary.json">Run details</a></p>'''
    (output / "report.html").write_text(page + ''.join(sections) + '</html>', encoding="utf-8")
    return sample_ids, failures


def label_index(index, part_types, output, *, device="auto", model_cache=None,
                batch_size=2048, min_score=0.1, min_margin=0.01):
    if batch_size <= 0 or not all(math.isfinite(x) for x in (min_score, min_margin)):
        raise ValueError("Invalid batch size or thresholds")
    if not -1 <= min_score <= 1 or not 0 <= min_margin <= 2:
        raise ValueError("Cosine cutoff must be in [-1, 1] and margin in [0, 2]")
    started = time.perf_counter()
    index, output, part_types = (Path(p).expanduser().resolve() for p in (index, output, part_types))
    ensure_separate(index, output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh, empty annotation output directory")
    labels, skipped = make_taxonomy(part_types)
    manifest, images, descriptor, backend = load_index(index, search_device="cpu")
    if not isinstance(descriptor, SemanticDescriptor):
        raise ValueError("Semantic labelling requires a pinned SigLIP 2 semantic index")
    root = Path(manifest["images_root"])
    ensure_separate(root, output)
    print("Checking source inventory against indexed images...", file=sys.stderr)
    current = ImageDataset(root)
    indexed_paths = {entry["relative_path"] for entry in images}
    # Corrupt files omitted by indexing are permitted; new or removed files are not.
    invalid = [json.loads(line)["relative_path"] for line in (index / "errors.jsonl").read_text().splitlines()]
    if {p.relative_to(root).as_posix() for p in current.paths} != indexed_paths | set(invalid):
        raise ValueError("Image inventory changed since indexing; rebuild the index")
    for entry in images:
        stat = (root / entry["relative_path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry["size_bytes"], entry["mtime_ns"]):
            raise ValueError(f"Image changed since indexing: {entry['relative_path']}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"status": "running"})
    write_json(output / "taxonomy.json", labels)
    prompt_vectors, runtime = encode_prompts(labels, device=device, cache_dir=model_cache)
    np.save(output / "prompt-embeddings.npy", prompt_vectors, allow_pickle=False)
    scores = np.lib.format.open_memmap(output / "label-scores.npy", mode="w+", dtype="float32",
                                      shape=(len(images), len(labels)))
    for start in range(0, len(images), batch_size):
        count = min(batch_size, len(images) - start)
        vectors = backend.index.reconstruct_n(start, count)
        validate_vectors(vectors, count, descriptor.dimension)
        scores[start:start + count] = score_labels(vectors, prompt_vectors, labels)
        if start % (batch_size * 20) == 0:
            print(f"Scored {start + count:,}/{len(images):,} images", file=sys.stderr)
    scores.flush()
    counts, priorities, view_ids, view_scores, part_ids = export_predictions(
        output, images, scores, labels, min_score=min_score, min_margin=min_margin)
    sample_ids, failures = render_review(output, root, images, labels, view_ids, view_scores, part_ids)
    # Preserve the exact image-to-score row mapping without copying original pixels.
    with (output / "image-paths.jsonl").open("w", encoding="utf-8") as stream:
        for entry in images:
            stream.write(json.dumps({"id": entry["id"], "image_path": entry["relative_path"]}) + "\n")
    summary = {"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "siglip2-zero-shot-mean-prompt-cosine-v1", "index": str(index),
        "index_manifest_sha256": file_hash(index / "manifest.json"), "index_files": manifest["files"],
        "image_root": str(root), "image_count": len(images), "invalid_index_images": len(invalid),
        "part_types_source": str(part_types), "part_types_sha256": file_hash(part_types),
        "source_rows_without_named_part_type": skipped,
        "model": descriptor.metadata(), "text_encoding": runtime, "scoring_device": "cpu",
        "view_label_count": sum(label["family"] == "view" for label in labels),
        "part_label_count": sum(label["family"] == "part" for label in labels),
        "min_score": min_score, "min_margin": min_margin, "label_counts": counts,
        "review_priority_counts": priorities, "review_sample_ids": sample_ids,
        "thumbnail_errors": failures, "elapsed_seconds": round(time.perf_counter() - started, 3),
        "limitations": ["All annotations are unreviewed zero-shot predictions, not ground truth.",
            "Cosines and margins are not calibrated probabilities; cutoffs are provisional.",
            "Part suggestions are ranked separately and are never confirmed part labels.",
            "Whole-vehicle images may show multiple parts; hidden parts, side and exact part identity are not verified.",
            "No independent labelled accuracy evaluation has been performed.",
            "Source inventory checked by paths, sizes and mtimes, not fresh pixel checksums."],
        "files": {name: {"sha256": file_hash(output / name)} for name in
                  ("taxonomy.json", "prompt-embeddings.npy", "label-scores.npy", "image-paths.jsonl", "semantic-labels.csv")}}
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", {"status": "complete", "image_count": len(images)})
    return summary
