"""Required photo categories from vehicle_angles.md, mined from an unlabeled index.

The collection has no labels. Text prompts from the specification are scored
against the stored image vectors (SigLIP 2 or FG-CLIP 2); the strongest matches per
category become an image-space reference set. Video frames are then compared with
those references (k nearest neighbours), which avoids the large text/image cosine gap.
Each category's `framing`/`avoid` text then splits its references into well- and
poorly-framed quartiles; frames are scored by which group they resemble more.
Assignments are unreviewed pseudo-labels, not detections or ground truth.
"""

from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image, ImageOps

from .download import write_json
from .indexing import ensure_separate, file_hash, load_index
from .labelling import validate_vectors
from .semantic import SemanticDescriptor

REQUIREMENTS_VERSION = "angle-requirements-knn-v2"
LIST_KEYS = ("views", "sides", "parts", "aliases")
SCALAR_KEYS = ("id", "description", "capture", "optional", "requirement", "framing", "avoid")
OTHER_ID = "_other"
# Competes in assignment so frames of the ground, people or paperwork are not
# forced into the closest vehicle category. Never a requirement.
OTHER_PROMPTS = ["a photo of the ground or pavement.", "a photo of a person.", "a photo of the sky.",
                 "a photo of a building or workshop.", "a blurry photograph.",
                 "a photo of a printed document.", "a photo of a hand."]


def parse_spec(path):
    """Parse the angle specification into categories, tolerating its list-indentation slips.

    Any `- key: value` line with a known key starts a field, whatever its indentation;
    other `- value` lines extend the latest list field. Slips are reported as warnings.
    """
    text = Path(path).read_text(encoding="utf-8")
    categories, warnings = [], []
    section = category = current_list = None
    key_indent = None
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        if line.startswith("## "):
            category = {"name": stripped[3:].strip(), "section": section, "line": number}
            categories.append(category)
            current_list = key_indent = None
            continue
        if line.startswith("# "):
            section, category, current_list = stripped[2:].strip(), None, None
            continue
        if not stripped.startswith("- "):
            raise ValueError(f"{path}:{number}: expected a '- ' list line")
        if category is None:
            raise ValueError(f"{path}:{number}: field outside a '## ' category")
        indent = len(line) - len(line.lstrip())
        item = stripped[2:].strip()
        match = re.fullmatch(r"([a-z_]+):\s*(.*)", item)
        if match and match.group(1) in LIST_KEYS + SCALAR_KEYS:
            key, value = match.groups()
            if key_indent is None:
                key_indent = indent
            elif indent != key_indent:
                warnings.append(f"line {number}: '{key}' indented as a list item; read as a {category['name']} field")
            if key in category:
                raise ValueError(f"{path}:{number}: duplicate '{key}' in {category['name']}")
            if key in LIST_KEYS:
                if value:
                    raise ValueError(f"{path}:{number}: '{key}' must be followed by list items")
                category[key], current_list = [], key
            else:
                category[key], current_list = value.strip("`").strip(), None
            continue
        if match:
            raise ValueError(f"{path}:{number}: unknown field '{match.group(1)}'")
        if current_list is None:
            raise ValueError(f"{path}:{number}: list item without a list field")
        if key_indent is not None and indent <= key_indent:
            warnings.append(f"line {number}: '{item}' not indented; read as {category['name']} {current_list}")
        category[current_list].append(item)
    seen = set()
    for category in categories:
        identifier = category.get("id")
        if not identifier or not re.fullmatch(r"[a-z][a-z0-9_]*", identifier):
            raise ValueError(f"Category {category['name']!r} needs a lowercase snake_case id")
        if identifier in seen:
            raise ValueError(f"Duplicate category id {identifier}")
        seen.add(identifier)
        optional = category.get("optional", "false").lower()
        if optional not in ("true", "false"):
            raise ValueError(f"{identifier}: optional must be true or false")
        category["optional"] = optional == "true"
        for key in LIST_KEYS:
            category.setdefault(key, [])
        category["slots"] = required_slots(category)
        category["prompts"], category["variant_prompts"] = category_prompts(category)
        category["framing_prompts"] = framing_prompts(category)
    if len(categories) < 2:
        raise ValueError("The specification needs at least two categories")
    return categories, warnings


def required_slots(category):
    """Distinct photos wanted per category; None means every distinct view found.

    Left/right entries count as separate photos, but are not verified as sides.
    """
    if category.get("requirement", "").lower().startswith("capture all"):
        return None
    return max(1, len(category["sides"])) * max(1, len(category["views"]) + len(category["parts"]))


def category_prompts(category):
    readable = lambda value: value.replace("-", " ").replace("_", " ").lower()
    names = [category["name"].lower(), *(readable(a) for a in category["aliases"])]
    prompts = [f"a photo of a vehicle's {name}." for name in dict.fromkeys(names)]
    if category.get("description"):
        prompts.append(category["description"].rstrip(".").lower() + ".")
    # Viewpoint and part entries diversify the mined references; they are not scored as targets.
    variants = [f"a photo of the {names[0]} seen from the {readable(view)}." for view in category["views"]]
    variants += [f"a photo of a vehicle's {readable(part)}." for part in category["parts"]]
    return prompts, variants


def framing_prompts(category):
    """Composition contrast for ranking within a category; defaults ask for the whole part in frame."""
    name = category["name"].lower()
    positive = category.get("framing") or f"A photo of the entire {name} fully in frame, centred and in sharp focus."
    negative = category.get("avoid") or f"A blurry or cropped photo showing only part of the {name}."
    return {"framing": [positive.rstrip(".").lower() + "."], "avoid": [negative.rstrip(".").lower() + "."],
            "source": "spec" if category.get("framing") else "default"}


def split_framing(reference_vectors, framing_vectors, avoid_vectors, *, fraction=0.25):
    """Indices of the best- and worst-framed references by text contrast (top/bottom fraction).

    Ties use six-decimal scores then reference order. Fewer than four references give
    empty groups, which makes the framing score neutral for that category.
    """
    count = len(reference_vectors)
    size = int(count * fraction)
    if size < 1:
        return [], []
    contrast = np.round((reference_vectors @ framing_vectors.T).mean(axis=1)
                        - (reference_vectors @ avoid_vectors.T).mean(axis=1), 6)
    order = np.lexsort((np.arange(count), -contrast))
    return sorted(order[:size].tolist()), sorted(order[-size:].tolist())


def framing_scores(frames, references, records, k=5):
    """Per category: mean top-k cosine to well-framed minus poorly-framed references (0 if unsplit)."""
    scores = np.zeros((len(frames), len(records)))
    for c, record in enumerate(records):
        good, poor = record.get("framing_good", []), record.get("framing_poor", [])
        if not good or not poor:
            continue
        offset = record["reference_offset"]
        spans = [{"rows": [offset + i for i in group]} for group in (good, poor)]
        both = knn_scores(frames, references, spans, k=k)
        scores[:, c] = both[:, 0] - both[:, 1]
    return scores


def category_spans(records):
    return [{"rows": list(range(r["reference_offset"], r["reference_offset"] + r["reference_count"]))}
            for r in records]


def load_requirement_index(path):
    """Verified v2 requirement artifacts for scoring frames (categories, records, reference vectors)."""
    path = Path(path).expanduser().resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    status = json.loads((path / "status.json").read_text())
    if status.get("status") != "complete" or manifest.get("version") != REQUIREMENTS_VERSION:
        raise ValueError(f"Requirement index must be a complete {REQUIREMENTS_VERSION} build; rerun index-requirements")
    for name, info in manifest["files"].items():
        if file_hash(path / name) != info["sha256"]:
            raise ValueError(f"Requirement artifact checksum mismatch: {name}")
    categories = json.loads((path / "categories.json").read_text())
    records = json.loads((path / "mined.json").read_text())
    if [r["id"] for r in records] != [c["id"] for c in categories] + [OTHER_ID]:
        raise ValueError("Requirement categories and mined references are out of order")
    references = np.load(path / "reference-vectors.npy", allow_pickle=False)
    return {"path": str(path), "manifest": manifest, "manifest_sha256": file_hash(path / "manifest.json"),
            "categories": categories, "records": records, "references": references}


def label_table(categories):
    """Assignment labels: each category's base prompts, plus the non-requirement background."""
    labels = [{"id": c["id"], "prompts": c["prompts"]} for c in categories]
    return labels + [{"id": OTHER_ID, "prompts": OTHER_PROMPTS}]


def assignment_scores(prompt_scores, owner, base_count, variant_end, label_count):
    """Per category: the stronger of its mean base-prompt cosine and its best view/part variant.

    A single variant (e.g. "the entire vehicle seen from the rear") can claim a photo that
    the averaged base prompts would leave to a look-alike category (rear views as towbars),
    while averaging keeps long alias lists from inflating a category.
    """
    columns = []
    for c in range(label_count):
        base = prompt_scores[:, [p for p in range(base_count) if owner[p] == c]].mean(axis=1)
        variants = [p for p in range(base_count, variant_end) if owner[p] == c]
        columns.append(np.maximum(base, prompt_scores[:, variants].max(axis=1)) if variants else base)
    return np.stack(columns, axis=1)


def mine_references(vectors, label_scores, prompt_scores, labels, prompt_owner, *,
                    per_category=1000, duplicate_similarity=0.9995):
    """Per category, round-robin over its prompts' best matches among images it wins.

    Images are only mined for the category whose mean prompt cosine is highest.
    Near-identical vectors (the same photo in several stock folders) count once.
    """
    winners = np.argmax(np.round(label_scores, 6), axis=1)
    mined = []
    for c, label in enumerate(labels):
        rows = np.flatnonzero(winners == c)
        columns = [p for p, owner in enumerate(prompt_owner) if owner == c]
        orders = [rows[np.lexsort((rows, -np.round(prompt_scores[rows, p], 6)))] for p in columns]
        kept, seen, cursor = [], set(), [0] * len(orders)
        kept_vectors = np.empty((per_category, vectors.shape[1]), dtype=np.float32)
        while len(kept) < per_category and any(cursor[i] < len(o) for i, o in enumerate(orders)):
            for i, order in enumerate(orders):
                while cursor[i] < len(order):
                    row = int(order[cursor[i]])
                    cursor[i] += 1
                    if row in seen:
                        continue
                    seen.add(row)
                    if kept and float(np.max(kept_vectors[:len(kept)] @ vectors[row])) >= duplicate_similarity:
                        continue
                    kept_vectors[len(kept)] = vectors[row]
                    kept.append(row)
                    break
                if len(kept) == per_category:
                    break
        mined.append({"id": label["id"], "rows": kept, "assigned_images": int(len(rows))})
    return mined


def knn_scores(frames, references, mined, k=5):
    """Mean of each category's k highest reference cosines (fewer if fewer were mined)."""
    scores = np.full((len(frames), len(mined)), -1.0)
    for c, item in enumerate(mined):
        if not item["rows"]:
            continue
        similarities = frames @ references[item["rows"]].T
        top = -np.sort(-similarities, axis=1)[:, :k]
        scores[:, c] = top.mean(axis=1)
    return scores


def build_requirement_index(index, spec, output, *, device="auto", model_cache=None,
                            per_category=1000, gallery_size=12):
    """Mine per-category reference sets from a semantic index; no image re-encoding."""
    if per_category <= 0 or gallery_size <= 0:
        raise ValueError("per_category and gallery_size must be positive")
    started = time.perf_counter()
    index, spec, output = (Path(p).expanduser().resolve() for p in (index, spec, output))
    ensure_separate(index, output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh, empty requirement index directory")
    categories, warnings = parse_spec(spec)
    manifest, images, descriptor, backend = load_index(index, search_device="cpu", extraction_device=device,
                                                       model_cache=model_cache)
    if not isinstance(descriptor, SemanticDescriptor):
        raise ValueError("Requirement mining needs a semantic (SigLIP 2 or FG-CLIP 2) index")
    root = Path(manifest["images_root"])
    ensure_separate(root, output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"status": "running"})
    labels = label_table(categories)
    prompts, owner = [], []
    for c, label in enumerate(labels):
        prompts += label["prompts"]
        owner += [c] * len(label["prompts"])
    base_count = len(prompts)
    for c, category in enumerate(categories):
        prompts += category["variant_prompts"]
        owner += [c] * len(category["variant_prompts"])
    variant_end = len(prompts)
    roles = ["assignment"] * base_count + ["mining_variant"] * (variant_end - base_count)
    for c, category in enumerate(categories):
        for role in ("framing", "avoid"):
            prompts += category["framing_prompts"][role]
            owner += [c] * len(category["framing_prompts"][role])
            roles += [role] * len(category["framing_prompts"][role])
    prompt_vectors, runtime = descriptor.encode_text(prompts)
    print(f"Scoring {len(images):,} stored image vectors...", file=sys.stderr)
    vectors = backend.index.reconstruct_n(0, len(images))
    validate_vectors(vectors, len(images), descriptor.dimension)
    prompt_scores = vectors @ prompt_vectors.T
    label_scores = assignment_scores(prompt_scores, owner, base_count, variant_end, len(labels))
    mined = mine_references(vectors, label_scores, prompt_scores[:, :variant_end], labels, owner[:variant_end],
                            per_category=per_category)
    rows = [row for item in mined for row in item["rows"]]
    np.save(output / "reference-vectors.npy", vectors[rows], allow_pickle=False)
    np.save(output / "prompt-embeddings.npy", prompt_vectors, allow_pickle=False)
    np.save(output / "label-scores.npy", label_scores.astype(np.float32), allow_pickle=False)
    offset, records = 0, []
    for c, item in enumerate(mined):
        good = poor = []
        if c < len(categories):
            role_vectors = {role: prompt_vectors[[p for p in range(len(prompts)) if owner[p] == c and roles[p] == role]]
                            for role in ("framing", "avoid")}
            good, poor = split_framing(vectors[item["rows"]], role_vectors["framing"], role_vectors["avoid"])
        records.append({"id": item["id"], "assigned_images": item["assigned_images"],
                        "reference_offset": offset, "reference_count": len(item["rows"]),
                        "framing_good": good, "framing_poor": poor,
                        "references": [{"image_id": images[r]["id"], "relative_path": images[r]["relative_path"],
                                        "label_score": round(float(label_scores[r, c]), 6)} for r in item["rows"]]})
        offset += len(item["rows"])
    write_json(output / "mined.json", records)
    write_json(output / "categories.json", categories)
    with (output / "prompts.jsonl").open("w", encoding="utf-8") as stream:
        for p, text in enumerate(prompts):
            stream.write(json.dumps({"row": p, "label": labels[owner[p]]["id"], "prompt": text,
                                     "role": roles[p]}) + "\n")
    render_gallery(output, root, categories, records, warnings, gallery_size)
    summary = {"status": "complete", "version": REQUIREMENTS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(), "spec": str(spec), "spec_sha256": file_hash(spec),
        "index": str(index), "index_manifest_sha256": file_hash(index / "manifest.json"),
        "index_sha256": manifest["files"]["index.faiss"]["sha256"], "image_root": str(root),
        "image_count": len(images), "model": descriptor.metadata(), "text_encoding": runtime,
        "per_category": per_category, "reference_duplicate_similarity": 0.9995,
        "category_count": len(categories), "parse_warnings": warnings,
        "background_label": OTHER_ID, "background_prompts": OTHER_PROMPTS,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "method": ("The larger of mean base-prompt cosine and best view/part variant cosine assigns each stored "
                   "image to one category or background; "
                   "each category keeps a round-robin of its prompts' strongest assigned images. "
                   "Framing minus avoid prompt cosine splits each category's references into top and bottom quartiles."),
        "framing_fraction": 0.25,
        "limitations": ["Mined references are unreviewed zero-shot pseudo-labels, not ground truth.",
            "Cosines are uncalibrated; fine engine-bay parts and similar controls are often confused.",
            "Left/right entries set photo counts only; sides are not recognised or verified.",
            "Optional categories cannot be confirmed absent from weak matches.",
            "Framing groups are learned from text contrast; they do not measure object boundaries or text legibility."],
        "files": {name: {"sha256": file_hash(output / name)} for name in
                  ("categories.json", "mined.json", "prompts.jsonl", "prompt-embeddings.npy",
                   "label-scores.npy", "reference-vectors.npy")}}
    write_json(output / "manifest.json", summary)
    write_json(output / "status.json", {"status": "complete", "category_count": len(categories)})
    return {key: summary[key] for key in ("status", "category_count", "parse_warnings", "elapsed_seconds")}


def render_gallery(output, root, categories, records, warnings, gallery_size):
    """Audit page: what each category learned from the unlabeled collection."""
    (output / "thumbnails").mkdir()
    spec = {c["id"]: c for c in categories}
    sections = []
    for record in records:
        category = spec.get(record["id"])
        cards, names = [], set()
        for ref in record["references"]:
            name = Path(ref["relative_path"]).name
            if name in names:
                continue
            names.add(name)
            target = f"thumbnails/{ref['image_id']}.jpg"
            try:
                with Image.open(root / ref["relative_path"]) as image, ImageOps.exif_transpose(image) as oriented:
                    oriented.thumbnail((300, 220))
                    with oriented.convert("RGB") as rgb:
                        rgb.save(output / target, quality=82)
            except (OSError, ValueError):
                continue
            cards.append(f'<figure><img loading="lazy" src="{target}"><figcaption>'
                         f'{html.escape(ref["relative_path"])} · {ref["label_score"]:.3f}</figcaption></figure>')
            if len(cards) == gallery_size:
                break
        if category:
            slots = "all distinct" if category["slots"] is None else category["slots"]
            heading = (f'{html.escape(category["name"])} <code>{record["id"]}</code>'
                       f'{" · optional" if category["optional"] else ""}')
            detail = (f'{html.escape(category["section"] or "")} · photos wanted: {slots} · '
                      f'{html.escape(category.get("description", ""))}')
        else:
            heading, detail = f'Background <code>{record["id"]}</code>', "Not a requirement; absorbs unrelated frames."
        framing = ""
        if category and record.get("framing_good"):
            groups = []
            for label, key in (("Best framed", "framing_good"), ("Worst framed", "framing_poor")):
                figures = []
                for i in record[key][:6] if key == "framing_good" else record[key][-6:]:
                    ref = record["references"][i]
                    target = f"thumbnails/{ref['image_id']}.jpg"
                    if not (output / target).exists():
                        try:
                            with Image.open(root / ref["relative_path"]) as image, ImageOps.exif_transpose(image) as oriented:
                                oriented.thumbnail((300, 220))
                                with oriented.convert("RGB") as rgb:
                                    rgb.save(output / target, quality=82)
                        except (OSError, ValueError):
                            continue
                    figures.append(f'<figure><img loading="lazy" src="{target}"></figure>')
                groups.append(f'<h3>{label} references</h3><div class="g">{"".join(figures)}</div>')
            rules = category["framing_prompts"]
            framing = (f'<p class="m">Framing ({rules["source"]}): {html.escape(rules["framing"][0])} '
                       f'Avoid: {html.escape(rules["avoid"][0])}</p>' + "".join(groups))
        sections.append(f'<section><h2>{heading}</h2><p>{detail}</p><p class="m">{record["assigned_images"]:,} '
                        f'collection images assigned · {record["reference_count"]:,} kept as references</p>'
                        f'<div class="g">{"".join(cards)}</div>{framing}</section>')
    notes = "".join(f"<li>{html.escape(w)}</li>" for w in warnings) or "<li>None</li>"
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Required category references</title><style>body{{font:15px system-ui;margin:24px;background:#f4f6f8;color:#17202a}}
section{{background:#fff;border-radius:8px;padding:12px 16px;margin:16px 0}}.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}}
figure{{margin:0}}img{{width:100%;height:150px;object-fit:contain;background:#eee}}figcaption{{font-size:11px;overflow-wrap:anywhere}}.m{{color:#555}}</style>
<h1>Required category references</h1><p>Each category's reference photos were chosen by zero-shot text matching against the
unlabeled collection. They are unreviewed pseudo-labels: check that each gallery shows the intended part before trusting video assignments.
Captions show the image path and mean prompt cosine (uncalibrated, not a probability).</p>
<p><a href="manifest.json">Provenance</a> · <a href="mined.json">All references</a></p><h2>Specification parse warnings</h2><ul>{notes}</ul>
{"".join(sections)}</html>'''
    (output / "report.html").write_text(page, encoding="utf-8")
