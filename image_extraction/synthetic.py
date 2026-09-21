"""Deterministic geometric scenes with explicit composition ground truth.

These are code-rendered fixtures, not photorealistic training images. Labels
come from geometry and are independent of any image descriptor or its scores.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

from .download import write_json

GENERATOR_VERSION = "composition-scenes-v1"
SUBJECTS = ("car", "tree", "person")
APPEARANCES = ("neutral", "warm", "cool", "textured")
VARIANTS = ("base", "position", "scale", "background")


@dataclass(frozen=True)
class Layout:
    center_x: float
    center_y: float
    width: float
    height: float
    horizon: float


def scene_seed(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")


def family_layouts(seed: int) -> dict[str, Layout]:
    rng = random.Random(seed)
    x = rng.uniform(0.26, 0.34)
    if rng.choice((True, False)):
        x = 1 - x
    base = Layout(x, rng.uniform(0.55, 0.67), rng.uniform(0.18, 0.26),
                  rng.uniform(0.22, 0.32), rng.uniform(0.25, 0.45))
    return {
        "base": base,
        "position": replace(base, center_x=1 - base.center_x),
        "scale": replace(base, width=base.width * 1.5, height=base.height * 1.5),
        "background": replace(base, horizon=base.horizon + 0.35),
    }


def render_scene(
    layout: Layout, subject: str, appearance: str, *, size: tuple[int, int], seed: int
) -> Image.Image:
    """Render one RGB fixture; each subject fits the same normalized bounding box."""
    if subject not in SUBJECTS or appearance not in APPEARANCES:
        raise ValueError("Unknown subject or appearance")
    palettes = {
        "neutral": ((220, 225, 230), (145, 150, 155), (40, 45, 50)),
        "warm": ((240, 215, 185), (180, 145, 110), (85, 45, 25)),
        "cool": ((185, 220, 240), (110, 150, 180), (25, 50, 85)),
        "textured": ((225, 220, 210), (155, 150, 135), (55, 50, 35)),
    }
    sky, ground, foreground = palettes[appearance]
    image = Image.new("RGB", size, sky)
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.rectangle((0, round(layout.horizon * h), w - 1, h - 1), fill=ground)
    x0 = (layout.center_x - layout.width / 2) * (w - 1)
    y0 = (layout.center_y - layout.height / 2) * (h - 1)
    bw, bh = layout.width * (w - 1), layout.height * (h - 1)

    def point(x, y):
        return round(x0 + x * bw), round(y0 + y * bh)

    def box(left, top, right, bottom):
        return (*point(left, top), *point(right, bottom))

    if subject == "car":
        draw.polygon([point(0.15, 0.5), point(0.30, 0), point(0.72, 0), point(0.88, 0.5)], fill=foreground)
        draw.rectangle(box(0, 0.45, 1, 0.85), fill=foreground)
        draw.ellipse(box(0.08, 0.65, 0.33, 1), fill=foreground)
        draw.ellipse(box(0.67, 0.65, 0.92, 1), fill=foreground)
    elif subject == "tree":
        draw.rectangle(box(0.40, 0.55, 0.60, 1), fill=foreground)
        draw.polygon([point(0.5, 0), point(0, 0.75), point(1, 0.75)], fill=foreground)
    else:
        draw.ellipse(box(0.32, 0, 0.68, 0.30), fill=foreground)
        draw.polygon([point(0.30, 0.30), point(0.70, 0.30), point(1, 0.60),
                      point(0.70, 0.65), point(0.70, 1), point(0.53, 1),
                      point(0.50, 0.70), point(0.47, 1), point(0.30, 1),
                      point(0.30, 0.65), point(0, 0.60)], fill=foreground)
    if appearance == "textured":
        pixels = np.asarray(image, dtype=np.int16)
        noise = np.random.Generator(np.random.PCG64(seed)).integers(-8, 9, (h, w, 1), dtype=np.int16)
        textured = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))
        image.close()
        image = textured
    return image


def generate_synthetic(
    output: str | Path = "data/synthetic", *, seed: int = 20260921,
    families_per_split: int = 12, size: tuple[int, int] = (384, 256),
) -> dict:
    """Write PNGs, metadata, explicit preference pairs, and a visual overview.

    A family and all its altered layouts belong to exactly one split. Rerunning
    an identical configuration regenerates its files; different configurations
    require a different directory. The manifest is marked complete last.
    """
    if families_per_split < 1 or min(size) < 32:
        raise ValueError("Need at least one family per split and image dimensions >= 32")
    output = Path(output)
    config = {
        "generator_version": GENERATOR_VERSION, "seed": seed,
        "families_per_split": families_per_split, "size": list(size),
        "subjects": list(SUBJECTS), "appearances": list(APPEARANCES),
        "pillow_version": pillow_version, "numpy_version": np.__version__,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["config"] != config:
            raise ValueError("Synthetic configuration differs; use a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, {"status": "generating", "config": config})
    records = []
    pairs = []
    for split in ("dev", "test"):
        for family_index in range(families_per_split):
            family = f"{split}-{family_index:03d}"
            layouts = family_layouts(scene_seed(seed, family))
            for variant, layout in layouts.items():
                layout_id = f"{family}-{variant}"
                for subject_index, subject in enumerate(SUBJECTS):
                    for appearance_index, appearance in enumerate(APPEARANCES):
                        identifier = f"{layout_id}-{subject}-{appearance}"
                        relative_path = f"images/{split}/{identifier}.png"
                        path = output / relative_path
                        path.parent.mkdir(parents=True, exist_ok=True)
                        # Hold texture noise fixed across geometry changes for
                        # the same subject/appearance within a family.
                        render_seed = scene_seed(seed, f"{family}-{subject}-{appearance}")
                        with render_scene(layout, subject, appearance, size=size, seed=render_seed) as image:
                            image.save(path)
                        records.append({
                            "id": identifier, "path": relative_path, "split": split,
                            "family_id": family, "layout_id": layout_id, "variant": variant,
                            "subject": subject, "appearance": appearance,
                            "layout": asdict(layout), "seed": render_seed,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        })
                        # Same layout/different subject versus the SAME subject
                        # and appearance under a one-factor geometry change.
                        if variant == "base":
                            positive_subject = SUBJECTS[(subject_index + 1) % len(SUBJECTS)]
                            for change in ("subject", "subject_and_appearance"):
                                positive_appearance = appearance if change == "subject" else APPEARANCES[(appearance_index + 1) % len(APPEARANCES)]
                                positive = f"{layout_id}-{positive_subject}-{positive_appearance}"
                                for negative_variant in VARIANTS[1:]:
                                    pairs.append({
                                        "split": split, "query": identifier, "positive": positive,
                                        "negative": f"{family}-{negative_variant}-{subject}-{appearance}",
                                        "change": negative_variant, "positive_change": change,
                                    })
    for name, values in (("images.jsonl", records), ("pairs.jsonl", pairs)):
        with (output / name).open("w", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, sort_keys=True) + "\n")
    summary = {
        "status": "complete", "config": config, "image_count": len(records),
        "layout_count": families_per_split * 2 * len(VARIANTS),
        "pair_count": len(pairs), "split_policy": "whole families; no family or layout shared across dev/test",
        "relevance": "same layout_id; exclude the query image itself",
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                  for name in ("images.jsonl", "pairs.jsonl")},
    }
    _write_preview(output, records)
    write_json(manifest_path, summary)
    return summary


def _write_preview(root: Path, records: list[dict]) -> None:
    """An inspectable HTML overview of one family in each split."""
    rows = []
    for split in ("dev", "test"):
        for variant in VARIANTS:
            samples = [r for r in records if r["family_id"] == f"{split}-000" and r["variant"] == variant]
            figures = "".join(
                f'<figure><img src="{r["path"]}" alt="{r["id"]}"><figcaption>{r["subject"]} / {r["appearance"]}</figcaption></figure>'
                for r in samples
            )
            rows.append(f"<h2>{split} / {variant}</h2><div class='grid'>{figures}</div>")
    (root / "preview.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Synthetic composition fixtures</title>"
        "<style>body{font:16px system-ui;margin:2rem;background:#f4f5f7;color:#18202b}"
        ".grid{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:12px}"
        "figure{margin:0;background:white;padding:8px}img{width:100%}figcaption{padding:6px}</style>"
        "<h1>Synthetic composition fixtures</h1><p>One family from each split. Within a row, geometry is fixed. "
        "Following rows change only position, scale, or background relative to that family's base layout.</p>"
        + "".join(rows) + "</html>", encoding="utf-8",
    )
