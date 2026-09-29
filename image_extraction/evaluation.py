"""Small, exact synthetic benchmark; not a production retrieval index."""

from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import sys
import time
from urllib.parse import quote

import numpy as np
from PIL import __version__ as pillow_version

from .dataset import ImageDataset
from .descriptors import SpatialDescriptor
from .download import write_json
from .synthetic import GENERATOR_VERSION

TIE_TOLERANCE = 1e-6


def _load_labels(root: Path) -> tuple[dict, list[dict], list[dict]]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest["config"]["generator_version"] != GENERATOR_VERSION:
        raise ValueError("Synthetic dataset is incomplete or has an unsupported generator version")
    for name in ("images.jsonl", "pairs.jsonl"):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError(f"Label checksum mismatch: {name}; regenerate the dataset")
    records = [json.loads(line) for line in (root / "images.jsonl").read_text().splitlines()]
    pairs = [json.loads(line) for line in (root / "pairs.jsonl").read_text().splitlines()]
    if not records or len(records) != manifest["image_count"] or len(pairs) != manifest["pair_count"]:
        raise ValueError("Dataset counts do not match the manifest")
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records) or len({r["path"] for r in records}) != len(records):
        raise ValueError("Duplicate image IDs or paths")
    families, layouts = {}, {}
    for record in records:
        split = record["split"]
        if split not in ("dev", "test"):
            raise ValueError("Unknown dataset split")
        path = (root / record["path"]).resolve()
        if not path.is_relative_to((root / "images" / split).resolve()):
            raise ValueError("Image path is outside its split directory")
        family = record["family_id"]
        if family in families and families[family] != split:
            raise ValueError("Family leakage between development and test splits")
        families[family] = split
        signature = (split, json.dumps(record["layout"], sort_keys=True))
        previous = layouts.setdefault(record["layout_id"], signature)
        if previous != signature:
            raise ValueError("A layout ID has inconsistent geometry or split")
    dev_geometry = {signature[1] for signature in layouts.values() if signature[0] == "dev"}
    test_geometry = {signature[1] for signature in layouts.values() if signature[0] == "test"}
    if dev_geometry & test_geometry:
        raise ValueError("Layout geometry leakage between development and test splits")
    for pair in pairs:
        if any(pair[key] not in by_id for key in ("query", "positive", "negative")):
            raise ValueError("Preference pair references an unknown image")
        query, positive, negative = (by_id[pair[key]] for key in ("query", "positive", "negative"))
        if any(r["split"] != pair["split"] for r in (query, positive, negative)):
            raise ValueError("Preference pair crosses splits")
        if query["layout_id"] != positive["layout_id"] or query["subject"] == positive["subject"]:
            raise ValueError("Invalid same-layout/different-subject positive")
        if (query["layout_id"] == negative["layout_id"] or query["subject"] != negative["subject"]
                or query["appearance"] != negative["appearance"]):
            raise ValueError("Invalid different-layout/same-subject negative")
    return manifest, records, pairs


def preference_metrics(margins: list[float]) -> dict:
    values = np.asarray(margins, dtype=np.float64)
    if not len(values):
        raise ValueError("No labeled preference pairs")
    wins = int(np.count_nonzero(values > TIE_TOLERANCE))
    ties = int(np.count_nonzero(np.abs(values) <= TIE_TOLERANCE))
    return {
        "pairs": len(values), "wins": wins, "ties": ties,
        "losses": len(values) - wins - ties,
        "accuracy": wins / len(values), "tie_rate": ties / len(values),
    }


def rank_candidates(scores: np.ndarray, query_index: int, k: int = 10) -> np.ndarray:
    """Exclude self explicitly, then break exact ties by stable manifest order."""
    candidates = np.arange(len(scores))
    candidates = candidates[candidates != query_index]
    return candidates[np.argsort(-scores[candidates], kind="stable")[:k]]


def evaluate_synthetic(
    dataset: str | Path = "data/synthetic", output: str | Path = "outputs/synthetic-baseline",
    *, split: str = "dev", descriptor: SpatialDescriptor | None = None,
) -> dict:
    if split not in ("dev", "test", "all"):
        raise ValueError("split must be dev, test, or all")
    root, output = Path(dataset).resolve(), Path(output).resolve()
    # Preserve the published synthetic baseline independently of indexing defaults.
    descriptor = descriptor or SpatialDescriptor(max_side=256)
    manifest, records, pairs = _load_labels(root)
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "descriptor": descriptor.metadata(), "dataset_config": manifest["config"],
        "dataset_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pillow": pillow_version,
                        "platform": platform.platform(), "processor": platform.processor()},
        "protocol": {
            "relevance": "same layout_id, exclude self, candidates only from the same split",
            "preference_accuracy": "wins / all pairs; ties are not wins",
            "tie_tolerance": TIE_TOLERANCE, "ranking_ties": "stable image ID order",
            "timing": "one CPU run; per-query scoring includes dot products and sorting; not a scale benchmark",
        },
        "splits": {},
    }
    for name in (("dev", "test") if split == "all" else (split,)):
        selected = sorted((r for r in records if r["split"] == name), key=lambda r: r["id"])
        selected_pairs = [p for p in pairs if p["split"] == name]
        if len(selected) < 11 or not selected_pairs:
            raise ValueError("Each evaluated split needs at least 11 images and labeled pairs")
        index = {r["id"]: i for i, r in enumerate(selected)}
        # Verify image bytes as well as labels before reporting frozen-benchmark results.
        for record in selected:
            if hashlib.sha256((root / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Image checksum mismatch: {record['path']}")
        local = ImageDataset(root / "images" / name)
        path_indices = {path: i for i, path in enumerate(local.paths)}
        if set(path_indices) != {(root / r["path"]).resolve() for r in selected}:
            raise ValueError("Discovered images differ from the benchmark manifest")

        vectors = np.empty((len(selected), descriptor.dimension), dtype=np.float32)
        started = time.perf_counter()
        for i, record in enumerate(selected):
            sample = local[path_indices[(root / record["path"]).resolve()]]
            try:
                vectors[i] = descriptor.extract(sample.image)
            finally:
                sample.close()
        extraction_seconds = time.perf_counter() - started
        np.save(output / f"features-{name}.npy", vectors, allow_pickle=False)
        write_json(output / f"features-{name}.json", {
            "descriptor": descriptor.metadata(), "dataset_manifest_sha256": results["dataset_manifest_sha256"],
            "image_ids": [r["id"] for r in selected],
        })

        by_query = defaultdict(list)
        for pair in selected_pairs:
            by_query[pair["query"]].append(pair)
        precision, top1, latencies, outcomes = [], [], [], []
        chance = []
        with (output / f"rankings-{name}.jsonl").open("w", encoding="utf-8") as stream:
            for i, query in enumerate(selected):
                started = time.perf_counter()
                scores = vectors @ vectors[i]
                ranked = rank_candidates(scores, i)
                latencies.append(time.perf_counter() - started)
                relevant = [selected[j]["layout_id"] == query["layout_id"] for j in ranked]
                precision.append(sum(relevant) / 10)
                top1.append(int(relevant[0]))
                chance.append((sum(r["layout_id"] == query["layout_id"] for r in selected) - 1) / (len(selected) - 1))
                stream.write(json.dumps({
                    "query": query["id"], "precision_at_10": precision[-1],
                    "matches": [{"id": selected[j]["id"], "score": float(scores[j]), "relevant": relevant[k]}
                                for k, j in enumerate(ranked)],
                }) + "\n")
                for pair in by_query[query["id"]]:
                    positive = float(scores[index[pair["positive"]]])
                    negative = float(scores[index[pair["negative"]]])
                    outcomes.append({**pair, "positive_score": positive, "negative_score": negative,
                                     "margin": positive - negative})
        with (output / f"pairs-{name}.jsonl").open("w", encoding="utf-8") as stream:
            for outcome in outcomes:
                stream.write(json.dumps(outcome) + "\n")
        summary = {
            "images": len(selected), "queries": len(selected),
            "layouts": len({r["layout_id"] for r in selected}),
            "precision_at_10": float(np.mean(precision)), "top1_accuracy": float(np.mean(top1)),
            "random_expected_precision_at_10": float(np.mean(chance)),
            "preference": preference_metrics([r["margin"] for r in outcomes]),
            "preference_by_change": {change: preference_metrics([r["margin"] for r in outcomes if r["change"] == change])
                                     for change in sorted({r["change"] for r in outcomes})},
            "preference_by_positive_change": {change: preference_metrics([r["margin"] for r in outcomes if r["positive_change"] == change])
                                              for change in sorted({r["positive_change"] for r in outcomes})},
            "extraction_seconds": extraction_seconds,
            "extraction_images_per_second": len(selected) / extraction_seconds,
            "scoring_median_ms": float(np.median(latencies) * 1000),
            "scoring_p95_ms": float(np.percentile(latencies, 95) * 1000),
            "descriptor_bytes": vectors.nbytes,
            "bytes_per_image": descriptor.dimension * np.dtype(np.float32).itemsize,
            "zero_descriptors": int(np.count_nonzero(np.linalg.norm(vectors, axis=1) < 1e-6)),
        }
        results["splits"][name] = summary
        _write_report(root, output, name, selected, outcomes, summary)
        print(f"{name}: {len(selected)} images; precision@10={summary['precision_at_10']:.3f}; "
              f"preference accuracy={summary['preference']['accuracy']:.3f}", file=sys.stderr)
    write_json(output / "metrics.json", results)
    return results


def _write_report(root: Path, output: Path, split: str, records: list[dict], outcomes: list[dict], summary: dict) -> None:
    records_by_id = {r["id"]: r for r in records}
    rows = []
    for outcome in sorted(outcomes, key=lambda p: p["margin"])[:12]:
        figures = []
        for role in ("query", "positive", "negative"):
            record = records_by_id[outcome[role]]
            src = quote(os.path.relpath(root / record["path"], output).replace(os.sep, "/"), safe="/.")
            figures.append(f'<figure><img src="{src}" alt="{html.escape(record["id"])}"><figcaption>'
                           f'{role}: {html.escape(record["subject"])} / {html.escape(record["appearance"])}</figcaption></figure>')
        rows.append(f'<h2>{outcome["change"]}: margin {outcome["margin"]:.4f}</h2><div class="row">'
                    + "".join(figures) + "</div>")
    (output / f"report-{split}.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Composition baseline evaluation</title>"
        "<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:1rem;background:#f4f5f7}"
        ".row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0;background:white;padding:8px}"
        "img{width:100%}figcaption{padding:6px}</style>"
        f"<h1>{split}: synthetic composition baseline</h1><p>Precision@10: {summary['precision_at_10']:.1%}; "
        f"preference accuracy: {summary['preference']['accuracy']:.1%}.</p>"
        "<p>The 12 lowest-margin comparisons are shown below to expose weaknesses. A positive margin means "
        "the same-layout positive outranks the changed-layout negative. These fixtures do not establish photographic accuracy.</p>"
        + "".join(rows) + "</html>", encoding="utf-8",
    )
