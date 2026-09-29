"""Reference-guided video-frame ranking with explicit, reviewable limitations."""

from contextlib import ExitStack
from datetime import datetime, timezone
from html import escape
from io import BytesIO
import json
import csv
from pathlib import Path
import shutil
import time
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps

from .descriptors import SpatialDescriptor
from .download import write_json
from .indexing import ensure_separate, file_hash, load_index
from .selection_decisions import rank_candidates, explain_step, DECISION_PRECISION

SELECTION_VERSION = "reference-quality-diversity-v2"
WEIGHTS = {"relevance": 0.35, "quality": 0.20, "reference_layout": 0.20, "feedback": 0.25}


def image_quality(image):
    """Cheap photographic proxies, not object completeness or text legibility."""
    import cv2
    with image.convert("L") as gray:
        gray.thumbnail((512, 512), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.uint8)
    return {"laplacian_variance": float(cv2.Laplacian(pixels, cv2.CV_32F).var()),
            "dark_fraction": float(np.mean(pixels <= 8)),
            "bright_fraction": float(np.mean(pixels >= 247))}


def scale(values):
    values = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(values, (10, 90))
    if high - low < 1e-12:
        return np.full(len(values), 0.5)
    return np.clip((values - low) / (high - low), 0, 1)


def diverse_references(ids, scores, reconstruct, *, limit=3, duplicate_similarity=0.9995):
    """Identical reference vectors in different stock folders count only once."""
    chosen, vectors = [], []
    for identifier, score in sorted(zip(ids, scores), key=lambda pair: (-round(float(pair[1]), 6), int(pair[0]))):
        if int(identifier) < 0:
            continue
        vector = reconstruct(int(identifier))
        if any(float(vector @ previous) >= duplicate_similarity for previous in vectors):
            continue
        chosen.append((int(identifier), float(score)))
        vectors.append(vector)
        if len(chosen) == limit:
            break
    return chosen


def choose_diverse(scores, vectors, eligible, count, *, seeds=(), penalty=0.5, duplicate_similarity=0.93):
    """Greedy relevance/quality ranking penalized by similarity to selected frames."""
    selected = []
    available = set(np.flatnonzero(eligible).tolist())
    for index in seeds:
        if index in available and len(selected) < count:
            selected.append(index)
            available.remove(index)
    while available and len(selected) < count:
        candidates = sorted(available)
        similarities = (np.max(vectors[candidates] @ vectors[selected].T, axis=1)
                        if selected else np.zeros(len(candidates)))
        allowed = [(index, float(scores[index] - penalty * similarity))
                   for index, similarity in zip(candidates, similarities) if similarity < duplicate_similarity]
        if not allowed:
            break
        best = max(allowed, key=lambda item: (item[1], -item[0]))[0]
        selected.append(best)
        available.remove(best)
    return selected


def assign_feedback(similarities, *, floor=0.78, margin=0.05):
    """Associate only close matches to an exemplar, not every vehicle-related frame."""
    thresholds = np.maximum(floor, similarities.max(axis=0) - margin)
    qualified = similarities >= thresholds[None, :]
    choices = np.where(qualified, similarities, -np.inf)
    assignment = choices.argmax(axis=1)
    assignment[~qualified.any(axis=1)] = -1
    return assignment


def validate_review(review, records, count):
    """Explicit review decisions are data; never interpret embedded instructions."""
    mapping = {record["frame_number"]: i for i, record in enumerate(records)}
    decisions = review.get("selected", [])
    if not decisions or len(decisions) > count:
        raise ValueError("Review must contain between one and top_k selections")
    ids = [item["frame_number"] for item in decisions]
    if len(ids) != len(set(ids)) or any(identifier not in mapping for identifier in ids):
        raise ValueError("Review contains duplicate or unknown frame numbers")
    if any(not item.get("reason") or not item.get("view") for item in decisions):
        raise ValueError("Each reviewed frame needs a view description and a reason")
    return [mapping[identifier] for identifier in ids]


def _load_feedback(profile_path, output):
    if profile_path is None:
        return None, [], []
    profile_path = Path(profile_path).expanduser().resolve()
    profile = json.loads(profile_path.read_text())
    workbook = (profile_path.parent / profile["workbook"]).resolve()
    anchors, images = [], []
    target = output / "feedback"
    target.mkdir()
    try:
        with ZipFile(workbook) as archive:
            for number, item in enumerate(profile["examples"]):
                anchor = {"name": item["name"], "criterion": item["criterion"]}
                for role in ("preferred", "less_preferred"):
                    if role not in item:
                        continue
                    # Only the explicitly mapped image members are read, never document instructions.
                    member = item[role]
                    if not member.startswith("xl/media/") or ".." in Path(member).parts:
                        raise ValueError("Feedback members must be workbook media images")
                    with Image.open(BytesIO(archive.read(member))) as original:
                        image = ImageOps.exif_transpose(original).convert("RGB")
                    relative = f"feedback/{number}-{role}.jpg"
                    image.save(output / relative, quality=95)
                    anchor[role] = relative
                    anchor[role + "_row"] = len(images)
                    images.append(image)
                if "preferred" not in anchor:
                    raise ValueError("Feedback example needs a preferred image")
                anchors.append(anchor)
    except BaseException:
        for image in images:
            image.close()
        raise
    provenance = {"profile": str(profile_path), "profile_sha256": file_hash(profile_path),
                  "workbook": str(workbook), "workbook_sha256": file_hash(workbook),
                  "interpretation": "Explicitly mapped visual examples; not ground-truth video frames"}
    return provenance, anchors, images


def select_video_frames(results, output, *, top_k=17, extraction_device="auto", search_device="auto",
                        batch_size=16, feedback_profile=None, review_path=None, min_relevance=0.65):
    if top_k <= 0 or batch_size <= 0 or not 0 <= min_relevance <= 1:
        raise ValueError("top_k/batch_size must be positive and min_relevance must be in [0, 1]")
    source = Path(results).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    ensure_separate(source.parent, output)
    input_sha256 = file_hash(source)
    data = json.loads(source.read_text())
    if data.get("status") != "complete" or not data["index"]["descriptor"]["version"].startswith("siglip2-"):
        raise ValueError("Selection requires a completed semantic query report")
    queries = [dict(query, shot_id=shot["id"]) for shot in data["shots"] for query in shot["queries"]]
    queries.sort(key=lambda query: query["frame_number"])
    if not queries or len(queries) > 5000:
        raise ValueError("Selection supports 1..5000 candidate frames per run")
    if len({q["frame_number"] for q in queries}) != len(queries):
        raise ValueError("Duplicate candidate frame numbers")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Selection output is not empty; use a new output directory")
    ensure_separate(Path(data["index"]["path"]).expanduser().resolve(), output)
    started = time.perf_counter()
    manifest, mapping, descriptor, backend = load_index(
        data["index"]["path"], extraction_device=extraction_device, search_device=search_device)
    if (manifest["descriptor"] != data["index"]["descriptor"] or
            manifest["files"]["index.faiss"]["sha256"] != data["index"]["sha256"]):
        raise ValueError("Query results do not match the current reference index")
    ensure_separate(Path(manifest["images_root"]), output)
    output.mkdir(parents=True)
    write_json(output / "status.json", {"status": "incomplete"})
    layout = SpatialDescriptor(max_side=256)
    vectors, spatial, records = [], [], []
    (output / "candidates").mkdir()
    for offset in range(0, len(queries), batch_size):
        with ExitStack() as stack:
            images = []
            for query in queries[offset:offset + batch_size]:
                path = (source.parent / query["thumbnail"]).resolve()
                if not path.is_relative_to(source.parent):
                    raise ValueError("Query image path escapes the report directory")
                image = stack.enter_context(Image.open(path))
                images.append(image)
                relative = f"candidates/frame-{query['frame_number']:08d}.jpg"
                with image.convert("RGB") as thumbnail:
                    thumbnail.thumbnail((400, 225), Image.Resampling.LANCZOS)
                    thumbnail.save(output / relative, quality=85)
                records.append({"frame_number": query["frame_number"], "timestamp_seconds": query["timestamp_seconds"],
                                "shot_id": query["shot_id"], "source_frame": str(path),
                                "source_sha256": file_hash(path), "thumbnail": relative, "quality_raw": image_quality(image)})
                spatial.append(layout.extract(image))
            vectors.extend(descriptor.extract_batch(images))
    vectors, spatial = np.stack(vectors), np.stack(spatial)
    # Wider retrieval avoids letting repeated reference files saturate the saved top ten.
    scores, identifiers = backend.search(vectors, min(64, len(mapping)))
    portable = backend.cpu_index()
    reference_cache = {}
    reference_layout = {}
    (output / "references").mkdir()
    for row, record in enumerate(records):
        matches = diverse_references(identifiers[row], scores[row], portable.reconstruct)
        references, layouts = [], []
        for identifier, score in matches:
            entry = mapping[identifier]
            if identifier not in reference_cache:
                path = Path(manifest["images_root"]) / entry["relative_path"]
                if not path.resolve().is_relative_to(Path(manifest["images_root"])):
                    raise ValueError("Reference image escapes the collection")
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != (entry["size_bytes"], entry["mtime_ns"]):
                    raise ValueError("Reference image changed since indexing")
                with Image.open(path) as original, ImageOps.exif_transpose(original) as image:
                    reference_layout[identifier] = layout.extract(image)
                    with image.convert("RGB") as thumbnail:
                        thumbnail.thumbnail((400, 300), Image.Resampling.LANCZOS)
                        thumbnail.save(output / f"references/{identifier}.jpg", quality=90)
                reference_cache[identifier] = {"image_id": identifier, "source_path": str(path),
                                               "relative_path": entry["relative_path"],
                                               "thumbnail": f"references/{identifier}.jpg"}
            references.append({**reference_cache[identifier], "score": score})
            layouts.append(float(spatial[row] @ reference_layout[identifier]))
        record["references"] = references
        record["relevance_raw"] = float(np.mean([s for _, s in matches]))
        record["reference_layout_raw"] = float(np.mean(layouts))
        record["distinct_reference_count"] = len(matches)
    feedback_source, anchors, feedback_images = _load_feedback(feedback_profile, output)
    assigned = np.full(len(records), -1)
    feedback_scores = np.full(len(records), 0.5)
    anchor_similarity = np.empty((len(records), 0))
    try:
        if anchors:
            feedback_vectors = descriptor.extract_batch(feedback_images)
            feedback_layout = np.stack([layout.extract(image) for image in feedback_images])
            preferred_rows = [anchor["preferred_row"] for anchor in anchors]
            anchor_similarity = vectors @ feedback_vectors[preferred_rows].T
            assigned = assign_feedback(anchor_similarity)
            for i, record in enumerate(records):
                j = int(assigned[i])
                if j < 0:
                    continue
                anchor = anchors[j]
                similarity = float(anchor_similarity[i, j])
                pos = anchor["preferred_row"]
                detail = {"name": anchor["name"], "similarity": similarity}
                if "less_preferred_row" in anchor:
                    neg = anchor["less_preferred_row"]
                    semantic_margin = float(vectors[i] @ (feedback_vectors[pos] - feedback_vectors[neg]))
                    layout_margin = float(spatial[i] @ (feedback_layout[pos] - feedback_layout[neg]))
                    feedback_scores[i] = np.clip(0.5 + 2 * semantic_margin + layout_margin, 0, 1)
                    detail.update(semantic_preference_margin=semantic_margin, layout_preference_margin=layout_margin)
                else:
                    feedback_scores[i] = (1 + float(spatial[i] @ feedback_layout[pos])) / 2
                record["feedback_match"] = detail
    finally:
        for image in feedback_images:
            image.close()
    sharpness = scale([np.log1p(record["quality_raw"]["laplacian_variance"]) for record in records])
    for i, record in enumerate(records):
        raw = record["quality_raw"]
        exposure = max(0.0, 1 - raw["dark_fraction"] - raw["bright_fraction"])
        components = {"relevance": float(np.clip((record["relevance_raw"] - min_relevance) / max(1e-9, 1 - min_relevance), 0, 1)),
                      "quality": float(0.7 * sharpness[i] + 0.3 * exposure),
                      "reference_layout": (1 + record["reference_layout_raw"]) / 2,
                      "feedback": float(feedback_scores[i])}
        record["components"] = components
        record["score"] = sum(WEIGHTS[key] * components[key] for key in WEIGHTS)
        record["eligible"] = record["relevance_raw"] >= min_relevance
    # Freeze rounded pairwise similarities; replay never depends on GPU arithmetic.
    similarities = np.round(vectors.astype(np.float64) @ vectors.astype(np.float64).T, DECISION_PRECISION)
    seed_groups, missing = [], []
    for j, anchor in enumerate(anchors):
        candidates = [i for i, record in enumerate(records) if assigned[i] == j and record["eligible"]]
        if candidates:
            seed_groups.append({"name": anchor["name"], "rows": candidates})
        else:
            missing.append(anchor["name"])
    automatic, trace, dispositions = rank_candidates(
        records, similarities, top_k, weights=WEIGHTS, seed_groups=seed_groups)
    for record, disposition in zip(records, dispositions):
        record["automatic_disposition"] = disposition
        record["weighted_components"] = {key: WEIGHTS[key] * value for key, value in record["components"].items()}
    for step in trace:
        record = records[step["winner"]["row"]]
        record["automatic_reason"] = explain_step(step)
        record["decision_step"] = step["step"]
    review = None
    final = automatic
    if review_path:
        review_path = Path(review_path).resolve()
        review = json.loads(review_path.read_text())
        if review.get("results_sha256") != input_sha256:
            raise ValueError("Review must identify this exact results.json by SHA256")
        final = validate_review(review, records, top_k)
        review = {**review, "path": str(review_path), "sha256": file_hash(review_path)}
    np.save(output / "frame-embeddings.npy", vectors, allow_pickle=False)
    np.save(output / "frame-layouts.npy", spatial, allow_pickle=False)
    np.save(output / "frame-similarities.npy", similarities, allow_pickle=False)
    decisions = {item["frame_number"]: item for item in review["selected"]} if review else {}
    (output / "selected").mkdir()
    for rank, i in enumerate(final, 1):
        record = records[i]
        suffix = Path(record["source_frame"]).suffix.lower()
        relative = f"selected/{rank:02d}-frame-{record['frame_number']:08d}{suffix}"
        shutil.copyfile(record["source_frame"], output / relative)
        if file_hash(output / relative) != record["source_sha256"]:
            raise ValueError("A query frame changed during selection")
        record.update(selected_rank=rank, export=relative)
        if record["frame_number"] in decisions:
            record["review"] = decisions[record["frame_number"]]
    with ZipFile(output / "selected-frames.zip", "w") as archive:
        for i in final:
            relative = records[i]["export"]
            archive.write(output / relative, Path(relative).name)
    selected_vectors = vectors[final]
    for i, record in enumerate(records):
        if final:
            nearest = int(np.argmax(vectors[i] @ selected_vectors.T))
            record["nearest_selected_frame"] = records[final[nearest]]["frame_number"]
            record["nearest_selected_similarity"] = float(vectors[i] @ selected_vectors[nearest])
        else:
            record["nearest_selected_frame"] = None
            record["nearest_selected_similarity"] = None
    report = {"schema_version": 1, "status": "complete", "version": SELECTION_VERSION,
              "results_path": str(source), "results_sha256": input_sha256, "video": data["video"],
              "descriptor": descriptor.metadata(), "extraction": descriptor.runtime_metadata(), "search": backend.metadata(),
              "reference_index": data["index"], "candidate_count": len(records),
              "normalization": {"sharpness_log1p_percentiles_10_90": np.percentile(
                  [np.log1p(r["quality_raw"]["laplacian_variance"]) for r in records], (10, 90)).tolist(),
                  "relevance": "clip((mean distinct reference cosine - min_relevance)/(1-min_relevance),0,1)",
                  "quality": "0.7 * percentile-scaled log1p sharpness + 0.3 * (1-dark_fraction-bright_fraction)",
                  "layout": "(1 + mean spatial cosine to references)/2",
                  "feedback_pair": "clip(0.5 + 2*semantic_preference_margin + layout_preference_margin,0,1)",
                  "feedback_positive_only": "(1 + spatial cosine to preferred example)/2",
                  "feedback_unmatched": 0.5},
              "decision_trace": trace, "feedback_pools": seed_groups,
              "reproducibility": {"decision_precision": DECISION_PRECISION,
                                  "tie_break": "lower frame number after rounding utility to six decimals",
                                  "selection_source_sha256": file_hash(Path(__file__)),
                                  "decision_source_sha256": file_hash(Path(__file__).with_name("selection_decisions.py")),
                                  "scope": "Exact decision replay from frozen components and similarities; fresh inference may differ across software/hardware."},
              "files": {path.relative_to(output).as_posix(): {"sha256": file_hash(path)}
                        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "status.json"}, "target_count": top_k,
              "selected_count": len(final), "mode": "reviewed" if review else "automatic_proposal",
              "settings": {"weights": WEIGHTS, "min_relevance": min_relevance,
                           "reference_shortlist": 64, "reference_duplicate_similarity": 0.9995,
                           "frame_duplicate_similarity": 0.93, "diversity_penalty": 0.5,
                           "feedback_assignment_similarity": 0.78, "feedback_assignment_margin": 0.05, "batch_size": batch_size},
              "feedback_source": feedback_source, "feedback_examples": anchors,
              "unmatched_feedback_examples": missing,
              "automatic_frame_numbers": [records[i]["frame_number"] for i in automatic],
              "selected_frame_numbers": [records[i]["frame_number"] for i in final],
              "review": review, "candidates": records,
              "limitations": ["Scores and thresholds are uncalibrated heuristics, not probabilities.",
                              "Layout similarity and workbook preferences do not detect object boundaries or prove completeness.",
                              "Sharpness and exposure proxies do not establish text readability.",
                              "Unlabeled references cannot guarantee coverage of every part; weak matches do not prove absence.",
                              "Workbook examples guide this video; this is not held-out evaluation."],
              "created_at": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - started}
    if file_hash(source) != input_sha256:
        raise ValueError("Query results changed during selection")
    write_json(output / "selection.json", report)
    write_audit_tables(report, output)
    render_selection(report, output / "report.html")
    write_json(output / "status.json", {"status": "complete", "selected_count": len(final),
                                       "selection_sha256": file_hash(output / "selection.json")})
    return {"output": str(output), "selected_count": len(final), "candidate_count": len(records),
            "mode": report["mode"], "seconds": report["seconds"]}


def render_selection(report, path):
    records = {r["frame_number"]: r for r in report["candidates"]}
    title = f"Selected frames · {Path(report['video']['path']).name}"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>body{{font:16px system-ui;background:#10141a;color:#edf2f7;max-width:1400px;margin:auto;padding:24px}}a{{color:#9fd2ff}}img{{max-width:100%;object-fit:contain}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}}article{{background:#1c2430;padding:14px;border-radius:8px}}.refs{{display:flex;gap:8px}}.refs figure{{margin:0;width:33%;font-size:12px}}.refs img{{height:100px;width:100%}}.main{{width:100%;height:260px}}.muted{{color:#bcc7d6}}summary{{cursor:pointer}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #465161;text-align:left}}</style>
<h1>{escape(title)}</h1><p>{report['selected_count']} selected from {report['candidate_count']} candidates · Target: {report['target_count']}</p>
<p>{'Selection explicitly reviewed; automatic rankings retained for comparison.' if report['review'] else 'Selected entirely by code, with no manual frame overrides. Scores explain the ranking; they do not guarantee framing quality or part coverage.'}</p>
<p><a href="selection.json">Scores and provenance (JSON)</a> · <a href="candidate-scores.csv">Candidate scores (CSV)</a> · <a href="decision-trace.csv">Every decision comparison (CSV)</a> · <a href="selected-frames.zip">Download selected photos (ZIP)</a> · <a href="selected/">Native-resolution frames</a></p>''')
        if report["review"]:
            stream.write('<h2>Review notes</h2>')
            for note in report["review"].get("notes", []):
                stream.write(f'<p>{escape(note)}</p>')
        if report.get("decision_trace"):
            formula = " + ".join(f"{weight:.2f} × {key}" for key, weight in report["settings"]["weights"].items())
            stream.write(f'<h2>Selection rule</h2><p>Base score = {escape(formula)}.</p><p>Workbook slots select the highest base score within their semantic match pool. Remaining slots maximize base score minus {report["settings"]["diversity_penalty"]:.2f} × similarity to the closest previously selected frame. Candidates with similarity ≥ {report["settings"]["frame_duplicate_similarity"]:.2f} are blocked in that second stage. Ties use the lower frame number after six-decimal rounding.</p>')
            stream.write(f'<p>Frames with mean distinct-reference cosine below {report["settings"]["min_relevance"]:.2f} are ineligible. The target is {report["target_count"]} frames; the code returns fewer if no qualifying distinct candidates remain.</p>')
            stream.write('<p>Components are normalized to 0–1. Reference relevance is mean cosine similarity to up to three distinct collection photos. Quality uses sharpness/exposure proxies. Layout compares spatial descriptors; workbook guidance uses the supplied visual preferences. These weights and thresholds are heuristics, not learned probabilities.</p>')
        stream.write('<div class="grid">')
        for number in report["selected_frame_numbers"]:
            r = records[number]; decision = r.get("review", {})
            label = decision.get("view", "Distinct candidate")
            stream.write(f'<article><h2>{r["selected_rank"]}. {escape(label)}</h2><a href="{r["export"]}"><img class="main" src="{r["export"]}"></a><p>{r["timestamp_seconds"]:.2f}s · Frame {number}</p>')
            if decision:
                stream.write(f'<p>{escape(decision["reason"])}</p>')
            elif r.get("automatic_reason"):
                stream.write(f'<p>{escape(r["automatic_reason"])}</p>')
                step = report["decision_trace"][r["decision_step"] - 1]
                runner = step["runner_up"]
                if runner:
                    other = records[runner["frame_number"]]
                    stream.write('<details><summary>Why this frame beat the runner-up</summary>')
                    if other.get("thumbnail"):
                        stream.write(f'<img width="320" loading="lazy" src="{other["thumbnail"]}" alt="Runner-up frame"><p>Runner-up frame {other["frame_number"]} · {other["timestamp_seconds"]:.2f}s</p>')
                    stream.write('<table><tr><th>Weighted component</th><th>Selected</th><th>Runner-up</th><th>Difference</th></tr>')
                    for key, value in r["weighted_components"].items():
                        alternative = other["weighted_components"][key]
                        stream.write(f'<tr><td>{escape(key)}</td><td>{value:.6f}</td><td>{alternative:.6f}</td><td>{value-alternative:+.6f}</td></tr>')
                    stream.write(f'<tr><td>Redundancy penalty</td><td>{step["winner"]["diversity_penalty"]:.6f}</td><td>{runner["diversity_penalty"]:.6f}</td><td></td></tr></table></details>')
            stream.write(f'<p class="muted">Reference relevance: {r["relevance_raw"]:.3f} · Heuristic rank score: {r["score"]:.3f}</p><details><summary>Supporting collection photos and scores</summary><div class="refs">')
            for ref in r["references"]:
                link = escape(Path(ref["source_path"]).as_uri(), quote=True)
                stream.write(f'<figure><a href="{link}"><img loading="lazy" src="{ref["thumbnail"]}"></a><figcaption>{escape(ref["relative_path"])}<br>{ref["score"]:.3f}</figcaption></figure>')
            stream.write('</div><pre>' + escape(json.dumps(r["components"], indent=2)) + '</pre></details></article>')
        stream.write('</div><h2>Workbook framing references</h2><div class="grid">')
        for anchor in report["feedback_examples"]:
            stream.write(f'<article><h3>{escape(anchor["name"])}</h3><p>{escape(anchor["criterion"])}</p><img loading="lazy" src="{anchor["preferred"]}">')
            if "less_preferred" in anchor:
                stream.write(f'<details><summary>Less-preferred example</summary><img loading="lazy" src="{anchor["less_preferred"]}"></details>')
            stream.write('</article>')
        stream.write('</div><details><summary>All candidates and automatic proposal</summary><p>Automatic proposal: ' + escape(str(report["automatic_frame_numbers"])) + '</p><table><tr><th>Frame</th><th>Time</th><th>Score</th><th>Relevance</th><th>Selected / nearest selected view</th></tr>')
        for r in sorted(records.values(), key=lambda r: -r["score"]):
            link = escape(Path(r["source_frame"]).as_uri(), quote=True)
            decision = (f"Selected #{r['selected_rank']}" if "selected_rank" in r else
                        f"Nearest: {r['nearest_selected_frame']} ({r['nearest_selected_similarity']:.3f})"
                        if r["nearest_selected_frame"] is not None else "No qualifying selection")
            if r.get("automatic_disposition") and not report["review"]:
                disposition = r["automatic_disposition"]
                decision = disposition["status"].replace("_", " ")
                if disposition.get("selected_frame") is not None:
                    decision += f" → frame {disposition['selected_frame']} ({disposition['similarity']:.6f})"
            stream.write(f'<tr><td><a href="{link}">{r["frame_number"]}</a></td><td>{r["timestamp_seconds"]:.2f}s</td><td>{r["score"]:.3f}</td><td>{r["relevance_raw"]:.3f}</td><td>{decision}</td></tr>')
        stream.write('</table></details><details><summary>Scoring limitations</summary>')
        for limitation in report["limitations"]:
            stream.write(f'<p>{escape(limitation)}</p>')
        stream.write('</details></html>')


def write_audit_tables(report, output):
    """Plain CSV evidence for comparisons outside the HTML report."""
    fields = ["frame_number", "timestamp_seconds", "eligible", "score", "relevance_raw",
              *report["settings"]["weights"], "automatic_status", "selected_rank"]
    with (output / "candidate-scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in report["candidates"]:
            writer.writerow({**{key: record.get(key, "") for key in fields[:5]},
                             **record["components"],
                             "automatic_status": record["automatic_disposition"]["status"],
                             "selected_rank": record.get("selected_rank", "")})
    with (output / "decision-trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "phase", "feedback_example", "winner",
            "frame_number", "base_score", "nearest_previously_selected", "max_similarity",
            "diversity_penalty", "utility", "status"])
        writer.writeheader()
        for step in report["decision_trace"]:
            for row in step["comparison_pool"]:
                writer.writerow({"step": step["step"], "phase": step["phase"],
                    "feedback_example": step["feedback_example"], "winner": step["winner"]["frame_number"],
                    **{key: value for key, value in row.items() if key != "row"}})


def replay_selection(selection, output):
    """Recompute automatic decisions from checksummed frozen inputs, without inference."""
    selection = Path(selection).expanduser().resolve()
    source = selection.parent
    output = Path(output).expanduser().resolve()
    ensure_separate(source, output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Replay output is not empty; use a new output directory")
    report = json.loads(selection.read_text())
    status = json.loads((source / "status.json").read_text())
    if status.get("status") != "complete" or status.get("selection_sha256") != file_hash(selection):
        raise ValueError("Selection is incomplete or its checksum changed")
    if report.get("version") != SELECTION_VERSION or report.get("review") is not None:
        raise ValueError("Replay requires a current automatic selection with no review override")
    if report["reproducibility"]["decision_source_sha256"] != file_hash(Path(__file__).with_name("selection_decisions.py")):
        raise ValueError("Decision code changed; use the recorded code version for exact replay")
    for name, info in report["files"].items():
        artifact = (source / name).resolve()
        if not artifact.is_relative_to(source) or file_hash(artifact) != info["sha256"]:
            raise ValueError(f"Selection artifact checksum mismatch: {name}")
    similarities = np.load(source / "frame-similarities.npy", allow_pickle=False)
    settings = report["settings"]
    chosen, trace, dispositions = rank_candidates(
        report["candidates"], similarities, report["target_count"], weights=settings["weights"],
        seed_groups=report["feedback_pools"], penalty=settings["diversity_penalty"],
        duplicate_similarity=settings["frame_duplicate_similarity"])
    ids = [report["candidates"][i]["frame_number"] for i in chosen]
    if ids != report["selected_frame_numbers"] or trace != report["decision_trace"]:
        raise ValueError("Replayed decisions do not match the recorded selection")
    if dispositions != [record["automatic_disposition"] for record in report["candidates"]]:
        raise ValueError("Replayed exclusion reasons do not match")
    # The saved exports allow replay even if the original corpus/video is offline.
    for i in chosen:
        record = report["candidates"][i]
        path = (source / record["export"]).resolve()
        if not path.is_relative_to(source) or file_hash(path) != record["source_sha256"]:
            raise ValueError("Saved selected frame changed")
    output.mkdir(parents=True)
    write_json(output / "status.json", {"status": "incomplete"})
    for folder in ("selected", "references", "feedback", "candidates"):
        if (source / folder).exists():
            shutil.copytree(source / folder, output / folder)
    for name in ("frame-embeddings.npy", "frame-layouts.npy", "frame-similarities.npy", "selected-frames.zip"):
        shutil.copyfile(source / name, output / name)
    report["replay"] = {"source": str(selection), "source_sha256": file_hash(selection),
                        "selected_ids_identical": True, "decision_trace_identical": True,
                        "model_inference_performed": False}
    write_json(output / "selection.json", report)
    write_audit_tables(report, output)
    render_selection(report, output / "report.html")
    write_json(output / "status.json", {"status": "complete", "selected_count": len(ids),
                                       "selection_sha256": file_hash(output / "selection.json")})
    return {"output": str(output), "selected_count": len(ids), "replay_identical": True}
