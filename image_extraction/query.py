"""Query a saved index with selected video frames and write a local contact sheet."""

from datetime import datetime, timezone
from html import escape
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image, ImageOps

from .download import write_json
from .indexing import ensure_separate, file_hash, load_index
from .video import detect_shots, read_selected_frames, select_frames


def save_thumbnail(image, path, max_side=320):
    with image.copy() as thumbnail:
        thumbnail.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        thumbnail.save(path, "JPEG", quality=85)


def save_query_frame(image, path):
    """Export a decoded video frame at its native resolution."""
    image.save(path, "JPEG", quality=95)


def query_video(video, index, output, *, top_k=10, search_device="auto", batch_size=16,
                interval_seconds=None, threshold=27.0, min_scene_frames=15) -> dict:
    if top_k <= 0 or top_k > 2048 or batch_size <= 0:
        raise ValueError("top_k must be 1..2048 and batch_size must be positive")
    if interval_seconds is not None and (not math.isfinite(interval_seconds) or interval_seconds <= 0):
        raise ValueError("interval_seconds must be positive and finite")
    video = Path(video).expanduser().resolve()
    index = Path(index).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    ensure_separate(index, output)
    if not video.is_file():
        raise ValueError(f"Video does not exist: {video}")
    if output == video or output in video.parents:
        raise ValueError("Report directory must not contain the source video")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Report directory is not empty; use a new --output directory")
    started = time.perf_counter()
    manifest, images, descriptor, backend = load_index(index, search_device=search_device)
    load_seconds = time.perf_counter() - started
    source_root = Path(manifest["images_root"])
    ensure_separate(source_root, output)
    video_stat = video.stat()
    print(f"Detecting shots in {video.name}", file=sys.stderr)
    video_info, shots, timestamps = detect_shots(video, threshold=threshold, min_scene_frames=min_scene_frames)
    selected = []
    for shot in shots:
        shot["queries"] = []
        selected.extend((number, shot) for number in select_frames(shot, timestamps, interval_seconds))
    output.mkdir(parents=True, exist_ok=True)
    (output / "queries").mkdir()
    (output / "candidates").mkdir()
    write_json(output / "status.json", {"status": "incomplete", "video": str(video), "index": str(index)})
    print(f"Detected {len(shots)} shots; retrieving {len(selected)} query frames", file=sys.stderr)
    timing = {"index_load_and_device_setup": load_seconds,
              "video_scan_and_shot_detection": video_info.pop("scan_seconds"),
              "selected_frame_decode": 0.0, "query_encoding": 0.0,
              "retrieval": 0.0, "thumbnails": 0.0}
    candidate_cache, thumbnail_errors, search_batches = {}, [], []

    def candidate_thumbnail(entry):
        identifier = entry["id"]
        if identifier in candidate_cache:
            return candidate_cache[identifier]
        source = source_root / entry["relative_path"]
        relative = f"candidates/{identifier:06d}.jpg"
        try:
            if not source.resolve().is_relative_to(source_root):
                raise ValueError("Source path now resolves outside the collection")
            stat = source.stat()
            if (stat.st_size, stat.st_mtime_ns) != (entry["size_bytes"], entry["mtime_ns"]):
                raise ValueError("Source image changed since indexing; rebuild the index")
            with Image.open(source) as original:
                original.load()
                with ImageOps.exif_transpose(original) as oriented, oriented.convert("RGB") as rgb:
                    save_thumbnail(rgb, output / relative)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            thumbnail_errors.append({"path": str(source), "error": str(error)})
            relative = None
        candidate_cache[identifier] = relative
        return relative

    def flush(batch):
        tick = time.perf_counter()
        scores, ids = backend.search(np.stack([item[0] for item in batch]), top_k)
        elapsed = time.perf_counter() - tick
        timing["retrieval"] += elapsed
        search_batches.append({"queries": len(batch), "seconds": elapsed})
        tick = time.perf_counter()
        for row, (_, query) in enumerate(batch):
            query["matches"] = []
            for rank, (score, identifier) in enumerate(zip(scores[row], ids[row]), start=1):
                entry = images[int(identifier)]
                query["matches"].append({"rank": rank, "image_id": entry["id"],
                                         "relative_path": entry["relative_path"],
                                         "source_path": str(source_root / entry["relative_path"]),
                                         "score": float(score), "thumbnail": candidate_thumbnail(entry)})
        timing["thumbnails"] += time.perf_counter() - tick

    frame_reader = read_selected_frames(video, [number for number, _ in selected])
    batch = []
    try:
        for number, shot in selected:
            tick = time.perf_counter()
            actual, pts, pixels = next(frame_reader)
            timing["selected_frame_decode"] += time.perf_counter() - tick
            if actual != number:
                raise ValueError("Selected frame order mismatch")
            if video_info["timestamp_source"] == "opencv_presentation_time" and (
                    not math.isfinite(pts) or abs(pts - timestamps[number]) > max(0.002, 0.5 / video_info["fps"])):
                raise ValueError(f"Timestamp mismatch while decoding selected frame {number}")
            with Image.fromarray(pixels) as image:
                tick = time.perf_counter()
                vector = descriptor.extract(image)
                timing["query_encoding"] += time.perf_counter() - tick
                tick = time.perf_counter()
                thumbnail = f"queries/frame-{number:08d}.jpg"
                save_query_frame(image, output / thumbnail)
                timing["thumbnails"] += time.perf_counter() - tick
            query = {"frame_number": number, "timestamp_seconds": timestamps[number],
                     "thumbnail": thumbnail, "width": pixels.shape[1], "height": pixels.shape[0],
                     "zero_vector": bool(not np.any(vector))}
            shot["queries"].append(query)
            batch.append((vector, query))
            if len(batch) >= batch_size:
                flush(batch)
                batch.clear()
        if batch:
            flush(batch)
    finally:
        frame_reader.close()
    if (video.stat().st_size, video.stat().st_mtime_ns) != (video_stat.st_size, video_stat.st_mtime_ns):
        raise ValueError("Source video changed during processing")
    results = {"schema_version": 1, "status": "complete", "video": video_info,
               "index": {"path": str(index), "images": len(images),
                         "sha256": manifest["files"]["index.faiss"]["sha256"],
                         "descriptor": manifest["descriptor"]},
               "search": backend.metadata(), "top_k": top_k, "batch_size": batch_size,
               "interval_seconds": interval_seconds, "query_count": len(selected), "shot_count": len(shots),
               "shots": shots, "thumbnail_errors": thumbnail_errors,
               "query_frame_export": {"resolution": "native", "format": "JPEG", "quality": 95},
               "timings_seconds": timing, "search_batches": search_batches,
               "python_version": platform.python_version(), "numpy_version": np.__version__,
               "created_at": datetime.now(timezone.utc).isoformat()}
    tick = time.perf_counter()
    render_report(results, output / "report.html")
    timing["html_report"] = time.perf_counter() - tick
    timing["total_before_json_write"] = time.perf_counter() - started
    results["thumbnail_bytes"] = sum(path.stat().st_size for folder in ("queries", "candidates")
                                     for path in (output / folder).glob("*.jpg"))
    write_json(output / "results.json", results)
    write_json(output / "status.json", {"status": "complete", "results": "results.json", "report": "report.html",
                                         "results_sha256": file_hash(output / "results.json")})
    return {"output": str(output), "report": str(output / "report.html"),
            "shot_count": len(shots), "query_count": len(selected), "search": backend.metadata(),
            "thumbnail_errors": len(thumbnail_errors), "timings_seconds": timing}


def render_report(results, path):
    title = f"Composition matches · {Path(results['video']['path']).name}"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(title)}</title>
<style>body{{font:15px system-ui,sans-serif;background:#10141a;color:#e7ecf3;margin:0;padding:24px}}
h1{{margin-bottom:8px}}a{{color:#97caff}}.meta{{color:#a9b7c9}}section{{margin:32px 0;border-top:1px solid #384352}}
.row{{display:flex;gap:14px;overflow-x:auto;padding:12px 0 22px}}figure{{margin:0;flex:0 0 230px;background:#1c2430;padding:10px;border-radius:8px}}
.query{{border:2px solid #79b8ff}}img{{width:230px;height:155px;object-fit:contain;background:#0b0e13}}
figcaption{{overflow-wrap:anywhere;margin-top:8px;font-size:13px}}.missing{{height:155px;display:grid;place-items:center}}
</style><h1>{escape(title)}</h1>
<p class="meta">{results['shot_count']} shots · {results['query_count']} query frames · {results['index']['images']} indexed images · {escape(results['search']['device'])} exact search</p>
<p>Spatial grayscale/edge baseline. Scores indicate ranking similarity, not probabilities. <a href="results.json">Download JSON results</a></p>""")
        if results["search"]["fallback_reason"]:
            stream.write(f"<p class=meta>CPU fallback: {escape(results['search']['fallback_reason'])}</p>")
        if results["thumbnail_errors"]:
            stream.write(f"<p>{len(results['thumbnail_errors'])} source thumbnails unavailable; see JSON for details.</p>")
        for shot in results["shots"]:
            stream.write(f"<section><h2>Shot {shot['id']} · {shot['start_seconds']:.3f}–{shot['end_seconds']:.3f} s</h2>")
            for query in shot["queries"]:
                frame_link = escape(query["thumbnail"], quote=True)
                stream.write(f"<div class=row><figure class=query><a href=\"{frame_link}\"><img loading=lazy src=\"{frame_link}\" alt=\"Query frame\"></a><figcaption>Query · {query['timestamp_seconds']:.3f} s<br>Frame {query['frame_number']} · <a href=\"{frame_link}\">Open frame</a></figcaption>")
                if query["zero_vector"]:
                    stream.write("<p>No layout evidence: uniform frame.</p>")
                stream.write("</figure>")
                for match in query["matches"]:
                    source_link = escape(Path(match["source_path"]).as_uri(), quote=True)
                    stream.write(f"<figure><a href=\"{source_link}\">")
                    if match["thumbnail"]:
                        stream.write(f"<img loading=lazy src=\"{match['thumbnail']}\" alt=\"{escape(match['relative_path'], quote=True)}\">")
                    else:
                        stream.write('<div class=missing>Source unavailable or changed</div>')
                    stream.write(f"</a><figcaption>#{match['rank']} · {match['score']:.4f}<br>{escape(match['relative_path'])}</figcaption></figure>")
                stream.write("</div>")
            stream.write("</section>")
        stream.write("</html>")
