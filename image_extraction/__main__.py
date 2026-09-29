"""Local image indexing, video composition retrieval, and evaluation."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

from .dataset import ImageDataset, ImageLoadError
from .download import DATASET_REVISION, download_dataset, write_json


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def inspect_dataset(images: Path, output: Path) -> dict:
    """Decode each image and stream metadata/errors to disk without caching pixels."""
    started = time.perf_counter()
    dataset = ImageDataset(images)
    output.mkdir(parents=True, exist_ok=True)
    usable = invalid = total_bytes = 0
    formats: Counter[str] = Counter()
    with (output / "images.jsonl").open("w", encoding="utf-8") as manifest, (
        output / "errors.jsonl"
    ).open("w", encoding="utf-8") as errors:

        def report_error(error: ImageLoadError) -> None:
            nonlocal invalid
            invalid += 1
            errors.write(json.dumps({
                "relative_path": error.path.relative_to(dataset.root).as_posix(),
                "error": error.reason,
            }) + "\n")
            print(str(error), file=sys.stderr)

        # Use strict access here so an all-corrupt dataset still gets a summary.
        for index in range(len(dataset)):
            try:
                sample = dataset[index]
            except ImageLoadError as error:
                report_error(error)
                continue
            try:
                usable += 1
                total_bytes += sample.size_bytes
                formats[sample.path.suffix.lower()] += 1
                manifest.write(json.dumps({
                    "relative_path": sample.relative_path,
                    "width": sample.source_size[0],
                    "height": sample.source_size[1],
                    "mode": sample.image.mode,
                    "size_bytes": sample.size_bytes,
                }) + "\n")
            finally:
                sample.close()
            if (index + 1) % 100 == 0:
                print(f"Inspected {index + 1}/{len(dataset)} files", file=sys.stderr)

    summary = {
        "root": str(dataset.root),
        "candidate_images": len(dataset),
        "usable_images": usable,
        "invalid_images": invalid,
        "usable_image_bytes": total_bytes,
        "extensions": dict(sorted(formats.items())),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download/resume the pinned car-image dataset")
    download.add_argument("--output", type=Path, default=Path("data/car-images"))
    download.add_argument("--revision", default=DATASET_REVISION)
    download.add_argument("--workers", type=positive_int, default=4)
    inspect = commands.add_parser("inspect", help="Fully decode images and write a metadata inventory")
    inspect.add_argument("--images", type=Path, default=Path("data/car-images"))
    inspect.add_argument("--output", type=Path, default=Path("outputs/car-images"))
    synthetic = commands.add_parser("synthetic", help="Generate labeled synthetic composition scenes")
    synthetic.add_argument("--output", type=Path, default=Path("data/synthetic"))
    synthetic.add_argument("--seed", type=int, default=20260921)
    synthetic.add_argument("--families-per-split", type=positive_int, default=12)
    evaluate = commands.add_parser("evaluate-synthetic", help="Evaluate the fixed spatial descriptor on labeled scenes")
    evaluate.add_argument("--dataset", type=Path, default=Path("data/synthetic"))
    evaluate.add_argument("--output", type=Path, default=Path("outputs/synthetic-baseline"))
    evaluate.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    index = commands.add_parser("index", help="Build/resume portable image-index artifacts")
    index.add_argument("--images", type=Path, default=Path("data/car-images"))
    index.add_argument("--index", type=Path, default=Path("artifacts/car-images"))
    index.add_argument("--search-device", choices=("auto", "cpu", "cuda"), default="auto")
    index.add_argument("--batch-size", type=positive_int, default=64, help="FAISS add batch size")
    index.add_argument("--descriptor", choices=("spatial", "semantic"), default="spatial")
    index.add_argument("--extraction-device", choices=("auto", "cpu", "cuda"), default="auto")
    index.add_argument("--extraction-batch-size", type=positive_int, default=16,
                       help="Maximum decoded images per semantic inference batch")
    index.add_argument("--model-cache", type=Path, help="Semantic model cache (default: project .cache/models)")
    index.add_argument("--semantic-max-patches", type=int, choices=(256, 512, 1024), default=256)
    index.add_argument("--descriptor-max-side", type=positive_int,
                       help="Optionally downsize descriptor input; default retains native resolution")
    index.add_argument("--rebuild", action="store_true", help="Recompute checkpoints and replace this index's generated artifacts")
    query = commands.add_parser("query", help="Retrieve per-shot image matches and write JSON/HTML reports")
    query.add_argument("--video", type=Path, required=True)
    query.add_argument("--index", type=Path, default=Path("artifacts/car-images"))
    query.add_argument("--output", type=Path, default=Path("outputs/video-query"))
    query.add_argument("--search-device", choices=("auto", "cpu", "cuda"), default="auto")
    query.add_argument("--extraction-device", choices=("auto", "cpu", "cuda"), default="auto")
    query.add_argument("--model-cache", type=Path)
    query.add_argument("--top-k", type=positive_int, default=10)
    query.add_argument("--batch-size", type=positive_int, default=16)
    query.add_argument("--interval-seconds", type=positive_float, help="Add interval samples within shots, alongside each midpoint")
    query.add_argument("--threshold", type=positive_float, default=27.0, help="ContentDetector cut threshold")
    query.add_argument("--min-scene-frames", type=positive_int, default=15)
    select = commands.add_parser("select-frames", help="Rank semantic query frames and select distinct photos")
    select.add_argument("--results", type=Path, required=True, help="Completed semantic query results.json")
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--top-k", type=positive_int, default=17)
    select.add_argument("--batch-size", type=positive_int, default=16)
    select.add_argument("--extraction-device", choices=("auto", "cpu", "cuda"), default="auto")
    select.add_argument("--search-device", choices=("auto", "cpu", "cuda"), default="auto")
    select.add_argument("--feedback-profile", type=Path, help="Explicit JSON mapping of workbook visual examples")
    select.add_argument("--review", type=Path, help="Optional reviewed frame IDs/reasons bound to the input checksum")
    select.add_argument("--min-relevance", type=float, default=0.65, help="Uncalibrated relevance cutoff; default 0.65")
    replay = commands.add_parser("replay-selection", help="Reproduce automatic selections from saved scores; no model inference")
    replay.add_argument("--selection", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            metadata = download_dataset(args.output, revision=args.revision, workers=args.workers)
            print(json.dumps({key: value for key, value in metadata.items() if key != "image_files"}, indent=2))
        elif args.command == "synthetic":
            from .synthetic import generate_synthetic

            print(json.dumps(generate_synthetic(args.output, seed=args.seed, families_per_split=args.families_per_split), indent=2))
        elif args.command == "evaluate-synthetic":
            from .evaluation import evaluate_synthetic

            print(json.dumps(evaluate_synthetic(args.dataset, args.output, split=args.split), indent=2))
        elif args.command == "index":
            from .indexing import build_index
            from .descriptors import SpatialDescriptor

            if args.descriptor == "semantic":
                if args.descriptor_max_side is not None:
                    raise ValueError("--descriptor-max-side applies only to spatial descriptors")
                from .semantic import SemanticDescriptor
                descriptor = SemanticDescriptor(device=args.extraction_device, cache_dir=args.model_cache,
                                                max_num_patches=args.semantic_max_patches)
            else:
                if args.extraction_device == "cuda":
                    raise ValueError("Spatial extraction runs on CPU; use --descriptor semantic for CUDA extraction")
                descriptor = SpatialDescriptor(max_side=args.descriptor_max_side)
            print(json.dumps(build_index(args.images, args.index, search_device=args.search_device,
                                         batch_size=args.batch_size, rebuild=args.rebuild, descriptor=descriptor,
                                         extraction_batch_size=args.extraction_batch_size), indent=2))
        elif args.command == "replay-selection":
            from .selection import replay_selection

            print(json.dumps(replay_selection(args.selection, args.output), indent=2))
        elif args.command == "select-frames":
            from .selection import select_video_frames

            print(json.dumps(select_video_frames(
                args.results, args.output, top_k=args.top_k, batch_size=args.batch_size,
                extraction_device=args.extraction_device, search_device=args.search_device,
                feedback_profile=args.feedback_profile, review_path=args.review,
                min_relevance=args.min_relevance), indent=2))
        elif args.command == "query":
            from .query import query_video

            print(json.dumps(query_video(args.video, args.index, args.output, top_k=args.top_k,
                                         search_device=args.search_device, batch_size=args.batch_size,
                                         extraction_device=args.extraction_device, model_cache=args.model_cache,
                                         interval_seconds=args.interval_seconds, threshold=args.threshold,
                                         min_scene_frames=args.min_scene_frames), indent=2))
        else:
            summary = inspect_dataset(args.images, args.output)
            print(json.dumps(summary, indent=2))
            if not summary["usable_images"]:
                print("error: No usable images found", file=sys.stderr)
                return 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
