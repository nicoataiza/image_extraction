# Image Extraction

Find images in a local collection with composition similar to frames from a video.
The pipeline detects video shots, samples frames, searches an image index, and
produces an HTML report showing each frame alongside its ranked matches.

The current baseline compares grayscale layout and edge directions using
576-dimensional descriptors and FAISS exact search. CUDA vector indexing and
search are verified on the RTX 5060; image decoding and descriptor extraction run
on CPU at native image resolution. Similarity scores are ranking signals, not confidence percentages, and
retrieval quality on real images still needs labelled evaluation.

## Setup

Requires Python 3.11+. Run from this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[gpu]'
```

Use `.[retrieval]` instead for CPU-only installation. Install only one FAISS
provider: when switching an existing CPU environment to GPU, uninstall
`faiss-cpu` before installing `.[gpu]`.

## Download, index, and search

The real dataset is downloaded by the sibling [WCP_vehicle_images](../WCP_vehicle_images/readme.md)
project, which manages its own environment, credentials, and resumable downloads:

```bash
cd ../WCP_vehicle_images
.venv/bin/python run.py --all
cd ../image_extraction
```

Once downloading succeeds, build the image index. While downloading is active,
use a fixed snapshot for trials so the collection stays unchanged during indexing.

```bash
python -m image_extraction index \
  --images ../WCP_vehicle_images/data/downloads \
  --index artifacts/wcp-vehicle-images --search-device cuda

python -m image_extraction query \
  --video test_video.mp4 --index artifacts/wcp-vehicle-images \
  --output outputs/my-video --search-device cuda \
  --interval-seconds 0.5 --top-k 10
```

Open `outputs/my-video/report.html` in a browser; `results.json` contains scores,
source paths, timestamps, and timings. Exported query frames retain the video’s
native resolution; click a query image to open it at full size. New indexes also
extract descriptors at native resolution, and queries use the index’s stored settings.
Use a fresh output directory for each query.

By default, queries use one midpoint frame per detected shot. The example adds a
frame every 0.5 seconds within each shot. `--search-device cuda` requires GPU
execution; `auto` allows CPU fallback. Interrupted indexing resumes with the same
command; changing the image collection or descriptor resolution requires `--rebuild`
or a new index path. Native extraction uses more CPU time and memory. Existing
256-pixel indexes remain readable; new indexes use native resolution by default.
Use `index --descriptor-max-side 256` to reproduce the old resized mode.
The synthetic evaluator keeps its published 256-pixel baseline settings.

## Project files

- `image_extraction/`: ingestion, descriptors, indexing, video sampling, and retrieval.
- `tests/`: automated checks; run `python -m unittest discover -s tests -v`.
- `data/`, `artifacts/`, `outputs/`: local datasets, indexes, and reports, ignored by Git.
- [AGENTS.md](AGENTS.md): detailed project context, design, benchmarks, and development guidance preserved from the original README.

## Semantic reference-image index

The optional semantic encoder supports image-to-image relevance search without
part labels. It uses the image tower of
[SigLIP 2 base NaFlex](https://huggingface.co/google/siglip2-base-patch16-naflex),
pinned to revision `b53b807d3a2d5e2b3911292f2d69e5341cdc064c`. NaFlex preserves
aspect ratio with patch-aligned resizing and masked token padding; it does not
center-crop the reference photos. The default is 256 patches. Each image produces
one L2-normalized 768-dimensional float32 vector. Similarity is a ranking signal,
not a part-presence probability or a composition-quality score.

Install the encoder alongside the existing FAISS provider:

```bash
.venv/bin/python -m pip install --cache-dir .cache/pip -e '.[semantic]'
.venv/bin/python -m pip check
```

The `semantic` extra does not install or switch FAISS providers. For a fresh
environment use `.[semantic,gpu]` or `.[semantic,retrieval]`, never both GPU and
CPU FAISS extras. Model downloads go to project-local `.cache/models/`; use
`--model-cache PATH` to override. The auxiliary Xet cache defaults inside that
cache unless `HF_XET_CACHE` is already set. The first extraction downloads the
public checkpoint (approximately 1.4 GB); inference runs locally.

Build or resume a **separate** semantic index for the downloaded snapshot:

```bash
.venv/bin/python -m image_extraction index \
  --images data/wcp-all-downloaded \
  --index artifacts/wcp-all-semantic \
  --descriptor semantic \
  --extraction-device cuda --search-device cuda \
  --extraction-batch-size 16
```

`--extraction-device auto|cpu|cuda` controls the encoder independently of
`--search-device`, which controls FAISS. `auto` probes CUDA execution and records
a CPU fallback reason if unavailable; explicit `cuda` fails clearly. Spatial
extraction remains CPU-only. Both encoders keep portable CPU FAISS artifacts.
Use a smaller extraction batch for limited RAM/VRAM; at most that many source
images are decoded for one inference batch. `--batch-size` separately controls
FAISS additions. CPU users may also bound PyTorch threads using `OMP_NUM_THREADS`.

Embeddings and image metadata are checkpointed in `entries.sqlite` after each
complete inference batch. Rerun the same command after interruption: completed
entries, including recorded decode errors, are reused without decoding. A failed
inference leaves its batch uncommitted. Fully cached runs do not load model
weights. Unreadable images are listed in `errors.jsonl`; model/inference failures
stop the build instead of being treated as corrupt images.

The manifest records the pinned model, preprocessing, pooling, normalization,
extraction library versions, extraction/search devices, batch sizes, and timing.
Changing the collection, model/preprocessing settings, or extraction versions
requires a new index or explicit `--rebuild`. Device, cache path, and batch-size
changes can reuse cached vectors. A cache-only resume reports extraction device
`null`, because no inference ran in that invocation.

`--semantic-max-patches 512` or `1024` requests higher-resolution processing and
requires a separate/rebuilt index. `--descriptor-max-side` applies only to the
spatial encoder. Existing spatial indexes and their descriptor settings are
unchanged; spatial vectors are not reused as semantic embeddings.

The existing video query command can load either index type and automatically
uses the saved encoder/settings, with batched semantic extraction:

```bash
.venv/bin/python -m image_extraction query \
  --video videos/VA14022.mp4 --index artifacts/wcp-all-semantic \
  --output outputs/va14022-semantic \
  --extraction-device cuda --search-device cuda --interval-seconds 1
```

This still produces per-frame reference matches using the existing shot/interval
sampling. Reference-guided heuristic frame selection is now available as a separate
`select-frames` step, described below. Object-completeness assessment still needs
visual review. No part labels, relevance
threshold, or ground-truth video frames are assumed.

Offline validation: `python -m unittest discover -s tests -v`. Semantic tests use
a tiny randomly initialized model and do not download weights. A real-model smoke
index was built for all 19 photos in stock `173097`, with local evidence saved in
`outputs/semantic-smoke-173097.json`. This validates extraction/index mechanics,
not retrieval quality or full-collection throughput. The full 238,659-image
semantic index completed on September 28, 2026 in 3,963.9 seconds (66.1 minutes),
with zero invalid images and CUDA extraction/search. Artifacts are in
`artifacts/wcp-all-semantic/`. The VA14022 semantic report is at
`outputs/VA14022/wcp-semantic/report.html`: 382 query frames across 18 shots,
ten matches per frame, 53.6 seconds end to end, and no thumbnail errors. These
are single-run observations, not a retrieval-quality evaluation.
