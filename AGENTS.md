> Current extraction settings (September 23, 2026): new indexes default to native
> image resolution (`max_side=None`, `spatial-gray-edge-v2-native`). Query frames
> are exported at native resolution and encoded using the index's stored settings.
> Explicit `max_side=256` preserves the v1 descriptor and old index compatibility;
> `index --descriptor-max-side 256` selects it. Changing resolution requires a new
> index or `--rebuild`. The synthetic evaluator retains the published v1 baseline.
> Historical 256-pixel measurements below do not describe the native mode.

# Video-to-image composition retrieval

Given a video, find images in a local collection with similar composition: subject placement, relative scale, framing, and foreground/background arrangement. Images may contain different subjects and still be good matches.

**Status:** dataset ingestion, synthetic evaluation, resumable FAISS indexing, video processing, and JSON/HTML retrieval reports are implemented. Index creation and search support CPU and CUDA; the RTX 5060 setup is described below. Image decoding and the spatial descriptor run on CPU.

The initial target is approximately **100,000 images** on a local machine with an **RTX 5060**, with CPU fallback. The prototype will measure retrieval quality, processing time, memory, and disk usage before setting performance commitments.

## Task list and accomplishments

- [x] Inspect the car dataset's actual layout and pin its revision.
- [x] Implement a resumable download command with dataset attribution and provenance.
- [x] Implement deterministic local discovery and on-demand RGB loading.
- [x] Support bounded batches, EXIF orientation, and aspect-preserving downsizing.
- [x] Add a full-decode inspection command with metadata and error reports.
- [x] Test corrupt/truncated files, empty datasets, orientation, batching, and download state.
- [x] Download and fully inspect all 538 images: 538 usable, zero decode failures.
- [x] Generate 1,152 controlled synthetic scenes across 96 layouts with 1,728 preference pairs.
- [x] Separate development and held-out test families; record seeds, versions, geometry, and checksums.
- [x] Implement and evaluate the 576-dimensional grayscale/edge spatial descriptor.
- [x] Publish development/test metrics, saved vectors, rankings, and visual failure reports.
- [x] Pass 26 tests and extract finite, normalized descriptors from all 538 car photographs.
- [x] Build a resumable image index with FAISS exact search on CPU or GPU, video frame selection, and retrieval reports.
- [ ] Compare learned features with GPU extraction and CPU fallback, then benchmark CPU/GPU search at up to 100,000 images.

## Real dataset: WCP vehicle images

Use the downloader in the sibling `../WCP_vehicle_images` repository. It reads
`data/to_download_stock_numbers.csv`, downloads only vehicle images, and stores
images in `../WCP_vehicle_images/data/downloads/<stock_number>/`. The indexer reads
these folders recursively, so no copying or renaming is needed.

Set up the downloader once:

```bash
cd ../WCP_vehicle_images
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# Only create .env if it does not already exist, then fill in the credentials.
test -f .env || (umask 077; cp .env.example .env)
```

Set `WORKFLOW_APPLICATION_NAME` and `WORKFLOW_APPLICATION_SECRET` in that `.env`.
The template includes the Workflow URL, local paths, and download concurrency.
Keep the default `DOWNLOADS_DIR` for the commands below. Always run the downloader
from its own repository so it finds the correct `.env` and progress files.

First download the entire stock list (rerun the same command to resume):

```bash
cd ~/projects/WCP_vehicle_images
.venv/bin/python run.py --all
.venv/bin/python run.py --progress
```

Wait for `--all` to finish successfully before indexing. Failed stocks cause a
nonzero exit code; rerun the download to retry them. Stocks with no vehicle images
are recorded as complete. `--progress` reports recorded completions; rerunning
`--all` also checks for missing stock folders. Avoid changing the image collection
while an index is being built.

Then build/resume the real dataset index using the existing retrieval environment:

```bash
cd ~/projects/image_extraction
.venv/bin/python -m image_extraction index --images ../WCP_vehicle_images/data/downloads --index artifacts/wcp-vehicle-images --search-device cuda
```

The index command decodes images and records unreadable files in
`artifacts/wcp-vehicle-images/errors.jsonl`. Its manifest reports usable and invalid
counts. Interrupted indexing can resume with the same command once the download
is finished; a later change to the collection requires `--rebuild` or a new index
directory. The test dataset and its index remain available at their existing paths.

Query the real dataset explicitly:

```bash
.venv/bin/python -m image_extraction query --video test_video.mp4 --index artifacts/wcp-vehicle-images --search-device cuda --output outputs/wcp-video-query
```

## RTX 5060: GPU vector indexing and search

The project uses the official [FAISS GPU wheel](https://pypi.org/project/faiss-gpu/1.15.1/).
The `gpu` extra installs FAISS 1.15.1, its CUDA runtime dependencies, and the video
retrieval dependencies. Install only one FAISS provider: `faiss-cpu` and `faiss-gpu`
share the same Python module and must not be installed together. On this Linux
x86-64 machine, switch the existing environment with:

```bash
.venv/bin/python -m pip uninstall -y faiss-cpu
.venv/bin/python -m pip install -e '.[gpu]'
.venv/bin/python -m pip check
```

For a fresh GPU environment, only the install command is needed. Use `.[retrieval]`
for a CPU environment, or `.[video]` when managing FAISS separately. Do not install
`.[retrieval]` over the GPU setup, since that adds the conflicting CPU provider.

Both `index` and `query` default to `--search-device auto`, which probes real CUDA
add/search operations and uses the GPU when they succeed. Pass `--search-device cuda`
to require GPU execution and fail clearly if it is unavailable. The WCP workflow
uses explicit `cuda`. Generated manifests and query results record `search.device`,
`search.index_type`, `search.gpu_device`, and any automatic fallback reason.

Image decoding, resizing, and the 576-dimensional spatial descriptor still run on
CPU. CUDA holds the vector index and executes similarity searches. Index creation
extracts descriptors on CPU first, then adds them to the GPU index in batches and
saves a portable CPU representation. GPU utilization can therefore remain low
during image extraction; this is expected and does not indicate search fallback.

Verify the GPU and run tests that fail if CUDA is unavailable:

```bash
nvidia-smi
IMAGE_EXTRACTION_REQUIRE_CUDA=1 .venv/bin/python -m unittest discover -s tests -p test_retrieval.py -v
```

Verified on September 23, 2026 with Python 3.14.4, FAISS GPU 1.15.1, CUDA runtime
12.9.79, driver 595.84, and the RTX 5060 (8,151 MiB VRAM): all 17 retrieval tests
passed, including GPU indexing/resume, CPU/GPU portable index loading, video
queries, and 576-dimensional search agreement at top-k 10 and 2,048. These checks
establish correctness, not an end-to-end indexing speedup.

## Quick start: dataset ingestion

Python 3.11 or newer is required. This step uses the CPU and does not require PyTorch or CUDA.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --cache-dir .cache/pip -e .

# Download 538 images plus the upstream dataset card (approximately 575 MB).
python -m image_extraction download

# Fully decode every candidate and write metadata and error reports.
python -m image_extraction inspect

# Run the offline test suite.
python -m unittest discover -s tests -v
```

The download defaults to `data/car-images/` and the pinned revision
`0dca057af12e29f55febea065eb96b43ab5d18f9` of
[Smily6820/car-images](https://huggingface.co/datasets/Smily6820/car-images/tree/0dca057af12e29f55febea065eb96b43ab5d18f9).
Images are stored directly at the dataset root, as in the upstream repository.
Rerun the same command after an interruption: the Hugging Face client reuses completed downloads and resumes incomplete files where supported. Four download workers are used by default; change this with `--workers`.

`dataset-source.json` records the source URL, exact revision, declared license, expected image paths/count, downloaded image bytes, and completion time. `status: downloaded` means the transfer finished; `inspect` separately checks that the images decode. The upstream `README.md` is retained for attribution. Use a separate output directory for a different `--revision` to avoid mixing snapshots.

Download metadata and default Hub/Xet caches live inside the ignored dataset directory. Explicit `HF_HOME` or `HF_XET_CACHE` environment settings override those cache defaults. Image files are downloaded directly to the destination rather than duplicated in a separate Hub blob cache.

Inspection defaults to `outputs/car-images/` and writes:

| File | Contents |
| --- | --- |
| `summary.json` | Candidate, usable, and invalid counts; usable image bytes; extension counts; elapsed time |
| `images.jsonl` | One record per usable image with relative path, orientation-corrected dimensions, RGB mode, and file size |
| `errors.jsonl` | One record per unreadable image with relative path and failure reason |

Inspection decodes one image at a time, closes its pixel buffer after recording metadata, and does not modify the dataset. Rerunning inspection replaces the previous reports. Unreadable files are reported and skipped; an empty or entirely unusable collection exits unsuccessfully. A partially usable collection succeeds and reports its invalid count. An interrupted inspection must be rerun; inspection is not resumable yet.

To inspect another local collection:

```bash
python -m image_extraction inspect --images /path/to/images --output outputs/my-collection
```

### Python loader

```python
from image_extraction import ImageDataset

dataset = ImageDataset("data/car-images", max_side=518)
print(f"{len(dataset)} candidate files")

for batch in dataset.iter_batches(batch_size=16):
    try:
        for sample in batch:
            print(sample.relative_path, sample.source_size, sample.image.size)
            # Pass sample.image to SpatialDescriptor.extract(), shown below.
    finally:
        for sample in batch:
            sample.close()
```

Discovery stores sorted paths, not image pixels. It recursively accepts JPEG, PNG, WebP, BMP, and TIFF extensions, ignores hidden paths and symlinks, and uses each relative path as its stable identifier. `len(dataset)` counts candidate files, including corrupt files not yet decoded.

`dataset[index]` loads one image and raises `ImageLoadError` if decoding fails. Iteration and `iter_batches` skip invalid images with a warning; pass `on_error=callback` to collect failures instead. An entirely unusable collection raises `ValueError`. Animated or multipage files use their first frame/page.

Images are detached from their source file handles, corrected for EXIF orientation, and converted to RGB. Alpha is discarded during RGB conversion. Optional `max_side` downsizes without cropping, stretching, padding, or upscaling; `source_size` preserves the display dimensions before resizing. Batches are lists because image sizes vary. The caller owns returned images and should close them after processing. Memory depends on image resolution and the number of retained samples; downsizing still requires decoding the source image first.

### First verified run

Validated locally on September 21, 2026 using Python 3.14.4, Pillow 12.3.0, and huggingface-hub 1.32.0:

| Check | Observed result |
| --- | --- |
| Downloaded image files | 538, totaling 574,743,638 bytes (574.7 MB / 548.1 MiB) |
| Full-decode inspection | 538 usable; zero invalid; 11.518 seconds |
| File extensions | 421 `.jpg`, 67 `.jpeg`, 42 `.png`, 8 `.webp` |
| Batched RGB loading with `max_side=518` | All 538 loaded; 33 batches of 16 and one of 10; 16.023 seconds |
| Repeat download | Existing files reused; command completed in under one second |
| Offline tests | 12 passed |
| Package installation | Editable installation and CLI verified; dependency check passed |

The inspected path set and byte total match the download manifest. Local evidence is in `data/car-images/dataset-source.json` and `outputs/car-images/{summary.json,images.jsonl,errors.jsonl}`; these generated files remain ignored by Git. Timings are single-run ingestion observations on this machine, not retrieval benchmarks or performance commitments. No GPU was used, and peak memory has not yet been measured.

## Synthetic scenes and the spatial baseline

After installing the package, run:

```bash
# Deterministically generate the default scenes, labels, and HTML overview.
python -m image_extraction synthetic

# Development evaluation is the default when selecting or tuning settings.
python -m image_extraction evaluate-synthetic

# Evaluate held-out scenes once settings are fixed; use a separate report directory.
python -m image_extraction evaluate-synthetic --split test --output outputs/synthetic-test

# Reproduce the complete baseline report using the fixed default descriptor.
python -m image_extraction evaluate-synthetic --split all
```

### Dataset and label protocol

The generator renders geometric car, tree, and person silhouettes at **384×256** pixels. It uses 12 families per split, four layouts per family, three subjects, and four appearances: neutral, warm, cool, and textured. This gives **576 images and 48 layouts per split**. These are controlled fixtures rather than photorealistic scenes.

Each family has a base layout and three alternatives: mirrored horizontal subject position, 1.5× subject width/height, or a shifted background horizon. Within a layout, all subjects use the same normalized bounding box. Appearance changes preserve geometry; texture noise is held fixed between same-subject/same-appearance layout variants. A family and all its variants stay entirely in `dev` or `test`; exact geometry is checked for cross-split overlap.

Retrieval relevance means the **same layout ID**, excluding the query itself. Each query has 11 relevant candidates. Precision@10 is the fraction of its ten matches with that layout, averaged across every image in the split. Top-1 can benefit from same-subject/different-appearance matches, so it does not by itself establish cross-subject composition matching.

The harder preference test uses base-layout queries. A positive has the same layout but a different subject, with either the same or a different appearance. A negative keeps the query's subject and appearance while changing position, scale, or background. There are **864 labeled comparisons per split**. Accuracy is wins divided by all comparisons; margins within `1e-6` of zero are reported separately as ties and do not count as wins.

The default seed is `20260921`. Use `--seed` and `--families-per-split` to generate another experiment in a new `--output` directory. Repeating an identical configuration regenerates the same files; mismatched configurations or dependency versions require a new directory. The first benchmark used Pillow 12.3.0 and NumPy 2.5.3. Byte-for-byte reproducibility assumes those recorded versions as well as the same seed/configuration.

Generated files live in the ignored `data/synthetic/` directory:

| File | Purpose |
| --- | --- |
| `images/{dev,test}/*.png` | Deterministic rendered fixtures |
| `manifest.json` | Completion state, generator settings, versions, counts, and label checksums |
| `images.jsonl` | Image IDs, paths, split/family/layout labels, normalized geometry, subject, appearance, seed, and image checksum |
| `pairs.jsonl` | Explicit query/positive/negative IDs and the kind of geometry/appearance change |
| `preview.html` | Visual overview of one family from each split |

Ground truth is generated from scene geometry, independently of descriptor scores. Evaluation verifies labels and image checksums, rejects incomplete or inconsistent datasets, and does not silently skip missing/corrupt fixtures. Treat the held-out results below as a baseline, and use development data for subsequent tuning; repeatedly tuning against these test results would invalidate their held-out interpretation.

### Descriptor definition and Python API

`SpatialDescriptor` version `spatial-gray-edge-v1` implements the following fixed defaults:

1. Apply EXIF orientation, convert to grayscale, and downsize the longest side to at most 256 pixels without upscaling, cropping, or padding.
2. Smooth with a Gaussian radius of 1 pixel, remove global mean brightness, and standardize contrast.
3. Pool mean intensity into an 8×8 spatial grid: 64 values.
4. Compute image gradients and pool magnitude-weighted, unsigned edge orientations into eight interpolated bins per cell: 512 values.
5. L2-normalize the intensity and edge blocks separately, combine with equal weights, and normalize the final vector.

The result is **576 float32 values / 2,304 bytes per image**. Unit vectors can be compared with a dot product. Uniform images return a zero vector because they contain no layout evidence. Grids use relative frame coordinates while gradient computation preserves pixel aspect ratio; no padded region enters the descriptor. This baseline does not explicitly encode aspect ratio as an additional feature.

```python
from image_extraction import ImageDataset
from image_extraction.descriptors import SpatialDescriptor

dataset = ImageDataset("data/car-images")
descriptor = SpatialDescriptor()
sample = dataset[0]
try:
    vector = descriptor.extract(sample.image)
    print(vector.shape, vector.dtype, vector.nbytes)  # (576,) float32 2304
finally:
    sample.close()

print(descriptor.metadata())  # Version and preprocessing settings for future indexes.
```

The implementation uses [Pillow drawing](https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html) for labeled fixtures and [NumPy gradients](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) for spatial edge features. It requires no training or GPU.

### Measured baseline results

The defaults above were fixed before evaluating the held-out split. On September 21, 2026, with seed `20260921`, Python 3.14.4, Pillow 12.3.0, and NumPy 2.5.3:

| Metric | Development | Held-out test |
| --- | --- | --- |
| Images / preference comparisons | 576 / 864 | 576 / 864 |
| Precision@10 | 48.94% | 52.52% |
| Expected precision@10 under random ranking | 1.91% | 1.91% |
| Top-1 same-layout accuracy | 97.05% | 97.22% |
| Cross-subject composition preference accuracy | 81.83% | 84.14% |
| Position-change preference accuracy | 100% | 100% |
| Background-change preference accuracy | 100% | 100% |
| Scale-change preference accuracy | 45.49% | 52.43% |
| Preference ties | 0 | 0 |
| Decode + feature extraction | 2.072 s | 2.310 s |
| Median / p95 scoring time per query | 0.032 / 0.038 ms | 0.031 / 0.038 ms |
| Raw descriptor array size | 1,327,104 bytes | 1,327,104 bytes |

The baseline detects the large position and horizon changes in these fixtures, but **scale matching across subjects is weak**. Its worst comparisons often prefer the same subject at the wrong size. Exact layout-ID labels also treat nearby layouts from other families as irrelevant, even when they look similar. These measurements establish a repeatable baseline, not photographic composition accuracy.

All **26 automated tests passed**. A separate smoke test extracted finite unit-length descriptors from all **538 car photographs**, with no zero vectors, in **15.945 seconds**. That verifies real-image compatibility, not retrieval quality; its local record is `outputs/car-spatial-smoke.json`.

The full synthetic report is in `outputs/synthetic-baseline/`:

- `metrics.json`: split metrics, descriptor settings, dataset fingerprint, and environment.
- `features-{dev,test}.npy` and matching `.json`: descriptor matrices with version/settings and ordered image IDs.
- `rankings-{dev,test}.jsonl`: per-query top-ten matches and relevance flags.
- `pairs-{dev,test}.jsonl`: positive/negative scores and margins for every preference comparison.
- `report-{dev,test}.html`: the 12 lowest-margin comparisons, with source images side by side. Keep the generated dataset in place to view the images.

The evaluator decodes one image at a time, holds the small descriptor matrix in memory, and scores one query at a time without materializing an all-pairs score matrix. It still performs exhaustive comparisons; it is not the future scalable image index. Scoring times include dot products and sorting over just 576 candidates. Timings are single-run CPU observations, and extraction overlapped with the car-image smoke test. Peak memory, cold/warm repeats, and 100,000-image latency remain unmeasured. At the current descriptor size, 100,000 raw vectors alone would require about 219.7 MiB, excluding index overhead and working memory.

## Intended workflow

1. Index a local image directory once.
2. Provide a video and select an existing index.
3. Detect shots and select representative frames.
4. Retrieve the ten highest-ranked images for each selected frame by default.
5. Review a static HTML contact sheet and machine-readable JSON results.

Results will be grouped by shot and include its time range, query-frame timestamp, query thumbnail, candidate thumbnails, source paths, ranks, and similarity scores. Scores are ranking signals, not calibrated probabilities. Original images will be referenced rather than copied; reports will generate only the thumbnails they need.

For the first prototype, a “scene” means a detected shot. Semantic grouping across multiple shots is outside the initial scope. Optional interval sampling within long shots will capture changes in composition that a midpoint frame misses.

## Proposed approach

### Image indexing

- Read local images without modifying the originals; keep source paths and metadata alongside reusable descriptors.
- Process bounded batches, record progress, and resume interrupted indexing without recomputing completed entries.
- Include FAISS exact search on CPU or GPU in the first indexing milestone; save a portable CPU index to disk and load it onto the GPU when selected.
- Record the model, descriptor, and preprocessing settings with each index. Reject incompatible settings and require an explicit rebuild.
- Preserve image aspect ratios. Exclude padding from feature pooling so padding does not become a composition signal.
- Report unreadable images and continue indexing usable files. An empty usable collection should produce a clear error.

### Video processing

Use [PySceneDetect](https://www.scenedetect.com/docs/head/) to identify shot boundaries and initially select the midpoint of each shot. Treat a video without detected cuts as one shot. Offer configurable interval sampling within long shots.

Decode frames as needed without persisting every frame. Preserve shot boundaries and query timestamps in the results. Report videos that cannot be decoded or contain no usable frames.

### Composition descriptors

Compare two approaches before choosing a default:

| Approach | Proposed representation | What the experiment should establish |
| --- | --- | --- |
| Lightweight spatial baseline | Spatial grids of grayscale intensity and edge orientation | Whether inexpensive layout cues retrieve useful compositions |
| Learned spatial features | Spatially pooled patch features from [DINOv2-small](https://github.com/facebookresearch/dinov2) | Whether learned features improve composition matching enough to justify their resource cost |

Spatial pooling must retain the relative locations of features. Overall semantic similarity alone is insufficient: a centered car should not automatically outrank a differently named subject that better matches the query's placement and framing.

DINOv2 is a candidate feature extractor, not a demonstrated solution to this project's composition objective. The lightweight descriptor now has the synthetic baseline measurements above; photographic composition accuracy and learned features remain unevaluated. No model training is required for the initial prototype.

Feature extraction and vector indexing/search have separate device requirements. The existing Pillow/NumPy spatial descriptor will remain on CPU; its vectors can still be indexed and searched on GPU. The later DINOv2 comparison will add GPU feature extraction with bounded batches and CPU fallback. Composition quality will be evaluated separately from device performance.

### Retrieval

Begin with L2-normalized float32 descriptors and exact inner-product search using [FAISS](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index), supporting both CPU and GPU from the first indexing milestone. Preserve the spatial descriptor's documented zero-vector behavior. Use CPU exact search as the correctness reference for GPU results and future approximate search.

Keep the index resident on the selected device and search video-frame descriptors in bounded batches. Save indexes in portable CPU form; convert them to GPU form when loading for GPU search, and back to CPU form before saving an index built on GPU, as described in the [FAISS GPU documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU). Benchmark single-frame and batched queries on both backends, including loading and transfer costs; GPU speedups remain to be measured.

Store compact descriptors rather than full patch-token arrays. Measure exact-search latency and index size before adding approximate indexes, dimensionality reduction, compression, or candidate reranking. Assess any optimization against both retrieval quality and resource usage.

### Resource management

Hardware observed with `nvidia-smi` on September 21, 2026: **NVIDIA GeForce RTX 5060**, **8,151 MiB total VRAM**, and **driver 595.84**. CUDA/FAISS index creation and search were verified on September 23, 2026; see the RTX 5060 setup above. Total VRAM is not the memory currently available to the application.

At the current 576-dimensional float32 descriptor size, 100,000 vectors require approximately **220 MiB** (219.7 MiB) before index overhead and working memory. Budget separately for FAISS temporary memory, query batches, and any learned model weights and inference buffers.

- Keep ingestion and the existing spatial descriptor on CPU; use the GPU for planned learned feature extraction with bounded batches and CPU fallback.
- Support GPU vector indexing/search independently of the feature extractor, retaining CPU search and reporting the selected backend.
- Record actual GPU memory, system memory, and software versions; do not infer available VRAM from the GPU model name.
- Separate one-time indexing costs from video decoding, query-frame encoding, retrieval, and report generation.
- Measure cold runs separately from warm runs with the model and index already loaded.
- Store model caches, generated fixtures, descriptors, indexes, and reports outside tracked source files.

## CLI and local storage

The retrieval commands are available:

```bash
python -m image_extraction index --images data/images --index artifacts/main --search-device auto
python -m image_extraction query --video data/videos/example.mp4 --index artifacts/main --search-device auto --top-k 10 --output outputs/example
```

`index` builds or resumes a reusable image index. `query` produces JSON and an HTML
report. A dedicated `benchmark` command remains planned.

Both commands accept `--search-device auto|cpu|cuda`. The default, `auto`, uses the
GPU when the CUDA/FAISS probe succeeds and otherwise reports a CPU fallback reason.
`cpu` forces CPU execution; explicit `cuda` fails if GPU execution is unavailable.
For `index`, this option selects the vector-index backend; spatial descriptor
extraction remains on CPU.

| Directory | Intended contents | Tracked in Git |
| --- | --- | --- |
| `data/` | Local image collections, downloaded datasets, input videos, generated fixtures | No |
| `artifacts/` | Descriptors, indexes, indexing metadata | No |
| `outputs/` | JSON results, HTML reports, thumbnails, benchmark measurements | No |
| `.cache/` | Project-local model and dataset caches | No |

Configure download and model tools to use these locations. Dataset attribution, generation code, evaluation definitions, and application source should remain tracked. The `.gitignore` excludes the local dataset files rather than their source URL or documentation.

## Prototype data

### Controlled synthetic scenes

The implemented `synthetic` command generates deterministic scenes with known layouts, varying object position, relative size, background divisions, and subject identity. It includes:

- Positives with the same layout but different subjects or appearances.
- Negatives with the same subject but different placement, size, or framing.
- Variations in color and texture that preserve the intended layout.

Development and evaluation families are separate, with generation seeds and layout labels recorded. See the dataset protocol and measured baseline above. Controlled scenes provide a composition test; they do not establish real-world retrieval quality.

### Car photographs

Use [Smily6820/car-images](https://huggingface.co/datasets/Smily6820/car-images) for initial real-image validation. Its dataset page declares **CC BY 4.0**; retain source attribution and license information when preparing the local dataset and any redistributed examples.

The pinned dataset is downloaded under `data/car-images/`, with its revision recorded in `dataset-source.json`. All 538 image files passed full decoding. Treat it as a small validation collection rather than the 100,000-image scale benchmark.

Manually label a small held-out composition evaluation set and inspect the resulting contact sheets. Car-only results are limited domain evidence and do not demonstrate cross-subject generalization.

## Validation and milestones

| Milestone | Deliverable | Acceptance evidence |
| --- | --- | --- |
| 1. Controlled benchmark — complete | Deterministic layouts, relevance labels, and spatial baseline | 1,152 scenes; separate dev/test metrics; saved rankings and failure reports |
| 2. Real-image prototype | Resumable car-image index with FAISS CPU/GPU exact search, followed by video queries, JSON and HTML reports, and learned-feature comparison | Verified resume and CPU/GPU search agreement, inspectable per-shot results, and a comparison of baseline and learned descriptors |
| 3. Scale benchmark | CPU/GPU measurements at increasing collection sizes up to 100,000 images | Actual dataset sizes, single-frame/batched throughput and latency, loading/transfer overhead, memory, and disk measurements |
| 4. Optimization decision | Documented quality/resource comparison | Next changes justified by measured bottlenecks and quality tradeoffs |

Use **precision@10** for labeled retrieval queries and **composition preference accuracy** for paired comparisons: how often a same-layout/different-subject positive outranks a same-subject/different-layout negative. Report ties separately. Keep tuning and held-out evaluation results distinct.

Report the following for each descriptor, collection size, and CPU/GPU search backend:

- Indexing wall time and images processed per second, separating feature extraction from vector-index construction.
- Video decoding, query-frame encoding, retrieval, and report-generation times separately.
- Index/model loading and CPU/GPU transfer times, with their contribution to end-to-end latency.
- Single-frame and batched-query throughput, cold/warm end-to-end query times, and per-frame retrieval latency, including median and p95 across repeated queries. Synchronize GPU work before stopping timers.
- Peak system RAM and GPU VRAM, including search working memory and any learned-feature extraction.
- Descriptor, index, and report disk usage, separately from original dataset storage and model weights.
- Hardware, software versions (including driver, CUDA, and FAISS when used), descriptor settings, extraction/search devices, batch sizes, dataset size, and query count.

Label synthetic scale results separately from real-collection results. Repeating a small set of images can exercise storage and runtime but does not establish retrieval quality or corpus diversity at 100,000 images.

Functional checks should cover unreadable images, empty collections, videos without cuts, invalid videos, mixed aspect ratios, long-shot interval sampling, interrupted indexing, incompatible cached descriptors, and collections containing fewer than ten candidates. Verify that reported timestamps and source paths correspond to the selected frames and images.

For CPU/GPU search, verify score agreement within floating-point tolerance and matching rankings except for tied or near-tied scores, portable index save/load across backends, reported CPU fallback under `auto`, and clear failure for unavailable explicit `cuda` execution.

A successful initial prototype produces inspectable per-shot matches and reproducible quality/resource measurements. Numeric accuracy, latency, and memory targets remain unset until the baseline is measured. Publish measured results here as milestones are completed.

## Initial boundaries

The first implementation is local and single-user, with a CLI and static reports. It does not require a hosted service, an interactive web application, model training, semantic scene grouping, or copying the original collection into a second repository. Composition quality, generalization across subjects, and the best descriptor remain experimental questions.

## Semantic indexing implementation (September 28, 2026)

`index --descriptor semantic` now supports pinned SigLIP 2 base NaFlex image
embeddings, bounded CPU/CUDA inference batches, resumable SQLite checkpoints,
and portable CPU/GPU FAISS search. See README's **Semantic reference-image index**
section for installation and full-collection commands. Semantic dependencies are
optional and do not switch FAISS providers. Spatial defaults remain unchanged.
The `query` command loads the stored encoder and supports batched semantic
encoding. Reference-guided heuristic selection is now available below;
object-completeness assessment still needs visual review. No part labels or ground-truth video frames have been added.

Verification: 55 tests passed with `IMAGE_EXTRACTION_REQUIRE_CUDA=1`; dependency
check passed. All 19 photos in stock `173097` produced normalized semantic vectors
using CUDA, then matched themselves at rank 1 using CPU-extracted query vectors.
Maximum CPU/GPU embedding difference was `4.45e-6`; search-score difference was
`2.39e-7`. Resume reused all 19 entries without loading the encoder. Evidence:
`outputs/semantic-smoke-173097.json`. This is functional validation, not a
retrieval-quality evaluation or full-collection benchmark. The full semantic
collection index completed on September 28, 2026: 238,659 usable images, zero
invalid images, CUDA extraction/search, 3,963.9 seconds total. The index is
`artifacts/wcp-all-semantic/`. VA14022 was queried against it with half-second
interval sampling: 382 frames across 18 shots, ten semantic matches per frame,
53.6 seconds total, no thumbnail errors. Report:
`outputs/VA14022/wcp-semantic/report.html`. Retrieval quality remains unevaluated.
