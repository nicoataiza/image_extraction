# Image Extraction

Find images in a local collection with composition similar to frames from a video.
The pipeline detects video shots, samples frames, searches an image index, and
produces an HTML report showing each frame alongside its ranked matches.

The current baseline compares grayscale layout and edge directions using
576-dimensional descriptors and FAISS exact search. CUDA vector indexing and
search are verified on the RTX 5060; image decoding and descriptor extraction run
on CPU. Similarity scores are ranking signals, not confidence percentages, and
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
source paths, timestamps, and timings. Use a fresh output directory for each query.

By default, queries use one midpoint frame per detected shot. The example adds a
frame every 0.5 seconds within each shot. `--search-device cuda` requires GPU
execution; `auto` allows CPU fallback. Interrupted indexing resumes with the same
command; changing the image collection requires `--rebuild` or a new index path.

## Project files

- `image_extraction/`: ingestion, descriptors, indexing, video sampling, and retrieval.
- `tests/`: automated checks; run `python -m unittest discover -s tests -v`.
- `data/`, `artifacts/`, `outputs/`: local datasets, indexes, and reports, ignored by Git.
- [AGENTS.md](AGENTS.md): detailed project context, design, benchmarks, and development guidance preserved from the original README.
