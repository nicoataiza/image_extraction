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

### Reduced sampling and best-view selection (October 2 revision)

The three-alternative neighborhood experiment produced too many similar
candidates. Use `best-local` to keep only the highest stability score (quality
minus motion penalty) in each window. A one-second interval with ±0.5-second
windows inspects nearby decoded frames while passing far fewer images to SigLIP:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m image_extraction query \
  --video videos/NEW_VIDEO.mp4 --index artifacts/wcp-all-semantic \
  --output outputs/NEW_VIDEO/best-local \
  --interval-seconds 1 --sampling-mode best-local --neighborhood-seconds 0.5 \
  --extraction-device cuda --search-device cuda

OMP_NUM_THREADS=4 .venv/bin/python -m image_extraction select-frames \
  --results outputs/NEW_VIDEO/best-local/results.json \
  --output outputs/NEW_VIDEO/best-view --selection-mode best-view \
  --duplicate-similarity 0.85 --extraction-device cuda --search-device cuda
```

Query defaults remain uniform for compatibility; the new sampling mode is
explicit. Final selection now defaults to best-view with no fixed output count.
Component weights are unchanged. Best-view defaults to a stricter relevance floor
of 0.85 (legacy: 0.65) because removing the count cap otherwise admits weak
transition frames. Both cutoffs remain provisional heuristics, not calibrated probabilities. No hand-picked
frames are used. `view_decisions.py` implements the new decisions separately from
`selection_decisions.py`, preserving the old replay hash contract. Reports include
each winner's suppressed alternatives and scores; replay verifies those groups.
Without `--interval-seconds`, best-local still searches only shot midpoints.
`local-best-frame-v1` identifies the one-winner sampler; `best-distinct-views-v1`
identifies the new selection rule. Do not claim these heuristics recognize parts.

The VA14022 revision produced **189 candidates → 24 photos**, compared with
**833 → 17** for the earlier neighborhood experiment. There is no fixed target.
See `outputs/VA14022/best-view-comparison/report.html` for all three output sets,
and `outputs/VA14022/wcp-selected-best-view-v2/report.html` for each representative
and its suppressed alternatives. Exact replay, checksums and ZIP checks passed.
Zero selected pairs exceed cosine 0.85, versus six in the earlier neighborhood
set; this numerical rule does not prove visual uniqueness or part coverage.
Both new stages used CUDA. Thresholds were explored on VA14022 and still need
independent evaluation. Front/side overviews and distinct physical parts are not
reserved or guaranteed by the new score-first rule.

### Neighborhood candidate sampling

Use `--sampling-mode neighborhood` to search around the midpoint/interval anchors
before semantic retrieval. Existing commands default to `uniform` sampling for
baseline comparisons. For the active semantic index:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m image_extraction query \
  --video videos/VA14022.mp4 --index artifacts/wcp-all-semantic \
  --output outputs/VA14022/wcp-semantic-neighborhood \
  --extraction-device cuda --search-device cuda \
  --interval-seconds 0.5 --sampling-mode neighborhood --neighborhood-seconds 0.25
```

This inspects every frame within ±0.25 seconds of each anchor, clipped to its
shot, and retains up to three distinct alternatives: a quality winner, a stability
winner if different, and the best-quality frame at least 0.1 seconds from both.
Overlapping windows share candidates by frame number. Very short shots still
retain a candidate; the exact anchor need not survive. Omitting the interval still
searches only around each shot midpoint, so use an interval for continuous pans.

The existing scan collects grayscale Laplacian variance, usable exposure fraction,
and adjacent-frame luminance change at a maximum side of 320 pixels. It retains
scalar measurements and one preceding grayscale image, not the video's pixels.
Log sharpness and motion are min/max normalized within each window (constant
values map to 0.5). Quality is `0.8 * sharpness + 0.2 * exposure`; stability subtracts
`0.2 * motion`. Both subtract 0.15 within 0.1 seconds of internal shot boundaries.
Motion averages available neighbors within the same shot, never across a cut;
missing motion is neutral. Six-decimal scores and lower frame numbers resolve
ties. These new sampling heuristics do not change the final selection weights.

Motion here includes lighting/subject changes; it is not camera velocity. The
shortlist does not identify parts or establish object completeness, and it can
still miss useful angles. Semantic adaptive sampling is not implemented. Candidate
encoding and final selection may cost more because several alternatives replace
each anchor; selection currently accepts at most 5,000 candidates.

`sampling.json` records every window's competitors, raw measurements, scores and
retention roles. `results.json` records its checksum and each candidate's originating
anchors. The HTML report links the evidence. `select-frames` consumes these results
normally, preserves the sampling evidence, and passes all retained alternatives to
the existing reference/quality/diversity scorer. Replay preserves and verifies the
evidence; it replays final selection, not video decoding or sampling measurements.
Native-resolution JPEG export is unchanged. Use fresh output directories for each
experiment and compare the photos visually before treating this as an improvement
on real videos. Keep the VA14022 feedback profile scoped to VA14022.

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

## Select relevant, distinct video frames

`select-frames` ranks an existing semantic query's sampled frames against the
same reference index. The default `--selection-mode best-view` has no photo-count
cap: it exports the best-scoring representative from each directly similar group.
Use `--top-k` only for an explicit maximum; `--selection-mode legacy` preserves
the earlier seeded diversity selection (default maximum 17). Example:

```bash
.venv/bin/python -m image_extraction select-frames \
  --results outputs/VA14022/wcp-semantic/results.json \
  --output outputs/VA14022/selection-new \
  --extraction-device cuda --search-device cuda \
  --feedback-profile outputs/VA14022/feedback-profile.json
```

This is an **automatic proposal**, followed by visual review when composition
matters. It does not yet detect object boundaries, guarantee complete parts, or
measure plate-text readability. No labels are required on the collection.

The selector re-encodes candidate frames with the saved SigLIP settings and
retrieves 64 reference neighbors. It keeps up to three distinct reference
embeddings, collapsing similarities of at least 0.9995 so duplicate photos under
different stock folders do not inflate reference support. Relevance is the mean
cosine similarity of these distinct matches. The relevance floor defaults to 0.85 in best-view mode and 0.65 in legacy mode;
`--min-relevance` overrides it. This is an uncalibrated, adjustable cutoff, not a probability or proof of part absence.

Ranking combines reference relevance (35%), thumbnail sharpness/exposure proxies
(20%), spatial layout similarity to the reference photos (20%), and optional
workbook-example guidance (25%). The workbook term compares preferred and
less-preferred examples when a pair is supplied. It is neutral without feedback.
Only candidates within 0.05 of an exemplar's strongest similarity, and at least
0.78 similar, receive that exemplar's guidance. These are heuristics selected
while developing on VA14022, not independently validated thresholds. An unlabeled
reference photo's layout is not assumed to be a good composition.

Best-view first groups consecutive eligible samples within each shot. Every pair
in a group must have cosine >=0.80, and consecutive timestamps must be at most
2 seconds apart (`--temporal-similarity`, `--temporal-gap-seconds`). Ineligible
frames break groups. Each group contributes its highest weighted score. Selection
then takes the best remaining representative and suppresses representatives with cosine at least
`--duplicate-similarity` (default 0.85) to that winner. It continues until all
eligible candidates are represented, or an explicit `--top-k` cap is reached.
Feedback affects the score but never reserves slots or bypasses duplicate checks.
There is no diversity penalty. Temporal groups require pairwise similarity; global suppression compares their
representatives. This prevents unconstrained transitive chains through a pan.
These are similarity groups, not detected object instances: opposite-side mirrors
or different close-ups can still be confused. Lower thresholds suppress more
views and can lose useful differences.

A feedback profile explicitly maps workbook media to visual preferences; document
text is not interpreted as executable instructions. For example:

```json
{
  "workbook": "../../Nic images feedback .xlsx",
  "examples": [
    {
      "name": "Rear overview",
      "criterion": "Include the bumper and ground beneath it.",
      "preferred": "xl/media/image2.png",
      "less_preferred": "xl/media/image1.png"
    }
  ]
}
```

The workbook path is relative to the profile. Omit `less_preferred` for a positive
example such as a plate. The local VA14022 profile maps all five feedback topics.
The workbook examples are references for this video, not labeled correct frames.

For an explicitly reviewed selection, add `--review PATH.json`. The JSON must
identify the exact input `results.json` using `results_sha256` and contain
`selected` entries with `frame_number`, `view`, and `reason`; optional `notes`
appear in the report. These decisions override the automatic proposal openly.
Unknown/duplicate frame numbers and stale input checksums are rejected. Frame
labels in a review describe that selection only; they do not label the collection.

Each new output directory contains:

- `report.html`: selected photos, timestamps, reasons, supporting reference photos,
  workbook examples, and all candidate scores.
- `selected-frames.zip`: downloadable archive of the selected photos.
- `selected/`: exact copies of the native-resolution query JPEGs, without cropping,
  enhancement, or another JPEG encoding.
- `selection.json`: automatic proposal, final selections, component scores, review
  decisions, source checksums, model settings, and limitations.
- `frame-embeddings.npy` and `frame-layouts.npy`: vectors in `candidates` list order
  from `selection.json`, retained for inspection and later experiments.
- `references/` and `feedback/`: report-sized reference images and workbook examples.

The reviewed VA14022 result is `outputs/VA14022/wcp-selected/report.html`; its
explicit review is `outputs/VA14022/selection-review.json`. The first automatic
proposal is retained in `wcp-selected-auto/` for development comparison. The final
report also records the automatic proposal from the current settings.

This reviewed set includes both plates and the illuminated instrument cluster.
The rear views use angles with better bumper margins. The chosen engine detail
still does not satisfy the workbook's wider-bay preference; a wider capture is
needed. Visual review remains part of this result, not an automated quality claim.

### Legacy automatic selection and exact replay

The following describes the historical `--selection-mode legacy` behavior.
Use the automatic path **without `--review`** when the final selection must come
entirely from code. The current automatic VA14022 report is
`outputs/VA14022/wcp-selected-repeatable/report.html`. It is a different result
from the older assistant-reviewed `wcp-selected/` output.

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m image_extraction select-frames \
  --results outputs/VA14022/wcp-semantic/results.json \
  --output outputs/VA14022/automatic-new \
  --selection-mode legacy --top-k 17 --extraction-device cuda --search-device cuda \
  --feedback-profile outputs/VA14022/feedback-profile.json
```

No frame IDs are supplied. Workbook media mappings describe visual preferences,
not correct video frames. SigLIP reference matches establish relevance; fixed,
recorded quality/layout/preference rules rank the candidates. The pure decision
implementation is `image_extraction/selection_decisions.py`.

Every selected frame has a generated explanation and a runner-up comparison:

1. A workbook slot takes the highest base score among its eligible semantic
   exemplar matches. This is an explicit coverage constraint and applies **no**
   diversity penalty; the report says so.
2. Remaining slots maximize `base_score - 0.5 * maximum_similarity_to_selected`.
   Candidates at similarity 0.93 or above are blocked during this stage.
3. Utilities are rounded to six decimals; ties choose the lower frame number.

`selection.json` retains all comparison pools, score components, weighted
contributions, normalization parameters, settings, code hashes and artifact
checksums. `candidate-scores.csv` lists all 382 candidates and their final
selection status. `decision-trace.csv` lists every candidate considered or blocked
at each selection step, including the penalty and winning frame. The HTML report
shows the runner-up image and weighted component differences.

Exclusion statuses distinguish below-threshold relevance, redundancy in the
final set, and running out of the K-photo budget. The latter two are final-set
descriptions; the decision trace supplies the actual history of each comparison.
A low similarity score does not establish that a physical part is absent.

Replay the exact decisions without loading a model, FAISS, the video, or the
original image collection:

```bash
.venv/bin/python -m image_extraction replay-selection \
  --selection outputs/VA14022/wcp-selected-repeatable/selection.json \
  --output outputs/VA14022/replay-new
```

Replay verifies checksums, uses the frozen `frame-similarities.npy` and component
scores, recomputes every decision, and checks that all chosen IDs, runner-ups,
comparison pools and exclusion reasons match. It also verifies the decision-code
hash and rejects manual-review outputs. Keep the complete selection directory:
its exported frames, reference/candidate thumbnails and workbook examples make
the replay report self-contained. Fresh inference can vary with software/hardware
and near ties; the exact reproducibility guarantee applies to the frozen decision
inputs. This audit explains the implemented policy, not an objectively proven
composition optimum; the weights and thresholds still need independent evaluation.

Local verification: two independent VA14022 automatic runs produced identical
17-frame selections and all 4,508 recorded comparisons; the frozen replay matched
as well. Evidence is in `outputs/VA14022/selection-repeatability.json`. All 70 tests
passed, including CUDA-required checks and offline replay/tamper tests.

### Semantic image labelling

Stock CSV part types describe the requested stock item. The WCP downloader saves
`VehicleImages`, so assigning that stock's part type to every photograph does not
establish what each photo shows. `labels.csv` is stock metadata only.

Generate image-content suggestions by comparing the existing pinned SigLIP 2
image vectors with text descriptions. The stock CSV supplies the vocabulary of
part names; stock-to-part associations do not participate in scoring.

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m image_extraction semantic-label \
  --index artifacts/wcp-all-semantic \
  --part-types to_download_stock_numbers.csv \
  --output outputs/wcp-semantic-labels --device cuda
```

Use a fresh output directory for each run. Reports must remain outside the image
collection because their thumbnails would otherwise become dataset inputs. The
command verifies the index checksums and source paths/sizes/mtimes, then reuses
stored vectors without repeating image inference. It loads the text tower and
tokenizer of the same pinned model; the first run may download tokenizer assets.
`--device auto|cpu|cuda` selects text encoding; batched cosine scoring uses CPU.
Explicit CUDA requests fail if execution is unavailable; auto records fallback.

Outputs include:

- `semantic-labels.csv`: relative image paths, source folder stock numbers,
  proposed broad view labels, top three view/part candidates, cosine scores,
  runner-up margins, review priority and blank manual-review fields.
- `report.html`: thumbnails of up to three strongest and three ambiguous matches
  per winning view category, suppressing repeated filenames within a category.
- `taxonomy.json`, `prompt-embeddings.npy`, `label-scores.npy`,
  `image-paths.jsonl`, `summary.json`, and `status.json`: frozen vocabulary,
  scoring inputs/results, row mapping, provenance, checksums and completion state.

Prompts are lowercase and padded to 64 tokens, following the
[SigLIP 2 model documentation](https://huggingface.co/docs/transformers/model_doc/siglip2).
Each label score is the mean cosine across its two text prompts. View and part
candidates are ranked separately, with six-decimal scores and taxonomy order for
ties. `semantic_label` is `uncertain` when the best view cosine is below
`--min-score` (default 0.1), its runner-up margin is below `--min-margin` (default
0.01), or the `other_unclear` category wins. These cutoffs are provisional review
heuristics, not calibrated confidence. Even other labels remain `unreviewed`.

Part candidates are suggestions only: whole-car images can show multiple parts;
left/right distinctions, hidden parts and exact stock identities are not verified.
The method does not detect part bounding boxes, infer car/part numbers, establish
absence, or validate fine part labels. Independent human review is required before
using predictions as training ground truth or claiming annotation accuracy.

## Required photos per vehicle part (`vehicle_angles.md`)

`vehicle_angles.md` lists the photo categories wanted from each video (exterior
angles, lights, wheels, VIN plates, engine bay parts, interior controls, ...), with
photo counts from their sides/views and optional `framing:`/`avoid:` composition rules.
The collection is unlabeled, so `index-requirements` uses the index's own text encoder
to mine reference photos per category. It splits each category's references into best-
and worst-framed groups by its framing rule. This takes seconds and re-encodes nothing.

The tested encoder for this is [FG-CLIP 2 base](https://huggingface.co/qihoo360/fg-clip2-base)
(Apache-2.0, revision `430fbc8`), a fine-grained SigLIP 2-style model. Its reviewed model
code is vendored in `image_extraction/vendor/fgclip2/`, so remote code stays disabled.

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m image_extraction index \
  --images data/wcp-all-downloaded --index artifacts/wcp-all-fgclip2 \
  --descriptor fgclip2 --extraction-device cuda --search-device cuda --extraction-batch-size 32
OMP_NUM_THREADS=4 .venv/bin/python -m image_extraction index-requirements \
  --index artifacts/wcp-all-fgclip2 --spec vehicle_angles.md \
  --output artifacts/wcp-all-fgclip2-requirements-v2 --device cuda
OMP_NUM_THREADS=4 .venv/bin/python -m image_extraction query \
  --video videos/NEW_VIDEO.mp4 --index artifacts/wcp-all-fgclip2 \
  --output outputs/NEW_VIDEO/fgclip2-semantic-best-local --extraction-device cuda --search-device cuda \
  --interval-seconds 1 --sampling-mode best-local --neighborhood-seconds 0.5 --top-k 10
OMP_NUM_THREADS=4 .venv/bin/python -m image_extraction select-frames \
  --results outputs/NEW_VIDEO/fgclip2-semantic-best-local/results.json \
  --output outputs/NEW_VIDEO/fgclip2-selected-required \
  --selection-mode required-views --requirements artifacts/wcp-all-fgclip2-requirements-v2 \
  --extraction-device cuda --search-device cuda
```

`required-views` puts each frame in its single best-matching category (or none).
Within a category it ranks frames by 0.40 part match, 0.35 framing and 0.25
sharpness/exposure. Extra photos (other angles or sides) must be at least 3 s apart,
not near-duplicates, and reasonably strong. The report lists every category as found,
possible (reported, not exported) or not found, and flags close calls with "check".
Exports are named by category, and `replay-selection` reproduces the decisions offline.
"Not found" does not prove a part is absent; left/right sides, object completeness and
plate readability are not verified. See [AGENTS.md](AGENTS.md) for thresholds, evidence
and current results.

