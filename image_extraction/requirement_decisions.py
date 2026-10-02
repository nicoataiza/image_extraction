"""Best photo(s) per required category from frozen category, framing and quality scores.

Stage A (presence): each frame joins only its highest-scoring category; background
(`_other`) frames join none. A category is present when at least one frame joins it.
Stage B (best photo): within a category, frames are ranked by category match and
framing, each scaled against the category's best frame over a fixed width, plus quality.
Fixed widths keep a 0.01 cosine gap small (min-max would stretch it to a full point).
Extra photos (views, sides, parts) must be at least `min_separation_seconds` from the
category's other picks, below `duplicate_similarity` to each, and, for categories with a
fixed photo count, score at least `extra_slot_min_score`. "Capture all" categories keep
every distinct item without that floor. Embedding similarity alone cannot separate a
new angle from the same view a second later (both ~0.8-0.96 with FG-CLIP 2), hence the
time rule. Kept separate from other decision code for exact replay.
"""
import math

import numpy as np

VERSION = "required-category-views-v1"
WEIGHTS = {"match": 0.40, "framing": 0.35, "quality": 0.25}
# Median within-category ranges on VA14022 FG-CLIP 2 candidates: match 0.075, framing 0.058.
WIDTHS = {"match": 0.10, "framing": 0.10}


def _relative(values, width):
    """1 for the category's best frame, falling linearly to 0 at `width` below it."""
    values = np.asarray(values, dtype=np.float64)
    return np.clip(1 - (values.max() - values) / width, 0, 1)


def rank_required(records, category_scores, framing_scores, similarities, categories, *, weights=None,
                  duplicate_similarity, min_separation_seconds=3.0, extra_slot_min_score=0.5,
                  confident_margin=0.01, possible_margin=0.02):
    weights = dict(WEIGHTS if weights is None else weights)
    n, count = len(records), len(categories)
    scores, framing, matrix = (np.asarray(a, dtype=np.float64) for a in (category_scores, framing_scores, similarities))
    ids = [r["frame_number"] for r in records]
    if len(set(ids)) != n or scores.shape != (n, count + 1) or framing.shape != (n, count):
        raise ValueError("Category scores must cover every frame, each category and the background column")
    if matrix.shape != (n, n) or not all(np.isfinite(a).all() for a in (scores, framing, matrix)):
        raise ValueError("Invalid frozen score matrices")
    for value in (duplicate_similarity, min_separation_seconds, extra_slot_min_score, confident_margin, possible_margin):
        if not math.isfinite(value):
            raise ValueError("Thresholds must be finite")
    if min_separation_seconds < 0 or set(weights) != set(WEIGHTS) or any(w < 0 for w in weights.values()):
        raise ValueError("Invalid time separation or weights")
    times = [r["timestamp_seconds"] for r in records]
    ordered = np.sort(scores, axis=1)
    primary = scores.argmax(axis=1)  # Ties prefer the earlier specification category.
    margins = np.round(ordered[:, -1] - ordered[:, -2], 6)
    background = count
    dispositions = [{"status": "background"} if primary[i] == background else None for i in range(n)]
    selected, coverage, trace = [], [], []
    for c, category in enumerate(categories):
        pool = sorted((i for i in range(n) if primary[i] == c), key=lambda i: ids[i])
        entry = {"category": category["id"], "name": category["name"], "optional": category["optional"],
                 "slots": category["slots"], "pool_size": len(pool)}
        if not pool:
            near = [i for i in range(n) if primary[i] != background and scores[i, c] >= scores[i].max() - possible_margin]
            if near:
                best = min(near, key=lambda i: (-scores[i, c], ids[i]))
                entry.update(status="possible", frame_number=ids[best], category_score=float(scores[best, c]),
                             gap_to_frame_top=round(float(scores[best].max() - scores[best, c]), 6),
                             frame_top_category=categories[primary[best]]["id"])
            else:
                best = min(range(n), key=lambda i: (-scores[i, c], ids[i]))
                entry.update(status="optional_not_found" if category["optional"] else "not_found",
                             frame_number=ids[best], category_score=float(scores[best, c]))
            coverage.append(entry)
            continue
        components = {"match": _relative(scores[pool, c], WIDTHS["match"]),
                      "framing": _relative(framing[pool, c], WIDTHS["framing"]),
                      "quality": np.array([records[i]["components"]["quality"] for i in pool])}
        rows = []
        for k, i in enumerate(pool):
            parts = {key: float(components[key][k]) for key in WEIGHTS}
            rows.append({"row": i, "frame_number": ids[i], "category_score": float(scores[i, c]),
                         "framing_score": float(framing[i, c]), "margin": float(margins[i]),
                         "timestamp_seconds": times[i], "components": parts,
                         "score": round(sum(weights[key] * parts[key] for key in WEIGHTS), 6)})
        rows.sort(key=lambda r: (-r["score"], r["frame_number"]))
        picks = []
        for row in rows:
            i = row["row"]
            nearby = min((p for p in picks if abs(times[i] - times[p]) < min_separation_seconds),
                         key=lambda p: (abs(times[i] - times[p]), ids[p]), default=None)
            duplicate = max(picks, key=lambda p: (float(matrix[i, p]), -ids[p])) if picks else None
            if category["slots"] is not None and len(picks) >= category["slots"]:
                row["decision"] = {"status": "outside_category_slots"}
            elif picks and category["slots"] is not None and row["score"] < extra_slot_min_score:
                row["decision"] = {"status": "below_extra_slot_score"}
            elif nearby is not None:
                row["decision"] = {"status": "too_close_in_time", "selected_frame": ids[nearby]}
            elif duplicate is not None and matrix[i, duplicate] >= duplicate_similarity:
                row["decision"] = {"status": "duplicate_view", "selected_frame": ids[duplicate],
                                   "similarity": float(matrix[i, duplicate])}
            else:
                picks.append(i)
                selected.append(i)
                row["decision"] = {"status": "selected", "slot": len(picks), "rank": len(selected)}
            dispositions[i] = {"category": category["id"], **row["decision"]}
        best = rows[0]
        entry.update(status="found", selected_frames=[ids[i] for i in picks],
                     confidence="check" if best["margin"] < confident_margin else "confident",
                     best_margin=best["margin"])
        coverage.append(entry)
        trace.append({"category": category["id"], "pool": rows})
    return selected, coverage, trace, dispositions


def explain_required(category, row, pool_size):
    picked = "best" if row["decision"]["slot"] == 1 else f"distinct view {row['decision']['slot']}"
    return (f"{picked} of {pool_size} frame(s) whose top category is {category}: score {row['score']:.6f} from "
            f"relative match {row['components']['match']:.3f}, framing {row['components']['framing']:.3f} "
            f"and quality {row['components']['quality']:.3f}. Category margin over the frame's runner-up: "
            f"{row['margin']:.4f}.")
