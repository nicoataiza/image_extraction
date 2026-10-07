"""Required-category decisions v2: front/rear whole-vehicle photos and stricter presence.

Same stages as v1 (`requirement_decisions.py`, kept unchanged so v1 selections replay):
each frame joins only its top category; within a category frames are ranked by relative
match, framing and quality; extra photos need time separation, a distinct view and,
for fixed photo counts, a minimum score. Two changes:

1. Stricter presence for unreliable categories. A frame counts toward a category in
   `STRICT_CATEGORIES` only if its score beats its runner-up category by `strict_margin`.
   On the five FG-CLIP 2 videos (VA14022/32/41/52/53, agent visual checks) these seven
   categories gave 18 of 32 wrong exports and 2 correct ones; a 0.02 margin removed 17
   wrong and 1 correct. A floor on every category removed more correct photos than wrong
   ones, so other categories are unchanged. The list is tuned on those videos.
2. Front and rear views. A category with front/rear reference groups (the whole vehicle)
   fills its first slots with its best front-facing frame (orientation >= margin) and
   best rear-facing frame (<= -margin). Whole-car frames that rank another category first
   (headlight, taillight, towbar) may fill only these two slots, when this category is
   within `possible_margin` of their top score and their framing is within the framing
   width of the category's best own frame (whole-car views, not fender close-ups). Such a
   frame is then exported only here; its own category uses its remaining frames. Scores
   stay relative to the category's own frames, so borrowed frames never rescale them.
   Left/right is still not verified.
"""
import math

import numpy as np

VERSION = "required-category-views-v2"
WEIGHTS = {"match": 0.40, "framing": 0.35, "quality": 0.25}
WIDTHS = {"match": 0.10, "framing": 0.10}
STRICT_CATEGORIES = ("roof_accessories", "snorkel", "fuel_filter_housing", "engine_cold_side",
                     "pedals", "glovebox", "rear_heater_controls")
VIEWS = (("front", 1), ("rear", -1))


def _relative(values, width, top):
    """1 at the category's best own frame (`top`), falling linearly to 0 at `width` below it."""
    return np.clip(1 - (top - np.asarray(values, dtype=np.float64)) / width, 0, 1)


def rank_required(records, category_scores, framing_scores, orientation_scores, similarities, categories, *,
                  weights=None, duplicate_similarity, min_separation_seconds=3.0, extra_slot_min_score=0.5,
                  confident_margin=0.01, possible_margin=0.02, strict_margin=0.02, orientation_margin=0.02,
                  strict_categories=STRICT_CATEGORIES):
    weights = dict(WEIGHTS if weights is None else weights)
    n, count = len(records), len(categories)
    scores, framing, orientation, matrix = (np.asarray(a, dtype=np.float64) for a in (
        category_scores, framing_scores, orientation_scores, similarities))
    ids = [r["frame_number"] for r in records]
    if len(set(ids)) != n or scores.shape != (n, count + 1) or framing.shape != (n, count) or orientation.shape != (n, count):
        raise ValueError("Category scores must cover every frame, each category and the background column")
    if matrix.shape != (n, n) or not all(np.isfinite(a).all() for a in (scores, framing, orientation, matrix)):
        raise ValueError("Invalid frozen score matrices")
    thresholds = (duplicate_similarity, min_separation_seconds, extra_slot_min_score, confident_margin,
                  possible_margin, strict_margin, orientation_margin)
    if not all(math.isfinite(value) for value in thresholds):
        raise ValueError("Thresholds must be finite")
    if min(min_separation_seconds, strict_margin, orientation_margin) < 0:
        raise ValueError("Time separation and margins must be non-negative")
    if set(weights) != set(WEIGHTS) or any(w < 0 for w in weights.values()):
        raise ValueError("Invalid weights")
    times = [r["timestamp_seconds"] for r in records]
    ordered = np.sort(scores, axis=1)
    primary = scores.argmax(axis=1)  # Ties prefer the earlier specification category.
    margins = np.round(ordered[:, -1] - ordered[:, -2], 6)
    background = count
    floors = [strict_margin if category["id"] in strict_categories else 0.0 for category in categories]
    dispositions = [{"status": "background"} if primary[i] == background else None for i in range(n)]
    selected, coverage, trace = [], [], []
    claimed = {}  # frame row -> category that exported it as a front/rear view
    for c, category in enumerate(categories):
        members = [i for i in range(n) if primary[i] == c and i not in claimed]
        pool = sorted((i for i in members if margins[i] >= floors[c]), key=lambda i: ids[i])
        for i in members:
            if margins[i] < floors[c]:
                dispositions[i] = {"category": category["id"], "status": "below_category_margin",
                                   "margin": float(margins[i]), "required_margin": floors[c]}
        entry = {"category": category["id"], "name": category["name"], "optional": category["optional"],
                 "slots": category["slots"], "pool_size": len(pool)}
        if floors[c]:
            entry["required_margin"] = floors[c]
        if not pool:
            near = [i for i in range(n) if primary[i] != background and scores[i, c] >= scores[i].max() - possible_margin]
            if near:
                best = min(near, key=lambda i: (-scores[i, c], ids[i]))
                reason = ("exported_as_other_category" if best in claimed else
                          "below_category_margin" if primary[best] == c else "near_top_category")
                entry.update(status="possible", reason=reason, frame_number=ids[best],
                             category_score=float(scores[best, c]),
                             gap_to_frame_top=round(float(scores[best].max() - scores[best, c]), 6),
                             frame_top_category=categories[primary[best]]["id"], frame_margin=float(margins[best]))
                if best in claimed:
                    entry["exported_as"] = claimed[best]
            else:
                best = min(range(n), key=lambda i: (-scores[i, c], ids[i]))
                entry.update(status="optional_not_found" if category["optional"] else "not_found",
                             frame_number=ids[best], category_score=float(scores[best, c]))
            coverage.append(entry)
            continue
        oriented = bool(category.get("oriented"))
        borrowed = []
        tops = {"match": scores[pool, c].max(), "framing": framing[pool, c].max()}
        if oriented:
            taken = set(selected) | set(claimed)
            borrowed = sorted((i for i in range(n) if primary[i] not in (c, background) and i not in taken
                               and scores[i, c] >= scores[i].max() - possible_margin
                               and abs(orientation[i, c]) >= orientation_margin
                               and framing[i, c] > tops["framing"] - WIDTHS["framing"]), key=lambda i: ids[i])
        candidates = pool + borrowed
        components = {"match": _relative(scores[candidates, c], WIDTHS["match"], tops["match"]),
                      "framing": _relative(framing[candidates, c], WIDTHS["framing"], tops["framing"]),
                      "quality": np.array([records[i]["components"]["quality"] for i in candidates])}
        rows = []
        for k, i in enumerate(candidates):
            parts = {key: float(components[key][k]) for key in WEIGHTS}
            row = {"row": i, "frame_number": ids[i], "category_score": float(scores[i, c]),
                   "framing_score": float(framing[i, c]), "margin": float(margins[i]),
                   "timestamp_seconds": times[i], "components": parts,
                   "score": round(sum(weights[key] * parts[key] for key in WEIGHTS), 6)}
            if oriented:
                row["orientation_score"] = float(orientation[i, c])
            if k >= len(pool):
                row["primary_category"] = categories[primary[i]]["id"]
            rows.append(row)
        rows.sort(key=lambda r: (-r["score"], r["frame_number"]))
        picks, views = [], {view: None for view, _ in VIEWS}

        def pick(row, view=None):
            i = row["row"]
            picks.append(i)
            selected.append(i)
            row["decision"] = {"status": "selected", "slot": len(picks), "rank": len(selected)}
            if view:
                row["decision"]["view"] = view
                if "primary_category" in row:
                    claimed[i] = category["id"]
            dispositions[i] = {"category": category["id"], **row["decision"]}

        if oriented:
            for view, sign in VIEWS:
                if category["slots"] is not None and len(picks) >= category["slots"]:
                    break
                row = next((r for r in rows if "decision" not in r
                            and sign * orientation[r["row"], c] >= orientation_margin), None)
                views[view] = None if row is None else row["frame_number"]
                if row is not None:
                    pick(row, view)
        for row in rows:
            if "decision" in row:
                continue
            i = row["row"]
            if "primary_category" in row:
                # Borrowed whole-car frames fill only the front/rear slots; their own category decides the rest.
                row["decision"] = {"status": "not_front_or_rear_pick"}
                continue
            nearby = min((p for p in picks if abs(times[i] - times[p]) < min_separation_seconds),
                         key=lambda p: (abs(times[i] - times[p]), ids[p]), default=None)
            duplicate = max(picks, key=lambda p: (float(matrix[i, p]), -ids[p])) if picks else None
            if category["slots"] is not None and len(picks) >= category["slots"]:
                row["decision"] = {"status": "outside_category_slots"}
            elif picks and row is not rows[0] and category["slots"] is not None and row["score"] < extra_slot_min_score:
                row["decision"] = {"status": "below_extra_slot_score"}
            elif nearby is not None:
                row["decision"] = {"status": "too_close_in_time", "selected_frame": ids[nearby]}
            elif duplicate is not None and matrix[i, duplicate] >= duplicate_similarity:
                row["decision"] = {"status": "duplicate_view", "selected_frame": ids[duplicate],
                                   "similarity": float(matrix[i, duplicate])}
            else:
                pick(row)
                continue
            dispositions[i] = {"category": category["id"], **row["decision"]}
        best = next(r for r in rows if "primary_category" not in r)
        entry.update(status="found", selected_frames=[ids[i] for i in picks],
                     confidence="check" if best["margin"] < confident_margin else "confident",
                     best_margin=best["margin"])
        if oriented:
            entry.update(orientation=views, orientation_candidates=len(borrowed))
        coverage.append(entry)
        trace.append({"category": category["id"], "pool": rows})
    return selected, coverage, trace, dispositions


def explain_required(category, row, pool_size):
    view = row["decision"].get("view")
    if view:
        picked = f"best {view}-facing view (orientation {row['orientation_score']:+.3f})"
    else:
        picked = "best" if row["decision"]["slot"] == 1 else f"distinct view {row['decision']['slot']}"
    borrowed = (f" It ranked {row['primary_category']} first but is within the possible margin of {category}."
                if "primary_category" in row else "")
    return (f"{picked} of {pool_size} frame(s) whose top category is {category}: score {row['score']:.6f} from "
            f"relative match {row['components']['match']:.3f}, framing {row['components']['framing']:.3f} "
            f"and quality {row['components']['quality']:.3f}. Category margin over the frame's runner-up: "
            f"{row['margin']:.4f}.{borrowed}")
