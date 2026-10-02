"""Deterministic local candidate shortlists; no semantic or part assumptions."""

from bisect import bisect_left, bisect_right
import math

import numpy as np

SAMPLING_VERSION = "local-quality-motion-v1"
BEST_LOCAL_VERSION = "local-best-frame-v1"


def measure_frame(gray, previous):
    """Reduced-resolution proxies; retain scalars and one previous image only."""
    import cv2

    change = None
    if previous is not None and previous.shape == gray.shape:
        change = float(np.mean(cv2.absdiff(gray, previous)) / 255)
    return {"sharpness": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            "exposure": float(np.mean((gray > 8) & (gray < 247))),
            "change_from_previous": change}


def local_scale(values):
    values = np.asarray(values, dtype=np.float64)
    # Preserve isolated sharp/stable moments in short windows; percentile
    # clipping can flatten exactly the single good frame we want to recover.
    low, high = values.min(), values.max()
    if high - low < 1e-12:
        return np.full(len(values), 0.5)
    return np.clip((values - low) / (high - low), 0, 1)


def shortlist_neighbors(shot, timestamps, metrics, anchors, *, radius_seconds=0.25, single_winner=False):
    """Return chronological unique candidates and complete per-anchor comparisons.

    Quality and stability winners may coincide. A third role prefers quality
    among frames at least 0.1 seconds from both winners. Motion is adjacent-frame
    luminance change, not estimated camera velocity. Cuts never enter that metric.
    """
    if not math.isfinite(radius_seconds) or radius_seconds <= 0:
        raise ValueError("Neighborhood radius must be positive and finite")
    start, end = shot["start_frame"], shot["end_frame"]
    retained, windows = {}, []
    for anchor in anchors:
        center = timestamps[anchor]
        lo = bisect_left(timestamps, center - radius_seconds, lo=start, hi=end)
        hi = bisect_right(timestamps, center + radius_seconds, lo=start, hi=end)
        numbers = list(range(lo, hi))
        sharpness = local_scale([np.log1p(metrics[i]["sharpness"]) for i in numbers])
        motion = []
        for i in numbers:
            changes = []
            if i > start and metrics[i]["change_from_previous"] is not None:
                changes.append(metrics[i]["change_from_previous"])
            if i + 1 < end and metrics[i + 1]["change_from_previous"] is not None:
                changes.append(metrics[i + 1]["change_from_previous"])
            motion.append(float(np.mean(changes)) if changes else None)
        # Missing motion is neutral, never an artificial zero-motion advantage.
        valid_motion = [value for value in motion if value is not None]
        scaled_motion = iter(local_scale(valid_motion)) if valid_motion else iter(())
        rows = []
        for j, number in enumerate(numbers):
            near_cut = ((start > 0 and timestamps[number] - shot["start_seconds"] < 0.1)
                        or (end < len(timestamps) and shot["end_seconds"] - timestamps[number] <= 0.1))
            transition_penalty = 0.15 if near_cut else 0.0
            quality = 0.8 * sharpness[j] + 0.2 * metrics[number]["exposure"] - transition_penalty
            change = float(next(scaled_motion)) if motion[j] is not None else 0.5
            rows.append({"frame_number": number, "timestamp_seconds": timestamps[number],
                         "sharpness_raw": metrics[number]["sharpness"],
                         "sharpness_local": float(sharpness[j]), "exposure": metrics[number]["exposure"],
                         "motion_raw": motion[j], "motion_local": change,
                         "transition_penalty": transition_penalty,
                         "quality_score": round(float(quality), 6),
                         "stability_score": round(float(quality - 0.2 * change), 6),
                         "retained_as": []})

        def best(pool, score):
            return min(pool, key=lambda row: (-row[score], row["frame_number"]))

        quality_winner = best(rows, "quality_score")
        stability_winner = best(rows, "stability_score")
        if single_winner:
            stability_winner["retained_as"].append("best_local")
        else:
            quality_winner["retained_as"].append("quality")
            stability_winner["retained_as"].append("stability")
        separated = [row for row in rows if all(
            abs(row["timestamp_seconds"] - winner["timestamp_seconds"]) >= 0.1 - 1e-9
            for winner in (quality_winner, stability_winner))]
        if separated and not single_winner:
            best(separated, "quality_score")["retained_as"].append("temporal_alternative")
        for row in rows:
            if row["retained_as"]:
                retained.setdefault(row["frame_number"], []).append(
                    {"anchor_frame": anchor, "roles": list(row["retained_as"])})
        windows.append({"shot_id": shot["id"], "anchor_frame": anchor,
                        "anchor_seconds": center, "comparisons": rows})
    return dict(sorted(retained.items())), windows
