"""Score-first view suppression without a fixed output count or seed exceptions.

Each winner defines a group of directly similar alternatives. No transitive
clustering: a chain of camera movements cannot merge dissimilar endpoints.
Keep separate from legacy decision code so existing reports remain replayable.
"""
import math
import numpy as np

VERSION = "best-distinct-views-v1"


def rank_views(records, similarities, count=None, *, weights, duplicate_similarity=0.85,
               temporal_similarity=0.80, max_gap_seconds=2.0):
    n = len(records)
    ids = [r["frame_number"] for r in records]
    matrix = np.asarray(similarities)
    if len(set(ids)) != n or (count is not None and (not isinstance(count, int) or count <= 0)):
        raise ValueError("Unique frame IDs and an optional positive count are required")
    if matrix.shape != (n, n) or not np.isfinite(matrix).all():
        raise ValueError("Invalid candidate similarity matrix")
    if not math.isfinite(duplicate_similarity) or not -1 <= duplicate_similarity <= 1:
        raise ValueError("Invalid duplicate similarity")
    if not math.isfinite(temporal_similarity) or not -1 <= temporal_similarity <= 1:
        raise ValueError("Invalid temporal similarity")
    if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("Invalid temporal gap")
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Invalid weights")
    scores = [sum(weights[key] * r["components"][key] for key in weights) for r in records]
    if not all(math.isfinite(s) for s in scores):
        raise ValueError("Nonfinite candidate score")
    scores = [round(s, 6) for s in scores]
    groups, current = [], []
    chronological = sorted(range(n), key=lambda i: ids[i])
    for i in chronological:
        if not records[i]["eligible"]:
            current = []
            continue
        compatible = bool(current) and (
            records[i]["shot_id"] == records[current[-1]]["shot_id"]
            and 0 <= records[i]["timestamp_seconds"] - records[current[-1]]["timestamp_seconds"] <= max_gap_seconds
            and all(matrix[i, j] >= temporal_similarity for j in current))
        if not compatible:
            current = []
            groups.append(current)
        current.append(i)
    # Complete-link temporal groups prevent a slow pan from chaining unrelated views.
    representatives = {min(group, key=lambda i: (-scores[i], ids[i])): group for group in groups}
    remaining = set(representatives)
    dispositions = [{"status": "outside_top_k_budget" if r["eligible"] else "below_relevance_threshold"}
                    for r in records]
    selected, trace = [], []
    while remaining and (count is None or len(selected) < count):
        ordered = sorted(remaining, key=lambda i: (-scores[i], ids[i]))
        winner = ordered[0]
        rows = []
        for i in ordered:
            nearest = min(selected, key=lambda j: (-float(matrix[i, j]), ids[j])) if selected else None
            rows.append({"row": i, "frame_number": ids[i], "base_score": scores[i],
                         "nearest_previously_selected": ids[nearest] if nearest is not None else None,
                         "max_similarity": float(matrix[i, nearest]) if nearest is not None else 0.0,
                         "diversity_penalty": 0.0, "utility": scores[i], "status": "considered"})
        suppressed_reps = sorted((i for i in remaining if i == winner or matrix[winner, i] >= duplicate_similarity),
                                 key=lambda i: ids[i])
        members = sorted((i for rep in suppressed_reps for i in representatives[rep]), key=lambda i: ids[i])
        represented_by = {i: rep for rep in suppressed_reps for i in representatives[rep]}
        runner = rows[1] if len(rows) > 1 else None
        trace.append({"step": len(selected) + 1, "phase": "best_view", "feedback_example": None,
                      "selected_before": [ids[i] for i in selected], "winner": rows[0], "runner_up": runner,
                      "utility_margin": round(scores[winner] - runner["utility"], 6) if runner else None,
                      "eligible_competitors": len(rows), "comparison_pool": rows,
                      "group_members": [{"frame_number": ids[i], "score": scores[i],
                                         "similarity_to_winner": float(matrix[winner, i]),
                                         "temporal_representative": ids[represented_by[i]],
                                         "basis": "temporal_group" if represented_by[i] == winner else "global_duplicate"}
                                        for i in members]})
        selected.append(winner)
        for i in members:
            dispositions[i] = ({"status": "selected", "step": len(selected)} if i == winner else
                               {"status": "redundant_in_final_set", "selected_frame": ids[winner],
                                "similarity": float(matrix[winner, i]),
                                "temporal_representative": ids[represented_by[i]],
                                "basis": "temporal_group" if represented_by[i] == winner else "global_duplicate"})
        remaining.difference_update(suppressed_reps)
    return selected, trace, dispositions


def explain_view(step):
    return (f"Highest remaining weighted quality/relevance/layout/feedback score: {step['winner']['base_score']:.6f}. "
            f"Represents {len(step['group_members'])} eligible candidates from its temporal view group and globally suppressed groups. "
            "Each temporal group requires pairwise similarity and continuity; its best-scoring frame competes globally. "
            "No diversity penalty or workbook seed exemption; ties prefer the lower frame number.")
