"""Deterministic decisions from frozen semantic similarities and score components."""

import math
import numpy as np

DECISION_PRECISION = 6


def rank_candidates(records, similarities, count, *, weights, seed_groups=(),
                    penalty=0.5, duplicate_similarity=0.93):
    """Return selected row indices, complete step traces, and final dispositions.

    All comparisons use six-decimal values and ties prefer the lower frame number.
    Workbook slots are explicit constraints; they do not apply the diversity penalty.
    """
    n = len(records)
    similarities = np.asarray(similarities)
    ids = [r["frame_number"] for r in records]
    if len(set(ids)) != n or count <= 0:
        raise ValueError("Unique frame IDs and a positive selection count are required")
    if similarities.shape != (n, n) or not np.isfinite(similarities).all():
        raise ValueError("Invalid candidate similarity matrix")
    if not math.isfinite(penalty) or penalty < 0 or not -1 <= duplicate_similarity <= 1:
        raise ValueError("Invalid diversity settings")
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Invalid score weights")
    scores = []
    for record in records:
        score = sum(weights[key] * record["components"][key] for key in weights)
        if not math.isfinite(score):
            raise ValueError("Nonfinite candidate score")
        scores.append(round(score, DECISION_PRECISION))
    selected, trace = [], []
    available = {i for i, record in enumerate(records) if record["eligible"]}

    def evaluate(pool, phase, example=None):
        rows = []
        for i in sorted(pool, key=lambda i: ids[i]):
            nearest = (min(selected, key=lambda j: (-float(similarities[i, j]), ids[j]))
                       if selected else None)
            similarity = float(similarities[i, nearest]) if nearest is not None else 0.0
            duplicate = phase == "diversity" and nearest is not None and similarity >= duplicate_similarity
            deduction = round(penalty * similarity, DECISION_PRECISION) if phase == "diversity" else 0.0
            rows.append({"row": i, "frame_number": ids[i], "base_score": scores[i],
                         "nearest_previously_selected": ids[nearest] if nearest is not None else None,
                         "max_similarity": similarity, "diversity_penalty": deduction,
                         "utility": round(scores[i] - deduction, DECISION_PRECISION),
                         "status": "duplicate_blocked" if duplicate else "considered"})
        allowed = sorted((row for row in rows if row["status"] == "considered"),
                         key=lambda row: (-row["utility"], row["frame_number"]))
        if not allowed:
            return None
        winner = allowed[0]
        runner = allowed[1] if len(allowed) > 1 else None
        trace.append({"step": len(selected) + 1, "phase": phase, "feedback_example": example,
                      "selected_before": [ids[i] for i in selected],
                      "winner": dict(winner), "runner_up": dict(runner) if runner else None,
                      "utility_margin": round(winner["utility"] - runner["utility"], DECISION_PRECISION) if runner else None,
                      "eligible_competitors": len(allowed), "comparison_pool": rows})
        return winner["row"]

    for group in seed_groups:
        if len(selected) >= count:
            break
        pool = {i for i in group["rows"] if i in available}
        winner = evaluate(pool, "feedback", group["name"])
        if winner is not None:
            selected.append(winner)
            available.remove(winner)
    while available and len(selected) < count:
        winner = evaluate(available, "diversity")
        if winner is None:
            break
        selected.append(winner)
        available.remove(winner)
    dispositions = []
    for i, record in enumerate(records):
        if i in selected:
            dispositions.append({"status": "selected", "step": selected.index(i) + 1})
        elif not record["eligible"]:
            dispositions.append({"status": "below_relevance_threshold"})
        else:
            neighbors = [j for j in selected if similarities[i, j] >= duplicate_similarity]
            if neighbors:
                j = min(neighbors, key=lambda j: (-float(similarities[i, j]), ids[j]))
                dispositions.append({"status": "redundant_in_final_set", "selected_frame": ids[j],
                                     "similarity": float(similarities[i, j])})
            else:
                dispositions.append({"status": "outside_top_k_budget"})
    return selected, trace, dispositions


def explain_step(step):
    winner, runner = step["winner"], step["runner_up"]
    if step["phase"] == "feedback":
        reason = (f"Workbook slot '{step['feedback_example']}': highest weighted score "
                  f"among {step['eligible_competitors']} eligible exemplar matches. "
                  "This coverage constraint applies no diversity penalty. ")
    else:
        reason = (f"Highest remaining diversity-adjusted score among {step['eligible_competitors']} candidates: "
                  f"{winner['base_score']:.6f} base score - {winner['diversity_penalty']:.6f} "
                  f"redundancy penalty = {winner['utility']:.6f}. ")
    if runner:
        reason += (f"Runner-up: frame {runner['frame_number']} at {runner['utility']:.6f}; "
                   f"margin {step['utility_margin']:.6f}. ")
        if step["utility_margin"] == 0:
            reason += "The tie was resolved by the lower frame number."
    else:
        reason += "No other eligible candidate remained in this comparison pool."
    return reason
