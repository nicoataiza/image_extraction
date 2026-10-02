import unittest

import numpy as np

from image_extraction.requirement_decisions import rank_required

CATEGORIES = [{"id": "headlight", "name": "Headlights", "slots": 2, "optional": False},
              {"id": "vin", "name": "VIN", "slots": None, "optional": False},
              {"id": "snorkel", "name": "Snorkel", "slots": 1, "optional": True},
              {"id": "towbar", "name": "Towbar", "slots": 1, "optional": False}]


def records(count, quality=0.5, spacing=5.0):
    return [{"frame_number": i, "shot_id": 1, "timestamp_seconds": i * spacing, "components": {"quality": quality}}
            for i in range(count)]


def scores(rows):
    """Rows of (headlight, vin, snorkel, towbar, background) category scores."""
    return np.array(rows, dtype=float)


def run(rows, category_scores, framing=None, similarity=None, **kwargs):
    n = len(rows)
    framing = np.zeros((n, len(CATEGORIES))) if framing is None else np.asarray(framing, dtype=float)
    similarity = np.eye(n) if similarity is None else np.asarray(similarity, dtype=float)
    settings = {"duplicate_similarity": 0.92, "min_separation_seconds": 3.0, "extra_slot_min_score": 0.5, **kwargs}
    return rank_required(rows, category_scores, framing, similarity, CATEGORIES, **settings)


class RequiredViewTests(unittest.TestCase):
    def test_each_frame_joins_only_its_top_category_and_background_joins_none(self):
        chosen, coverage, _, states = run(records(3), scores([
            [.90, .80, .1, .1, .5], [.80, .91, .1, .1, .5], [.85, .1, .1, .1, .95]]))
        self.assertEqual(chosen, [0, 1])
        self.assertEqual(states[2], {"status": "background"})
        status = {c["category"]: c["status"] for c in coverage}
        self.assertEqual(status, {"headlight": "found", "vin": "found", "snorkel": "optional_not_found",
                                  "towbar": "not_found"})

    def test_framing_beats_slightly_stronger_match_within_a_category(self):
        rows = records(2)
        framing = [[-.05, 0, 0, 0], [.10, 0, 0, 0]]
        similar = np.array([[1, .95], [.95, 1]])
        chosen, coverage, trace, states = run(rows, scores([[.91, 0, 0, 0, 0], [.90, 0, 0, 0, 0]]), framing, similar)
        # Frame 0 leads match by only 0.01 (costs frame 1 0.04); frame 1 leads framing by 0.15 (full 0.35).
        self.assertEqual(trace[0]["pool"][0]["frame_number"], 1)
        self.assertEqual(chosen, [1])
        self.assertEqual(states[0], {"category": "headlight", "status": "duplicate_view", "selected_frame": 1,
                                     "similarity": 0.95})

    def test_extra_slots_need_time_separation_distinct_view_and_slot_budget(self):
        rows = records(5, spacing=1.0)
        rows[3]["timestamp_seconds"], rows[4]["timestamp_seconds"] = 30.0, 40.0
        similarity = np.full((5, 5), .5) + np.eye(5) * .5
        similarity[0, 3] = similarity[3, 0] = .95         # near-identical view, far apart in time
        chosen, coverage, _, states = run(rows, scores([[.9, 0, 0, 0, 0]] * 5), similarity=similarity)
        self.assertEqual(chosen, [0, 4])                  # headlight wants 2 photos
        self.assertEqual(states[1], {"category": "headlight", "status": "too_close_in_time", "selected_frame": 0})
        self.assertEqual(states[3]["status"], "duplicate_view")
        self.assertEqual(coverage[0]["selected_frames"], [0, 4])

    def test_opposite_side_at_0_86_similarity_fills_second_slot(self):
        rows = records(2, spacing=40.0)                   # e.g. left and right door mirrors
        similarity = np.array([[1, .86], [.86, 1]])
        chosen, _, _, _ = run(rows, scores([[.9, 0, 0, 0, 0], [.89, 0, 0, 0, 0]]), similarity=similarity)
        self.assertEqual(chosen, [0, 1])

    def test_weak_frames_do_not_fill_extra_slots_but_capture_all_keeps_them(self):
        rows = records(2)
        weak = scores([[.90, .90, 0, 0, 0], [.82, .82, 0, 0, 0]])   # second frame 0.08 below in match
        framing = [[.1, .1, 0, 0], [-.1, -.1, 0, 0]]
        headlight_rows = [dict(r) for r in rows]
        chosen, _, _, states = run(headlight_rows, weak[:, [0, 2, 2, 3, 4]] * [1, 0, 0, 0, 1], framing)
        self.assertEqual(chosen, [0])
        self.assertEqual(states[1], {"category": "headlight", "status": "below_extra_slot_score"})
        chosen, coverage, _, _ = run(rows, weak[:, [2, 1, 2, 3, 4]] * [0, 1, 0, 0, 1], framing)
        self.assertEqual(chosen, [0, 1])                  # VIN is "capture all": every distinct plate
        self.assertEqual(coverage[1]["selected_frames"], [0, 1])

    def test_capture_all_keeps_every_distinct_view(self):
        rows = records(5)
        chosen, coverage, _, _ = run(rows, scores([[0, .9, 0, 0, 0]] * 5))
        self.assertEqual(chosen, [0, 1, 2, 3, 4])
        self.assertEqual(coverage[1]["selected_frames"], [0, 1, 2, 3, 4])

    def test_possible_category_is_reported_not_exported_and_low_margin_is_flagged(self):
        chosen, coverage, _, _ = run(records(1), scores([[.900, .1, .1, .895, .2]]))
        self.assertEqual(chosen, [0])
        by_id = {c["category"]: c for c in coverage}
        self.assertEqual(by_id["headlight"]["confidence"], "check")
        self.assertEqual(by_id["towbar"]["status"], "possible")
        self.assertEqual(by_id["towbar"]["frame_top_category"], "headlight")

    def test_ties_prefer_lower_frame_and_earlier_category(self):
        rows = records(2)
        rows[0]["frame_number"], rows[1]["frame_number"] = 20, 10
        chosen, _, _, _ = run(rows, scores([[.9, .9, 0, 0, 0], [.9, .9, 0, 0, 0]]), similarity=np.ones((2, 2)))
        self.assertEqual(chosen, [1])

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            run(records(2), scores([[.9, 0, 0, 0]] * 2))
        with self.assertRaises(ValueError):
            run(records(1), scores([[np.nan, 0, 0, 0, 0]]))


if __name__ == "__main__":
    unittest.main()
