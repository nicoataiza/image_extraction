import unittest
import numpy as np
from image_extraction.view_decisions import rank_views


def records(scores):
    return [{"frame_number": i, "eligible": True, "shot_id": 1, "timestamp_seconds": i * 3.0, "components": {"quality": score}}
            for i, score in enumerate(scores)]


class BestViewTests(unittest.TestCase):
    def test_seventeen_similar_mirror_frames_keep_only_best(self):
        rows = records([0.5] * 17)
        rows[8]["components"]["quality"] = 0.9
        chosen, trace, dispositions = rank_views(rows, np.ones((17, 17)), weights={"quality": 1})
        self.assertEqual(chosen, [8])
        self.assertEqual(len(trace[0]["group_members"]), 17)
        self.assertTrue(all(d.get("selected_frame") == 8 for i, d in enumerate(dispositions) if i != 8))

    def test_uncapped_distinct_views_can_exceed_seventeen(self):
        chosen, _, _ = rank_views(records([1] * 25), np.eye(25), weights={"quality": 1})
        self.assertEqual(chosen, list(range(25)))

    def test_similarity_chain_does_not_merge_distinct_endpoints(self):
        matrix = np.array([[1, .95, .7], [.95, 1, .95], [.7, .95, 1]])
        chosen, _, _ = rank_views(records([1, .8, .9]), matrix, weights={"quality": 1})
        self.assertEqual(chosen, [0, 2])

    def test_score_first_ties_and_optional_cap(self):
        rows = records([.7, .9, .9])
        rows[1]["frame_number"], rows[2]["frame_number"] = 20, 10
        chosen, _, states = rank_views(rows, np.eye(3), 1, weights={"quality": 1})
        self.assertEqual(chosen, [2])
        self.assertEqual(states[1]["status"], "outside_top_k_budget")

    def test_ineligible_frames_do_not_suppress_eligible_views(self):
        rows = records([1, .7]); rows[0]["eligible"] = False
        chosen, _, states = rank_views(rows, np.ones((2, 2)), weights={"quality": 1})
        self.assertEqual(chosen, [1])
        self.assertEqual(states[0]["status"], "below_relevance_threshold")

    def test_invalid_thresholds_and_count(self):
        for threshold in (float("nan"), 1.1, -1.1):
            with self.assertRaises(ValueError):
                rank_views(records([1]), np.eye(1), weights={"quality": 1}, duplicate_similarity=threshold)
        with self.assertRaises(ValueError):
            rank_views(records([1]), np.eye(1), 0, weights={"quality": 1})

    def test_continuous_mirror_view_chooses_best_even_below_global_cutoff(self):
        rows = records([.5] * 17)
        for i, row in enumerate(rows):
            row["timestamp_seconds"] = i
        rows[8]["components"]["quality"] = .9
        matrix = np.full((17, 17), .82); np.fill_diagonal(matrix, 1)
        chosen, trace, _ = rank_views(rows, matrix, weights={"quality": 1})
        self.assertEqual(chosen, [8])
        self.assertEqual(len(trace[0]["group_members"]), 17)

    def test_temporal_group_respects_shots_gaps_and_ineligible_breaks(self):
        for mode in ("shot", "gap", "ineligible"):
            rows = records([.9, .8, .7])
            for i, row in enumerate(rows):
                row["timestamp_seconds"] = i
            if mode == "shot":
                rows[2]["shot_id"] = 2
            elif mode == "gap":
                rows[2]["timestamp_seconds"] = 10
            else:
                rows[1]["eligible"] = False
            matrix = np.full((3, 3), .82); np.fill_diagonal(matrix, 1)
            chosen, _, _ = rank_views(rows, matrix, weights={"quality": 1})
            self.assertEqual(chosen, [0, 2])

    def test_complete_link_prevents_slow_pan_from_chaining(self):
        rows = records([.9, .8, .7])
        for i, row in enumerate(rows):
            row["timestamp_seconds"] = i
        matrix = np.array([[1, .82, .6], [.82, 1, .82], [.6, .82, 1]])
        chosen, _, _ = rank_views(rows, matrix, weights={"quality": 1})
        self.assertEqual(chosen, [0, 2])
