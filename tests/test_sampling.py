"""Local sampling behavior independent of a model or reference collection."""

import unittest

from image_extraction.sampling import shortlist_neighbors


class NeighborhoodTests(unittest.TestCase):
    def setUp(self):
        self.times = [i / 20 for i in range(21)]
        self.shot = {"id": 1, "start_frame": 0, "end_frame": 21,
                     "start_seconds": 0, "end_seconds": 1.05}
        self.metrics = [{"sharpness": 10.0, "exposure": 1.0,
                         "change_from_previous": 0.1} for _ in self.times]

    def run_sample(self, anchors=(10,), **kwargs):
        return shortlist_neighbors(self.shot, self.times, self.metrics, anchors, **kwargs)

    def test_sharp_neighbor_replaces_blurry_anchor_with_inspectable_evidence(self):
        self.metrics[9]["sharpness"] = 1000
        selected, windows = self.run_sample()
        self.assertIn(9, selected)
        rows = {row["frame_number"]: row for row in windows[0]["comparisons"]}
        self.assertIn("quality", rows[9]["retained_as"])
        self.assertGreater(rows[9]["quality_score"], rows[10]["quality_score"])
        self.assertTrue(all(5 <= number <= 15 for number in selected))
        self.assertLessEqual(len(selected), 3)

    def test_stable_alternative_and_temporal_alternative_are_preserved(self):
        self.metrics[10]["change_from_previous"] = 0
        self.metrics[11]["change_from_previous"] = 0
        selected, windows = self.run_sample()
        self.assertIn(10, selected)
        rows = windows[0]["comparisons"]
        stable = next(row for row in rows if "stability" in row["retained_as"])
        self.assertEqual(stable["frame_number"], 10)
        alternative = next(row for row in rows if "temporal_alternative" in row["retained_as"])
        for row in rows:
            if set(row["retained_as"]) & {"quality", "stability"}:
                self.assertGreaterEqual(abs(row["timestamp_seconds"] - alternative["timestamp_seconds"]), 0.1 - 1e-9)

    def test_shot_boundary_never_contributes_motion_and_has_soft_penalty(self):
        self.shot.update(id=2, start_frame=10, start_seconds=0.5)
        self.metrics[10]["change_from_previous"] = 1.0  # preceding cut
        selected, windows = self.run_sample()
        rows = windows[0]["comparisons"]
        self.assertTrue(all(row["frame_number"] >= 10 for row in rows))
        self.assertAlmostEqual(rows[0]["motion_raw"], 0.1)
        self.assertEqual(rows[0]["transition_penalty"], 0.15)
        self.assertTrue(selected)

    def test_single_frame_shot_survives_without_motion_evidence(self):
        self.shot.update(start_frame=10, end_frame=11, start_seconds=0.5, end_seconds=0.55)
        selected, windows = self.run_sample()
        self.assertEqual(list(selected), [10])
        row = windows[0]["comparisons"][0]
        self.assertIsNone(row["motion_raw"])
        self.assertEqual(row["motion_local"], 0.5)

    def test_overlapping_windows_deduplicate_and_ties_are_repeatable(self):
        selected, windows = self.run_sample((10, 11))
        self.assertEqual((selected, windows), self.run_sample((10, 11)))
        self.assertEqual(list(selected), sorted(set(selected)))
        self.assertTrue(any(len(reasons) == 2 for reasons in selected.values()))
        self.assertEqual(windows[0]["comparisons"][0]["retained_as"][0], "quality")

    def test_window_uses_timestamps_not_frame_rate(self):
        self.times = [0.0, 0.01, 0.02, 0.5, 0.51, 1.0]
        self.shot.update(end_frame=6, end_seconds=1.1)
        selected, windows = self.run_sample((3,), radius_seconds=0.1)
        self.assertEqual([row["frame_number"] for row in windows[0]["comparisons"]], [3, 4])
        self.assertTrue(set(selected) <= {3, 4})

    def test_invalid_radius_rejected(self):
        for radius in (0, -1, float("nan"), float("inf")):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                self.run_sample(radius_seconds=radius)

    def test_best_local_keeps_one_stability_winner_per_anchor(self):
        self.metrics[9]["sharpness"] = 1000
        selected, windows = self.run_sample(single_winner=True)
        self.assertEqual(list(selected), [9])
        self.assertEqual(selected[9][0]["roles"], ["best_local"])
        self.assertEqual(sum(bool(row["retained_as"]) for row in windows[0]["comparisons"]), 1)


if __name__ == "__main__":
    unittest.main()
