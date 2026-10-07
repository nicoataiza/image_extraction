import unittest

import numpy as np

from image_extraction.requirement_decisions import rank_required as rank_v1
from image_extraction.requirement_decisions_v2 import rank_required

CATEGORIES = [{"id": "vehicle_exterior", "name": "Entire vehicle", "slots": 8, "optional": False, "oriented": True},
              {"id": "headlight", "name": "Headlights", "slots": 2, "optional": False},
              {"id": "glovebox", "name": "Glovebox", "slots": 1, "optional": False},
              {"id": "door_trim", "name": "Door trims", "slots": 4, "optional": False}]


def records(count, quality=0.5, spacing=5.0):
    return [{"frame_number": i, "shot_id": 1, "timestamp_seconds": i * spacing, "components": {"quality": quality}}
            for i in range(count)]


def run(rows, category_scores, orientation=None, framing=None, similarity=None, categories=CATEGORIES, **kwargs):
    n = len(rows)
    scores = np.asarray(category_scores, dtype=float)
    framing = np.zeros((n, len(categories))) if framing is None else np.asarray(framing, dtype=float)
    orientation = np.zeros((n, len(categories))) if orientation is None else np.asarray(orientation, dtype=float)
    similarity = np.eye(n) if similarity is None else np.asarray(similarity, dtype=float)
    settings = {"duplicate_similarity": 0.92, "min_separation_seconds": 3.0, "extra_slot_min_score": 0.5, **kwargs}
    return rank_required(rows, scores, framing, orientation, similarity, categories, **settings)


def facing(values):
    """Exterior orientation per frame (front > 0 > rear); other categories unoriented."""
    return np.array([[v, 0, 0, 0] for v in values], dtype=float)


class StrictPresenceTests(unittest.TestCase):
    def test_strict_category_needs_margin_and_is_reported_possible_otherwise(self):
        # Glovebox leads door trim by 0.006 (a door trim mistaken for a glovebox), then by 0.03.
        chosen, coverage, _, states = run(records(1), [[.1, .1, .856, .850, .2]])
        self.assertEqual(chosen, [])
        glovebox = coverage[2]
        self.assertEqual((glovebox["status"], glovebox["reason"], glovebox["required_margin"]),
                         ("possible", "below_category_margin", 0.02))
        self.assertEqual(states[0], {"category": "glovebox", "status": "below_category_margin",
                                     "margin": 0.006, "required_margin": 0.02})
        chosen, coverage, _, _ = run(records(1), [[.1, .1, .88, .85, .2]])
        self.assertEqual((chosen, coverage[2]["status"]), ([0], "found"))

    def test_other_categories_keep_top_category_presence(self):
        chosen, coverage, _, _ = run(records(1), [[.1, .1, .850, .856, .2]])   # door trim by 0.006
        self.assertEqual((chosen, coverage[3]["status"]), ([0], "found"))

    def test_strict_frames_below_margin_do_not_block_a_clear_frame(self):
        rows = records(2)
        chosen, coverage, _, _ = run(rows, [[.1, .1, .86, .855, .2], [.1, .1, .87, .80, .2]])
        self.assertEqual((chosen, coverage[2]["selected_frames"]), ([1], [1]))


class OrientationTests(unittest.TestCase):
    def test_front_and_rear_take_first_slots_before_a_stronger_side_view(self):
        rows = records(3, spacing=10.0)
        scores = [[.95, .1, .1, .1, .1], [.90, .1, .1, .1, .1], [.89, .1, .1, .1, .1]]
        chosen, coverage, trace, states = run(rows, scores, orientation=facing([0, .05, -.04]))
        self.assertEqual(chosen, [1, 2, 0])               # front, rear, then the side view
        self.assertEqual(coverage[0]["orientation"], {"front": 1, "rear": 2})
        self.assertEqual(states[1]["view"], "front")
        self.assertEqual(states[2]["view"], "rear")
        self.assertNotIn("view", states[0])                # best frame exempt from the extra-slot floor

    def test_no_clear_front_view_is_reported(self):
        rows = records(2, spacing=10.0)
        _, coverage, _, _ = run(rows, [[.9, .1, .1, .1, .1]] * 2, orientation=facing([.01, -.05]))
        self.assertEqual(coverage[0]["orientation"], {"front": None, "rear": 1})

    def test_whole_car_frame_from_headlight_fills_front_slot_and_leaves_headlight(self):
        rows = records(3, spacing=10.0)
        scores = [[.90, .10, .1, .1, .1],                 # rear-quarter exterior
                  [.88, .89, .1, .1, .1],                 # straight front: headlight first by 0.01
                  [.10, .92, .1, .1, .1]]                 # headlight close-up
        chosen, coverage, trace, states = run(rows, scores, orientation=facing([-.05, .06, .03]))
        self.assertEqual(chosen, [1, 0, 2])
        front = next(r for r in trace[0]["pool"] if r["frame_number"] == 1)
        self.assertEqual((front["primary_category"], front["decision"]["view"]), ("headlight", "front"))
        self.assertEqual(coverage[1]["selected_frames"], [2])
        self.assertEqual(coverage[1]["pool_size"], 1)
        self.assertEqual(states[1]["category"], "vehicle_exterior")

    def test_headlight_without_other_frames_reports_where_its_frame_went(self):
        rows = records(2, spacing=10.0)
        chosen, coverage, _, _ = run(rows, [[.90, .1, .1, .1, .1], [.88, .89, .1, .1, .1]],
                                     orientation=facing([-.05, .06]))
        self.assertEqual(chosen, [1, 0])
        self.assertEqual((coverage[1]["status"], coverage[1]["reason"], coverage[1]["exported_as"]),
                         ("possible", "exported_as_other_category", "vehicle_exterior"))

    def test_borrowed_frames_fill_only_front_and_rear_slots(self):
        rows = records(3, spacing=10.0)
        scores = [[.90, .1, .1, .1, .1], [.88, .89, .1, .1, .1], [.87, .885, .1, .1, .1]]
        chosen, coverage, trace, states = run(rows, scores, orientation=facing([0, .06, .04]))
        # Frame 1 is the better front view; frame 2 is a second front view, so it stays a headlight.
        self.assertEqual(coverage[0]["selected_frames"], [1, 0])
        borrowed = next(r for r in trace[0]["pool"] if r["frame_number"] == 2)
        self.assertEqual(borrowed["decision"], {"status": "not_front_or_rear_pick"})
        self.assertEqual(states[2]["category"], "headlight")
        self.assertEqual(coverage[1]["selected_frames"], [2])

    def test_close_ups_are_not_borrowed_and_do_not_rescale_own_frames(self):
        rows = records(3, spacing=10.0)
        # Frame 2 (a fender close-up ranked as a headlight) matches the vehicle best but is badly framed.
        scores = [[.78, .1, .1, .1, .1], [.77, .1, .1, .1, .1], [.80, .81, .1, .1, .1]]
        framing = [[-.02, 0, 0, 0], [-.03, 0, 0, 0], [-.15, 0, 0, 0]]
        _, coverage, trace, _ = run(rows, scores, orientation=facing([.04, -.04, .05]), framing=framing)
        self.assertEqual(coverage[0]["orientation"], {"front": 0, "rear": 1})
        self.assertEqual(coverage[0]["orientation_candidates"], 0)
        self.assertEqual(trace[0]["pool"][0]["components"]["match"], 1.0)
        self.assertEqual(coverage[1]["selected_frames"], [2])

    def test_frames_beyond_possible_margin_are_not_borrowed(self):
        rows = records(2, spacing=10.0)
        _, coverage, _, _ = run(rows, [[.90, .1, .1, .1, .1], [.85, .89, .1, .1, .1]], orientation=facing([-.05, .06]))
        self.assertEqual(coverage[0]["orientation"], {"front": None, "rear": 0})
        self.assertEqual(coverage[1]["selected_frames"], [1])


class CompatibilityTests(unittest.TestCase):
    def test_matches_v1_without_strict_or_oriented_categories(self):
        rng = np.random.default_rng(7)
        categories = [{k: v for k, v in c.items() if k != "oriented"} for c in CATEGORIES]
        for trial in range(25):
            n = 12
            rows = records(n, spacing=float(rng.uniform(.5, 6)))
            for r in rows:
                r["components"]["quality"] = float(rng.uniform())
            scores = np.round(rng.uniform(.6, .95, (n, 5)), 3)
            framing = np.round(rng.uniform(-.1, .1, (n, 4)), 3)
            similarity = np.round(rng.uniform(.6, 1, (n, n)), 3)
            similarity = np.maximum(similarity, similarity.T)
            np.fill_diagonal(similarity, 1)
            settings = {"duplicate_similarity": 0.92, "min_separation_seconds": 3.0, "extra_slot_min_score": 0.5}
            old = rank_v1(rows, scores, framing, similarity, categories, **settings)
            new = rank_required(rows, scores, framing, np.zeros((n, 4)), similarity, categories,
                                strict_categories=(), **settings)
            self.assertEqual(new[0], old[0])
            self.assertEqual(new[2], old[2])
            self.assertEqual(new[3], old[3])
            strip = lambda entries: [{k: v for k, v in e.items() if k not in ("reason", "frame_margin")} for e in entries]
            self.assertEqual(strip(new[1]), old[1])

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            run(records(1), [[.9, 0, 0, 0, 0]], orientation=np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            run(records(1), [[.9, 0, 0, 0, 0]], strict_margin=-0.01)


if __name__ == "__main__":
    unittest.main()
