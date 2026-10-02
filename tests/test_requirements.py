"""Offline checks for the angle specification parser and reference mining."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from image_extraction.requirements import (
    OTHER_ID, assignment_scores, framing_scores, knn_scores, label_table, mine_references, parse_spec, required_slots,
    split_framing,
)

SPEC = """# Exterior

## Door mirrors
- id: `door_mirror`
- description: Exterior door mirror assembly.
- sides:
  - left
  - right
  - views:
  - front
  - rear
- aliases:
  - wing mirror

## Roof lining
- id: `roof_lining`
- aliases:
- headliner

# Identification

## VIN
- id: `vin`
- requirement: Capture all vehicle identification plates.
- framing: The complete plate with readable text.
- avoid: A blurry cropped plate.

## Snorkel
- id: `snorkel`
- optional: true
"""


def parse(text):
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "spec.md"
        path.write_text(text)
        return parse_spec(path)


class ParserTests(unittest.TestCase):
    def test_indentation_slips_are_read_and_reported(self):
        categories, warnings = parse(SPEC)
        mirror, lining, vin, snorkel = categories
        self.assertEqual(mirror["sides"], ["left", "right"])
        self.assertEqual(mirror["views"], ["front", "rear"])
        self.assertEqual(lining["aliases"], ["headliner"])
        self.assertEqual(len(warnings), 2)
        self.assertEqual((mirror["section"], vin["section"]), ("Exterior", "Identification"))
        self.assertTrue(snorkel["optional"])
        self.assertFalse(vin["optional"])

    def test_slots_follow_sides_views_and_capture_all(self):
        categories, _ = parse(SPEC)
        self.assertEqual([c["slots"] for c in categories], [4, 1, None, 1])
        self.assertEqual(required_slots({"sides": [], "views": [], "parts": ["a", "b", "c"]}), 3)

    def test_variant_prompts_are_separate_from_assignment_prompts(self):
        mirror = parse(SPEC)[0][0]
        self.assertIn("a photo of a vehicle's wing mirror.", mirror["prompts"])
        self.assertEqual(len(mirror["variant_prompts"]), 2)
        labels = label_table(parse(SPEC)[0])
        self.assertEqual(labels[-1]["id"], OTHER_ID)
        self.assertEqual(labels[0]["prompts"], mirror["prompts"])

    def test_invalid_specifications_rejected(self):
        bad = [SPEC.replace("`vin`", "`VIN plate`"), SPEC.replace("`snorkel`", "`vin`"),
               SPEC.replace("- optional: true", "- optional: maybe"),
               SPEC.replace("- optional: true", "- colour: red"),
               "## Orphan\n- description: no id\n## Other\n- id: `other`\n"]
        for text in bad:
            with self.assertRaises(ValueError):
                parse(text)

    def test_real_specification_parses(self):
        path = Path(__file__).resolve().parents[1] / "vehicle_angles.md"
        if not path.exists():
            self.skipTest("vehicle_angles.md not present")
        categories, warnings = parse_spec(path)
        ids = {c["id"] for c in categories}
        self.assertTrue({"vehicle_exterior", "vin", "engine_bay", "gear_shifter"} <= ids)
        self.assertEqual(next(c for c in categories if c["id"] == "vehicle_exterior")["slots"], 8)
        self.assertEqual(warnings, [])
        sources = {c["id"]: c["framing_prompts"]["source"] for c in categories}
        self.assertEqual({k for k, v in sources.items() if v == "spec"}, {"vehicle_exterior", "engine_bay", "vin"})

    def test_framing_rules_from_spec_or_defaults(self):
        categories, _ = parse(SPEC)
        rules = {c["id"]: c["framing_prompts"] for c in categories}
        self.assertEqual(rules["vin"], {"framing": ["the complete plate with readable text."],
                                        "avoid": ["a blurry cropped plate."], "source": "spec"})
        self.assertEqual(rules["snorkel"]["source"], "default")
        self.assertIn("entire snorkel", rules["snorkel"]["framing"][0])
        # Framing text never joins the assignment prompts that decide the category.
        self.assertFalse(any("readable" in prompt for prompt in next(c for c in categories if c["id"] == "vin")["prompts"]))


class MiningTests(unittest.TestCase):
    def test_mining_respects_assignment_round_robin_and_duplicates(self):
        vectors = np.eye(6, dtype=np.float32)
        vectors[1] = vectors[0]  # The same photo saved in two stock folders.
        labels = [{"id": "a"}, {"id": "b"}]
        label_scores = np.array([[.9, .1], [.9, .1], [.8, .1], [.7, .1], [.1, .9], [.1, .8]])
        prompt_scores = np.array([[.9, .1, 0], [.9, .1, 0], [.1, .8, 0],
                                  [.5, .5, 0], [0, 0, .9], [0, 0, .8]])
        mined = mine_references(vectors, label_scores, prompt_scores, labels, [0, 0, 1], per_category=3)
        self.assertEqual(mined[0]["rows"], [0, 2, 3])
        self.assertEqual(mined[1]["rows"], [4, 5])
        self.assertEqual(mined[0]["assigned_images"], 4)

    def test_knn_uses_mean_of_top_k_and_marks_empty_categories(self):
        references = np.array([[1, 0], [.8, .6], [0, 1]], dtype=np.float32)
        mined = [{"rows": [0, 1]}, {"rows": [2]}, {"rows": []}]
        scores = knn_scores(np.array([[1, 0]], dtype=np.float32), references, mined, k=2)
        np.testing.assert_allclose(scores, [[.9, 0, -1]])


class AssignmentTests(unittest.TestCase):
    def test_best_variant_can_claim_an_image_but_aliases_are_averaged(self):
        # Prompts: A base x2, B base x1, then A variant. Owner order matches build_requirement_index.
        owner = [0, 0, 1, 0]
        scores = np.array([[.10, .02, .07, .05],    # A base mean .06 < B .07, but A's variant .05 does not help
                           [.03, .03, .07, .09]])   # A's rear-view variant .09 beats B
        result = assignment_scores(scores, owner, base_count=3, variant_end=4, label_count=2)
        np.testing.assert_allclose(result, [[.06, .07], [.09, .07]])


class FramingTests(unittest.TestCase):
    def test_split_takes_quartiles_by_text_contrast_with_stable_ties(self):
        references = np.array([[1, 0], [0, 1], [.6, .8], [.8, .6], [1, 0], [0, 1], [.6, .8], [.8, .6]], dtype=np.float32)
        good, poor = split_framing(references, np.array([[1, 0]]), np.array([[0, 1]]))
        self.assertEqual((good, poor), ([0, 4], [1, 5]))
        self.assertEqual(split_framing(references[:3], np.array([[1, 0]]), np.array([[0, 1]])), ([], []))

    def test_framing_score_is_good_minus_poor_and_neutral_when_unsplit(self):
        references = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
        records = [{"reference_offset": 0, "reference_count": 2, "framing_good": [0], "framing_poor": [1]},
                   {"reference_offset": 2, "reference_count": 1, "framing_good": [], "framing_poor": []}]
        scores = framing_scores(np.array([[1, 0], [0, 1]], dtype=np.float32), references, records, k=1)
        np.testing.assert_allclose(scores, [[1, 0], [-1, 0]])


if __name__ == "__main__":
    unittest.main()
