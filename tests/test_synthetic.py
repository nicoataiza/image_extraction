from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from image_extraction.evaluation import evaluate_synthetic, preference_metrics, rank_candidates
from image_extraction.synthetic import generate_synthetic


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "scenes"

    def generate(self, root=None, seed=41):
        return generate_synthetic(root or self.root, seed=seed, families_per_split=1, size=(96, 64))

    def test_reproducible_pixels_labels_and_seed_variation(self):
        first = self.generate()
        another = self.root.parent / "same"
        second = self.generate(another)
        self.assertEqual(first, second)
        for name in ("images.jsonl", "pairs.jsonl", "preview.html"):
            self.assertEqual((self.root / name).read_bytes(), (another / name).read_bytes())
        for path in (self.root / "images").rglob("*.png"):
            self.assertEqual(path.read_bytes(), (another / path.relative_to(self.root)).read_bytes())
        third = self.generate(self.root.parent / "different", seed=42)
        self.assertNotEqual(first["files"]["images.jsonl"], third["files"]["images.jsonl"])
        self.assertEqual(self.generate(), first)

    def test_layout_split_separation_and_one_factor_negatives(self):
        manifest = self.generate()
        self.assertEqual((manifest["image_count"], manifest["layout_count"], manifest["pair_count"]), (96, 8, 144))
        records = [json.loads(line) for line in (self.root / "images.jsonl").read_text().splitlines()]
        by_id = {r["id"]: r for r in records}
        dev = {json.dumps(r["layout"], sort_keys=True) for r in records if r["split"] == "dev"}
        test = {json.dumps(r["layout"], sort_keys=True) for r in records if r["split"] == "test"}
        self.assertFalse(dev & test)
        for line in (self.root / "pairs.jsonl").read_text().splitlines():
            pair = json.loads(line)
            q, p, n = (by_id[pair[key]] for key in ("query", "positive", "negative"))
            self.assertEqual(q["layout"], p["layout"])
            self.assertNotEqual(q["subject"], p["subject"])
            self.assertEqual((q["subject"], q["appearance"], q["seed"]), (n["subject"], n["appearance"], n["seed"]))
            self.assertEqual({q["split"], p["split"], n["split"]}, {pair["split"]})
            changed = {key for key in q["layout"] if q["layout"][key] != n["layout"][key]}
            expected = {"position": {"center_x"}, "scale": {"width", "height"}, "background": {"horizon"}}
            self.assertEqual(changed, expected[pair["change"]])

    def test_refuses_incompatible_or_unowned_destination(self):
        self.generate()
        with self.assertRaisesRegex(ValueError, "differs"):
            self.generate(seed=2)
        unowned = self.root.parent / "unowned"
        unowned.mkdir()
        (unowned / "keep.txt").write_text("user data")
        with self.assertRaisesRegex(ValueError, "not empty"):
            self.generate(unowned)
        self.assertEqual((unowned / "keep.txt").read_text(), "user data")

    def test_evaluation_reports_verifiable_metrics_and_excludes_self(self):
        self.generate()
        output = self.root.parent / "evaluation"
        with redirect_stderr(io.StringIO()):
            results = evaluate_synthetic(self.root, output, split="all")
        records = {r["id"]: r for r in map(json.loads, (self.root / "images.jsonl").read_text().splitlines())}
        for split, summary in results["splits"].items():
            vectors = np.load(output / f"features-{split}.npy", allow_pickle=False)
            self.assertEqual(vectors.shape, (48, 576))
            self.assertEqual(summary["descriptor_bytes"], vectors.nbytes)
            rankings = [json.loads(line) for line in (output / f"rankings-{split}.jsonl").read_text().splitlines()]
            precisions = []
            for ranking in rankings:
                self.assertNotIn(ranking["query"], {m["id"] for m in ranking["matches"]})
                relevant = 0
                for match in ranking["matches"]:
                    self.assertEqual(records[match["id"]]["split"], split)
                    expected = records[match["id"]]["layout_id"] == records[ranking["query"]]["layout_id"]
                    self.assertEqual(match["relevant"], expected)
                    relevant += expected
                precisions.append(relevant / 10)
            self.assertAlmostEqual(summary["precision_at_10"], float(np.mean(precisions)))
            outcomes = [json.loads(line) for line in (output / f"pairs-{split}.jsonl").read_text().splitlines()]
            self.assertEqual(summary["preference"], preference_metrics([p["margin"] for p in outcomes]))
            self.assertTrue((output / f"report-{split}.html").is_file())

    def test_default_evaluation_only_uses_development_split(self):
        self.generate()
        output = self.root.parent / "dev-only"
        with redirect_stderr(io.StringIO()):
            result = evaluate_synthetic(self.root, output)
        self.assertEqual(set(result["splits"]), {"dev"})
        self.assertFalse((output / "features-test.npy").exists())

    def test_modified_image_or_labels_fail_before_scoring(self):
        self.generate()
        path = next((self.root / "images" / "dev").glob("*.png"))
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Image checksum mismatch"):
            evaluate_synthetic(self.root, self.root.parent / "report")
        with (self.root / "pairs.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "Label checksum mismatch"):
            evaluate_synthetic(self.root, self.root.parent / "report")

    def test_invalid_pair_is_rejected_even_if_checksum_is_updated(self):
        self.generate()
        path = self.root / "pairs.jsonl"
        pairs = [json.loads(line) for line in path.read_text().splitlines()]
        pairs[0]["positive"] = pairs[0]["negative"]
        path.write_text("".join(json.dumps(pair) + "\n" for pair in pairs))
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["pairs.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Invalid same-layout"):
            evaluate_synthetic(self.root, self.root.parent / "report")

    def test_ties_are_reported_separately_and_self_is_never_ranked(self):
        result = preference_metrics([0.1, -0.1, 0, 1e-8])
        self.assertEqual((result["wins"], result["losses"], result["ties"]), (1, 1, 2))
        self.assertEqual(result["accuracy"], 0.25)
        self.assertEqual(result["tie_rate"], 0.5)
        np.testing.assert_array_equal(rank_candidates(np.zeros(12), 3), [0, 1, 2, 4, 5, 6, 7, 8, 9, 10])


if __name__ == "__main__":
    unittest.main()
