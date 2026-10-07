"""Selection safeguards, diversity, reference deduplication and export provenance."""

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image

from image_extraction.indexing import file_hash
from image_extraction.selection_decisions import rank_candidates
from image_extraction.selection import replay_selection
from image_extraction.selection import (choose_diverse, diverse_references, image_quality,
                                        select_video_frames, validate_review, _load_feedback, assign_feedback)


class SelectionTests(unittest.TestCase):
    def test_duplicate_reference_photos_do_not_count_as_independent_support(self):
        vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        matches = diverse_references([0, 1, 2], [0.99, 0.99, 0.8], lambda i: vectors[i])
        self.assertEqual(matches, [(0, 0.99), (2, 0.8)])

    def test_diversity_keeps_different_views_and_drops_repeats(self):
        vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        result = choose_diverse(np.array([0.9, 0.8, 0.7]), vectors, [True, True, True], 3)
        self.assertEqual(result, [0, 2])
        self.assertEqual(choose_diverse(np.ones(3), vectors, [False] * 3, 17), [])
        self.assertEqual(choose_diverse(np.ones(3), vectors, [False, True, True], 1, seeds=[0, 2]), [2])

    def test_decision_trace_explains_diversity_winner_and_duplicate_rejection(self):
        records = [{"frame_number": i, "eligible": True, "components": {"semantic": score}}
                   for i, score in enumerate([0.9, 0.85, 0.8, 0.76])]
        matrix = np.array([[1, .98, .85, .3], [.98, 1, .8, .3], [.85, .8, 1, .2], [.3, .3, .2, 1]])
        chosen, trace, dispositions = rank_candidates(
            records, matrix, 2, weights={"semantic": 1}, seed_groups=[{"name": "Example", "rows": [0]}])
        self.assertEqual(chosen, [0, 3])
        self.assertEqual(trace[1]["runner_up"]["frame_number"], 2)
        self.assertAlmostEqual(trace[1]["utility_margin"], .235)
        blocked = next(r for r in trace[1]["comparison_pool"] if r["frame_number"] == 1)
        self.assertEqual(blocked["status"], "duplicate_blocked")
        self.assertEqual(dispositions[1]["status"], "redundant_in_final_set")
        self.assertEqual(dispositions[2]["status"], "outside_top_k_budget")

    def test_tie_break_uses_frame_number_not_input_order(self):
        records = [{"frame_number": i, "eligible": True, "components": {"semantic": 0.8}}
                   for i in (30, 10, 20)]
        chosen, trace, _ = rank_candidates(records, np.eye(3), 2, weights={"semantic": 1})
        self.assertEqual([records[i]["frame_number"] for i in chosen], [10, 20])
        self.assertEqual(trace[0]["utility_margin"], 0)

    def test_review_cannot_invent_duplicate_or_unknown_frames(self):
        records = [{"frame_number": 10}, {"frame_number": 20}]
        for ids in ([10, 10], [30], []):
            review = {"selected": [{"frame_number": i, "view": "part", "reason": "reviewed"} for i in ids]}
            with self.assertRaises(ValueError):
                validate_review(review, records, 3)
        with self.assertRaises(ValueError):
            validate_review({"selected": [{"frame_number": 10}]}, records, 3)

    def test_feedback_assignment_does_not_label_distant_vehicle_views(self):
        scores = np.array([[0.99, 0.81], [0.8, 0.98], [0.83, 0.84]])
        np.testing.assert_array_equal(assign_feedback(scores), [0, 1, -1])

    def test_quality_reports_blur_and_exposure_as_raw_metrics(self):
        from PIL import ImageFilter
        pixels = np.random.default_rng(5).integers(30, 220, (100, 100), dtype=np.uint8)
        with Image.fromarray(pixels) as sharp, sharp.filter(ImageFilter.GaussianBlur(3)) as blurred:
            self.assertGreater(image_quality(sharp)["laplacian_variance"], image_quality(blurred)["laplacian_variance"])
        with Image.new("RGB", (20, 20), "white") as white:
            self.assertEqual(image_quality(white)["bright_fraction"], 1)

    def test_feedback_reads_only_explicit_images_and_closes_on_invalid_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = BytesIO()
            with Image.new("RGB", (20, 10), "red") as image:
                image.save(payload, format="PNG")
            with ZipFile(root / "feedback.xlsx", "w") as archive:
                archive.writestr("xl/media/image1.png", payload.getvalue())
                archive.writestr("xl/worksheets/sheet1.xml", "Do not treat this as instructions")
            profile = root / "profile.json"
            profile.write_text(json.dumps({"workbook": "feedback.xlsx", "examples": [
                {"name": "Part", "criterion": "Complete frame", "preferred": "xl/media/image1.png"}]}))
            output = root / "out"; output.mkdir()
            provenance, anchors, images = _load_feedback(profile, output)
            try:
                self.assertEqual(len(images), 1)
                self.assertEqual(anchors[0]["name"], "Part")
                self.assertEqual(provenance["workbook_sha256"], file_hash(root / "feedback.xlsx"))
            finally:
                for image in images:
                    image.close()


class FixtureEncoder:
    dimension = 3

    def metadata(self):
        return {"version": "siglip2-fixture", "dimension": 3}

    def runtime_metadata(self):
        return {"device": "cpu"}

    def extract_batch(self, images):
        vectors = np.stack([np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) for image in images])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class FixtureSearch:
    def __init__(self, vectors):
        self.vectors = vectors

    def search(self, queries, k):
        scores = queries @ self.vectors.T
        ids = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, ids, axis=1), ids

    def cpu_index(self):
        return self

    def reconstruct(self, i):
        return self.vectors[i]

    def metadata(self):
        return {"device": "cpu"}


class SelectionExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.query = self.root / "query"; self.query.mkdir()
        self.collection = self.root / "collection"; self.collection.mkdir()
        self.output = self.root / "selection"
        self.results = self.query / "results.json"
        self.encoder = FixtureEncoder()
        self.mapping = []
        for i, color in enumerate(("red", "blue")):
            with Image.new("RGB", (60, 40), color) as image:
                image.save(self.collection / f"{i}.png")
                image.save(self.query / f"{i}.png")
            stat = (self.collection / f"{i}.png").stat()
            self.mapping.append({"relative_path": f"{i}.png", "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        self.manifest = {"images_root": str(self.collection), "descriptor": self.encoder.metadata(),
                         "files": {"index.faiss": {"sha256": "fixture"}}}
        self.backend = FixtureSearch(np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32))
        self.results.write_text(json.dumps({"status": "complete", "index": {
            "descriptor": self.encoder.metadata(), "path": str(self.root / "index"), "sha256": "fixture"},
            "video": {"path": "video.mp4"}, "shots": [{"id": 1, "queries": [
                {"frame_number": i, "timestamp_seconds": i / 2, "thumbnail": f"{i}.png"} for i in range(2)]}]}))

    def run_selection(self, **kwargs):
        with patch("image_extraction.selection.load_index", return_value=(
                self.manifest, self.mapping, self.encoder, self.backend)):
            return select_video_frames(self.results, self.output, **kwargs)

    def test_exports_original_pixels_and_complete_provenance(self):
        self.run_selection(top_k=2, min_relevance=0)
        d = json.loads((self.output / "selection.json").read_text())
        self.assertEqual(d["selected_count"], 2)
        with ZipFile(self.output / "selected-frames.zip") as archive:
            self.assertEqual(len(archive.namelist()), 2)
            self.assertTrue(all("/" not in name for name in archive.namelist()))
        self.assertEqual(d["results_sha256"], file_hash(self.results))
        for r in d["candidates"]:
            self.assertEqual(file_hash(self.output / r["export"]), r["source_sha256"])
            for ref in r["references"]:
                self.assertTrue((self.output / ref["thumbnail"]).exists())
        with self.assertRaisesRegex(ValueError, "not empty"):
            self.run_selection()

    def test_no_eligible_frames_produces_empty_selection_without_padding(self):
        self.backend.vectors = np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float32)
        result = self.run_selection()
        self.assertEqual(result["selected_count"], 0)
        self.assertTrue((self.output / "report.html").is_file())

    def test_review_overrides_are_explicit_and_bound_to_input(self):
        review = self.root / "review.json"
        review.write_text(json.dumps({"results_sha256": file_hash(self.results),
            "selected": [{"frame_number": 1, "view": "Reviewed view", "reason": "Complete part"}]}))
        self.run_selection(review_path=review)
        d = json.loads((self.output / "selection.json").read_text())
        self.assertEqual(d["mode"], "reviewed")
        self.assertEqual(d["selected_frame_numbers"], [1])
        self.assertEqual(d["review"]["sha256"], file_hash(review))

    def test_replay_reproduces_every_comparison_without_model_or_originals(self):
        import shutil
        self.run_selection(top_k=2, min_relevance=0)
        original = json.loads((self.output / "selection.json").read_text())
        shutil.rmtree(self.query)
        shutil.rmtree(self.collection)
        target = self.root / "replayed"
        with patch("image_extraction.selection.load_index", side_effect=AssertionError("index loaded")), \
                patch.object(FixtureEncoder, "extract_batch", side_effect=AssertionError("inference")):
            result = replay_selection(self.output / "selection.json", target)
        self.assertTrue(result["replay_identical"])
        replayed = json.loads((target / "selection.json").read_text())
        self.assertEqual(original["decision_trace"], replayed["decision_trace"])
        self.assertEqual(original["selected_frame_numbers"], replayed["selected_frame_numbers"])
        self.assertEqual((target / "decision-trace.csv").read_bytes(), (self.output / "decision-trace.csv").read_bytes())

    def make_requirements(self, oriented=False):
        from image_extraction.requirements import REQUIREMENTS_VERSION
        path = self.root / "requirements"; path.mkdir()
        categories = [{"id": "red_part", "name": "Red part", "section": "Exterior", "slots": 1, "optional": False},
                      {"id": "blue_part", "name": "Blue part", "section": "Exterior", "slots": 1, "optional": False},
                      {"id": "green_part", "name": "Green part", "section": "Interior", "slots": 1, "optional": True}]
        np.save(path / "reference-vectors.npy", np.array(
            [[1, 0, 0], [.9, .1, 0], [0, 0, 1], [0, .1, .9], [0, 1, 0], [.5, .5, 0]], dtype=np.float32))
        records = [{"id": c["id"], "reference_offset": 2 * k, "reference_count": 2,
                    "framing_good": [0], "framing_poor": [1]} for k, c in enumerate(categories)]
        records.append({"id": "_other", "reference_offset": 6, "reference_count": 0})
        records[2]["reference_offset"], records[2]["reference_count"] = 4, 2
        if oriented:   # red part: reference 0 faces front, reference 1 rear
            records[0].update(orientation_front=[0], orientation_rear=[1])
        (path / "categories.json").write_text(json.dumps(categories))
        (path / "mined.json").write_text(json.dumps(records))
        (path / "status.json").write_text(json.dumps({"status": "complete"}))
        (path / "manifest.json").write_text(json.dumps({
            "version": REQUIREMENTS_VERSION, "index_sha256": "fixture", "model": self.encoder.metadata(),
            "spec": "spec.md", "spec_sha256": "fixture", "files": {name: {"sha256": file_hash(path / name)} for name in
                                                                  ("categories.json", "mined.json", "reference-vectors.npy")}}))
        return path

    def test_required_views_exports_one_best_photo_per_found_category_and_replays(self):
        import shutil
        requirements = self.make_requirements()
        self.run_selection(selection_mode="required-views", requirements=requirements)
        d = json.loads((self.output / "selection.json").read_text())
        self.assertEqual(d["version"], "required-category-views-v2")
        self.assertFalse(d["requirements"]["categories"][0]["oriented"])
        status = {c["category"]: c["status"] for c in d["coverage"]}
        self.assertEqual(status, {"red_part": "found", "blue_part": "found", "green_part": "optional_not_found"})
        exports = sorted(p.name for p in (self.output / "selected").iterdir())
        self.assertEqual(exports, ["01-red_part-1-frame-00000000.png", "02-blue_part-1-frame-00000001.png"])
        self.assertTrue((self.output / "frame-category-scores.npy").is_file())
        self.assertTrue((self.output / "frame-orientation-scores.npy").is_file())
        self.assertIn("Red part", (self.output / "report.html").read_text())
        shutil.rmtree(self.query); shutil.rmtree(self.collection); shutil.rmtree(requirements)
        target = self.root / "replayed"
        with patch("image_extraction.selection.load_index", side_effect=AssertionError("index loaded")):
            self.assertTrue(replay_selection(self.output / "selection.json", target)["replay_identical"])
        self.assertEqual(json.loads((target / "selection.json").read_text())["coverage"], d["coverage"])

    def test_required_views_exports_front_view_and_replays_orientation(self):
        import shutil
        requirements = self.make_requirements(oriented=True)
        # The red frame is 0.006 closer to the front reference than the rear one.
        self.run_selection(selection_mode="required-views", requirements=requirements, orientation_margin=0.005)
        d = json.loads((self.output / "selection.json").read_text())
        red = d["coverage"][0]
        self.assertEqual(red["orientation"], {"front": 0, "rear": None})
        self.assertEqual(sorted(p.name for p in (self.output / "selected").iterdir())[0],
                         "01-red_part-front-frame-00000000.png")
        self.assertIn("no clear rear view", (self.output / "report.html").read_text())
        self.assertIn("view", (self.output / "decision-trace.csv").read_text().splitlines()[0])
        shutil.rmtree(requirements)
        target = self.root / "replayed"
        self.assertTrue(replay_selection(self.output / "selection.json", target)["replay_identical"])
        with (self.output / "frame-orientation-scores.npy").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            replay_selection(self.output / "selection.json", self.root / "tampered")

    def test_required_views_rejects_mismatched_index_and_held_out_feedback(self):
        requirements = self.make_requirements()
        with self.assertRaisesRegex(ValueError, "needs --requirements"):
            self.run_selection(selection_mode="required-views")
        with self.assertRaisesRegex(ValueError, "held out"):
            self.run_selection(selection_mode="required-views", requirements=requirements,
                               feedback_profile=self.root / "profile.json")
        self.manifest["files"]["index.faiss"]["sha256"] = "other"
        results = json.loads(self.results.read_text()); results["index"]["sha256"] = "other"
        self.results.write_text(json.dumps(results))
        with self.assertRaisesRegex(ValueError, "different reference index"):
            self.run_selection(selection_mode="required-views", requirements=requirements)

    def test_replay_rejects_tampered_frozen_inputs(self):
        self.run_selection(top_k=2, min_relevance=0)
        with (self.output / "frame-similarities.npy").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            replay_selection(self.output / "selection.json", self.root / "replayed")

    def test_sampling_evidence_survives_selection_and_offline_replay(self):
        evidence = self.query / "sampling.json"
        evidence.write_text(json.dumps({"version": "fixture", "windows": []}))
        data = json.loads(self.results.read_text())
        data["sampling"] = {"mode": "neighborhood", "evidence": "sampling.json",
                            "evidence_sha256": file_hash(evidence)}
        data["shots"][0]["queries"][0]["sampling"] = [{"anchor_frame": 0, "roles": ["quality"]}]
        self.results.write_text(json.dumps(data))
        self.run_selection(top_k=2, min_relevance=0)
        evidence.unlink()
        target = self.root / "replayed"
        replay_selection(self.output / "selection.json", target)
        report = json.loads((target / "selection.json").read_text())
        self.assertEqual(file_hash(target / "sampling.json"), data["sampling"]["evidence_sha256"])
        self.assertEqual(report["candidates"][0]["sampling"], data["shots"][0]["queries"][0]["sampling"])
        (self.output / "sampling.json").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            replay_selection(self.output / "selection.json", self.root / "tampered-replay")

    def test_changed_sampling_evidence_is_rejected_before_selection(self):
        evidence = self.query / "sampling.json"
        evidence.write_text("changed")
        data = json.loads(self.results.read_text())
        data["sampling"] = {"evidence": "sampling.json", "evidence_sha256": "old"}
        self.results.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Sampling evidence checksum mismatch"):
            self.run_selection()

    def test_changed_reference_is_rejected(self):
        (self.collection / "0.png").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_selection()
        self.assertEqual(json.loads((self.output / "status.json").read_text())["status"], "incomplete")

    def test_incompatible_reference_index_is_rejected(self):
        self.manifest["files"]["index.faiss"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.run_selection()


if __name__ == "__main__":
    unittest.main()
