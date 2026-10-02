"""Offline checks for semantic annotations, provenance and source compatibility."""
import csv
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from image_extraction.labelling import (
    export_predictions, label_index, make_taxonomy, rank_scores, score_labels, validate_vectors,
)
from image_extraction.indexing import build_index
from image_extraction.semantic import SemanticDescriptor


class ScoringTests(unittest.TestCase):
    def test_prompt_ensemble_is_mean_and_family_ranking_is_separate(self):
        labels = [{"id": str(i), "family": family, "prompts": ["one", "two"]}
                  for family in ("view", "part") for i in range(3)]
        vectors = np.array([[1., 0.]])
        prompts = np.array([[1, 0], [0, 1], [.6, .8], [.6, -.8], [0, 1], [0, -1],
                            [1, 0], [1, 0], [0, 1], [0, 1], [-1, 0], [-1, 0]])
        scores = score_labels(vectors, prompts, labels)
        ids, values = rank_scores(scores, labels, "view")
        self.assertEqual(ids.tolist(), [[1, 0, 2]])
        np.testing.assert_allclose(values, [[.6, .5, 0]])
        self.assertEqual(rank_scores(scores, labels, "part")[0].tolist(), [[3, 4, 5]])

    def test_rounding_resolves_ties_by_taxonomy_order(self):
        labels = [{"family": "view"}] * 3
        ids, _ = rank_scores(np.array([[.5, .50000001, .2]]), labels, "view")
        self.assertEqual(ids.tolist(), [[0, 1, 2]])

    def test_nonfinite_and_zero_vectors_rejected(self):
        for vectors in (np.zeros((1, 3)), np.full((1, 3), np.nan)):
            with self.assertRaises(ValueError):
                validate_vectors(vectors, 1, 3)

    def test_ambiguous_predictions_abstain_and_remain_unreviewed(self):
        labels = [{"id": str(i), "name": str(i), "family": family}
                  for family in ("view", "part") for i in range(3)]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            export_predictions(out, [{"relative_path": "stock/photo.jpg"}],
                               np.array([[.5, .499, .1, .8, .3, .1]]), labels,
                               min_score=.1, min_margin=.01)
            with (out / "semantic-labels.csv").open(newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["semantic_label"], "uncertain")
            self.assertEqual(row["suggested_view"], "0")
            self.assertEqual(row["annotation_status"], "unreviewed")
            self.assertEqual(row["review_priority"], "ambiguous")
            self.assertEqual(row["reviewed_label"], "")

    def test_taxonomy_uses_vocabulary_not_stock_assignments(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "parts.csv"
            source.write_text("stock_number,part_type_code,part_type_name\n1,A,ENGINE\n2,B,BONNET\n3,C,GRILLE\n4,,\n")
            first, skipped = make_taxonomy(source)
            self.assertEqual(skipped, 1)
            source.write_text("stock_number,part_type_code,part_type_name\n999,C,GRILLE\n999,B,BONNET\n999,A,ENGINE\n")
            second, _ = make_taxonomy(source)
            self.assertEqual(first, second)
            source.write_text("part_type_code,part_type_name\nA,ENGINE\nA,BONNET\n")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                make_taxonomy(source)


@unittest.skipUnless(all(find_spec(n) for n in ("faiss", "filelock", "torch", "transformers")),
                     "Optional semantic/index dependencies missing")
class LabellingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for i in range(3):
            with Image.new("RGB", (24, 24), (i * 80, 50, 50)) as image:
                image.save(self.images / f"{i}.png")
        self.parts = self.root / "parts.csv"
        self.parts.write_text("part_type_code,part_type_name\nA,ENGINE\nB,BONNET\nC,GRILLE\n")
        self.index = self.root / "index"
        def image_vectors(descriptor, images):
            result = np.zeros((len(images), 768), dtype=np.float32)
            for i, image in enumerate(images):
                result[i, image.getpixel((0, 0))[0] // 80] = 1
            return result
        with patch.object(SemanticDescriptor, "extract_batch", image_vectors):
            build_index(self.images, self.index, descriptor=SemanticDescriptor(device="cpu"), search_device="cpu")

    @staticmethod
    def prompts(labels, **kwargs):
        vectors = np.zeros((sum(len(x["prompts"]) for x in labels), 768), dtype=np.float32)
        offset = 0
        for i, label in enumerate(labels):
            for _ in label["prompts"]:
                vectors[offset, i] = 1
                offset += 1
        return vectors, {"device": "test"}

    def test_full_export_reuses_vectors_and_preserves_row_mapping(self):
        out = self.root / "labels"
        with patch("image_extraction.labelling.encode_prompts", self.prompts), \
             patch.object(SemanticDescriptor, "_load", side_effect=AssertionError("image encoder loaded")):
            result = label_index(self.index, self.parts, out, batch_size=2)
        self.assertEqual(result["image_count"], 3)
        self.assertEqual(result["label_counts"], {"front_exterior": 1, "rear_exterior": 1, "side_exterior": 1})
        with (out / "semantic-labels.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["image_path"] for r in rows], ["0.png", "1.png", "2.png"])
        self.assertEqual(np.load(out / "label-scores.npy").shape, (3, 29))
        self.assertEqual(json.loads((out / "status.json").read_text())["status"], "complete")
        self.assertTrue((out / "report.html").is_file())
        with self.assertRaisesRegex(ValueError, "fresh"):
            label_index(self.index, self.parts, out)

    def test_changed_source_rejected_before_text_inference(self):
        (self.images / "0.png").write_bytes(b"changed")
        with patch("image_extraction.labelling.encode_prompts", side_effect=AssertionError("inference started")):
            with self.assertRaisesRegex(ValueError, "Image changed"):
                label_index(self.index, self.parts, self.root / "out")

    def test_output_inside_image_collection_rejected(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            label_index(self.index, self.parts, self.images / "annotations")
        self.assertFalse((self.images / "annotations").exists())


if __name__ == "__main__":
    unittest.main()
