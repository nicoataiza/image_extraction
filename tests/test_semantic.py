"""Offline checks for batched checkpoints and semantic index compatibility."""

from importlib.util import find_spec
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from image_extraction.descriptors import SpatialDescriptor
from image_extraction.indexing import build_index, load_index
from image_extraction.semantic import SemanticDescriptor


class BatchedDescriptor(SpatialDescriptor):
    """Small deterministic encoder to exercise storage independently of ML packages."""

    def extract_batch(self, images):
        return np.stack([self.extract(image) for image in images])


@unittest.skipUnless(find_spec("faiss") and find_spec("filelock"), "FAISS dependencies not installed")
class BatchedIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.images, self.index = root / "images", root / "index"
        self.images.mkdir()
        for i in range(5):
            with Image.new("RGB", (30 + i, 20), (i * 40, 100, 50)) as image:
                image.save(self.images / f"{i}.png")

    def build(self, descriptor=None, **kwargs):
        return build_index(self.images, self.index, descriptor=descriptor or BatchedDescriptor(),
                           search_device="cpu", extraction_batch_size=2, **kwargs)

    def test_interruption_keeps_only_complete_batches_and_resume_skips_decode(self):
        original = BatchedDescriptor.extract_batch
        batches = []

        def interrupted(descriptor, images):
            batches.append(len(images))
            if len(batches) == 2:
                raise KeyboardInterrupt()
            return original(descriptor, images)

        with patch.object(BatchedDescriptor, "extract_batch", interrupted), self.assertRaises(KeyboardInterrupt):
            self.build()
        with sqlite3.connect(self.index / "entries.sqlite") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM entries").fetchone()[0], 2)
        result = self.build()
        self.assertEqual(result["reused_candidates"], 2)
        self.assertEqual(result["processed_candidates"], 3)
        self.assertEqual(batches, [2, 2])
        with patch("image_extraction.dataset.ImageDataset.__getitem__", side_effect=AssertionError("decoded")):
            result = self.build()
        self.assertEqual(result["reused_candidates"], 5)
        self.assertEqual(result["processed_candidates"], 0)

    def test_corrupt_images_do_not_shift_vector_mapping(self):
        (self.images / "1.png").write_bytes(b"corrupt")
        result = self.build()
        self.assertEqual(result["invalid_images"], 1)
        self.assertEqual(result["usable_images"], 4)
        _, mapping, _, _ = load_index(self.index, search_device="cpu")
        self.assertEqual([e["relative_path"] for e in mapping], ["0.png", "2.png", "3.png", "4.png"])
        self.assertIn("1.png", (self.index / "errors.jsonl").read_text())

    def test_bad_batch_never_commits_partial_results(self):
        for invalid in (np.full((2, 576), np.nan, dtype=np.float32),
                        np.ones((1, 576), dtype=np.float32),
                        np.ones((2, 576), dtype=np.float32)):
            with patch.object(BatchedDescriptor, "extract_batch", return_value=invalid), self.assertRaises(ValueError):
                self.build()
            with sqlite3.connect(self.index / "entries.sqlite") as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM entries").fetchone()[0], 0)

    def test_source_mutation_during_batch_rejected(self):
        original = BatchedDescriptor.extract_batch

        def mutate(descriptor, images):
            (self.images / "0.png").write_bytes(b"changed")
            return original(descriptor, images)

        with patch.object(BatchedDescriptor, "extract_batch", mutate), self.assertRaisesRegex(ValueError, "changed"):
            self.build()
        with sqlite3.connect(self.index / "entries.sqlite") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM entries").fetchone()[0], 0)

    @unittest.skipUnless(find_spec("torch") and find_spec("transformers"), "Optional semantic dependencies missing")
    def test_semantic_roundtrip_is_lazy_and_rejects_incompatible_settings(self):
        def features(descriptor, images):
            result = np.zeros((len(images), descriptor.dimension), dtype=np.float32)
            result[:, 0] = 1
            return result

        descriptor = SemanticDescriptor(device="cpu")
        with patch.object(SemanticDescriptor, "extract_batch", features):
            self.build(descriptor)
        with patch.object(SemanticDescriptor, "_load", side_effect=AssertionError("model loaded")):
            manifest, mapping, loaded, backend = load_index(self.index, search_device="cpu", extraction_device="cpu")
            result = self.build(SemanticDescriptor(device="cuda"))
        self.assertIsInstance(loaded, SemanticDescriptor)
        self.assertIsNone(loaded._model)
        self.assertEqual(result["reused_candidates"], 5)
        self.assertEqual(backend.index.d, 768)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.build(SemanticDescriptor(max_num_patches=512))
        path = self.index / "manifest.json"
        manifest["descriptor"]["versions"]["transformers"] = "incompatible"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            load_index(self.index, search_device="cpu")

    @unittest.skipUnless(find_spec("torch") and find_spec("transformers") and find_spec("cv2")
                         and find_spec("scenedetect"), "Optional query dependencies missing")
    def test_semantic_video_queries_batch_images_and_label_report(self):
        import cv2
        from image_extraction.query import query_video
        batches = []

        def features(descriptor, images):
            batches.append(len(images))
            vectors = np.zeros((len(images), descriptor.dimension), dtype=np.float32)
            vectors[:, 0] = 1
            return vectors

        with patch.object(SemanticDescriptor, "extract_batch", features):
            self.build(SemanticDescriptor(device="cpu"))
            batches.clear()
            video = self.images.parent / "video.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(10):
                    writer.write(np.full((48, 64, 3), 100, dtype=np.uint8))
            finally:
                writer.release()
            output = self.images.parent / "report"
            query_video(video, self.index, output, search_device="cpu", extraction_device="cpu",
                        batch_size=2, interval_seconds=0.2)
        result = json.loads((output / "results.json").read_text())
        self.assertEqual(batches, [2, 2, 1])
        self.assertEqual(result["query_count"], 5)
        for query in result["shots"][0]["queries"]:
            self.assertEqual(len(query["matches"]), 5)
            with Image.open(output / query["thumbnail"]) as image:
                self.assertEqual(image.size, (64, 48))
        html = (output / "report.html").read_text()
        self.assertIn("Semantic matches", html)
        self.assertNotIn("Spatial grayscale/edge baseline", html)


@unittest.skipUnless(find_spec("torch") and find_spec("transformers"), "Optional semantic dependencies missing")
class SemanticEncoderTests(unittest.TestCase):
    def test_real_model_interface_with_tiny_random_model(self):
        # Exercise the actual model/processor API offline, without downloading weights.
        import torch
        from transformers import Siglip2VisionConfig, Siglip2VisionModel, Siglip2ImageProcessorPil

        descriptor = SemanticDescriptor(device="cpu")
        descriptor.dimension = 16
        descriptor.device = "cpu"
        descriptor._processor = Siglip2ImageProcessorPil()
        descriptor._model = Siglip2VisionModel(Siglip2VisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_patches=256, patch_size=16)).eval()
        with Image.new("RGB", (100, 40), "red") as wide, Image.new("RGB", (40, 100), "blue") as tall:
            vectors = descriptor.extract_batch([wide, tall])
            alone = descriptor.extract(wide)
            self.assertEqual(wide.size, (100, 40))
        self.assertEqual(vectors.shape, (2, 16))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        np.testing.assert_allclose(vectors[0], alone, atol=1e-5)
        self.assertEqual(descriptor.extract_batch([]).shape, (0, 16))

    def test_cuda_failure_is_clear_before_download(self):
        import torch
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA semantic extraction unavailable"):
                SemanticDescriptor(device="cuda")._load()

    def test_auto_cpu_fallback_and_pinned_model_loading(self):
        from types import SimpleNamespace
        import torch
        import transformers
        model = unittest.mock.MagicMock()
        model.config = SimpleNamespace(hidden_size=768)
        model.eval.return_value = model
        model.to.return_value = model
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(transformers.Siglip2ImageProcessorPil, "from_pretrained") as processor_load, \
                patch.object(transformers.Siglip2VisionModel, "from_pretrained", return_value=model) as model_load:
            descriptor = SemanticDescriptor()
            descriptor._load()
            descriptor._load()
        self.assertEqual(descriptor.device, "cpu")
        self.assertIn("unavailable", descriptor.fallback_reason)
        self.assertEqual(model_load.call_count, 1)
        self.assertEqual(model_load.call_args.kwargs["revision"], descriptor.revision)
        self.assertEqual(processor_load.call_args.kwargs["revision"], descriptor.revision)


if __name__ == "__main__":
    unittest.main()
