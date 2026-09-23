"""Functional checks for persistence, exact search, and video reports."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
    import faiss
    import filelock
    import scenedetect
except ImportError:
    cv2 = faiss = None

from image_extraction.descriptors import SpatialDescriptor
from image_extraction.indexing import build_index, file_hash, load_index
from image_extraction.query import query_video
from image_extraction.search import ExactSearch
from image_extraction.video import detect_shots, read_selected_frames, select_frames


@unittest.skipIf(faiss is None, "Install .[retrieval] to run retrieval tests")
class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.index = self.root / "artifacts"
        self.output = self.root / "outputs"
        for number, size in enumerate(((120, 80), (80, 120), (96, 96))):
            with Image.new("RGB", size, "white") as image:
                ImageDraw.Draw(image).rectangle((8 + 12 * number, 15, 35 + 12 * number, 65), fill="black")
                image.save(self.images / f"image-{number}.png")

    def build(self, **kwargs):
        return build_index(self.images, self.index, search_device="cpu", batch_size=2, **kwargs)

    def video(self, *, cuts=True):
        path = self.root / "sample.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
        self.assertTrue(writer.isOpened())
        try:
            for number in range(40):
                frame = np.full((64, 96, 3), 220 if cuts and number >= 20 else 20, dtype=np.uint8)
                frame[12:50, 15:40] = 80 if cuts and number >= 20 else 240
                writer.write(frame)
        finally:
            writer.release()
        return path

    def test_saved_index_matches_numpy_and_resume_never_decodes(self):
        result = self.build()
        self.assertEqual(result["usable_images"], 3)
        manifest, mapping, descriptor, backend = load_index(self.index, search_device="cpu")
        vectors = backend.index.reconstruct_n(0, 3)
        scores, ids = backend.search(vectors, 10)
        self.assertEqual(scores.shape, (3, 3))
        expected = vectors @ vectors.T
        np.testing.assert_allclose(scores, np.take_along_axis(expected, ids, axis=1), atol=1e-6)
        np.testing.assert_array_equal(ids[:, 0], np.arange(3))
        checksum = file_hash(self.index / "index.faiss")
        with patch.object(SpatialDescriptor, "extract", side_effect=AssertionError("must reuse")):
            resumed = self.build()
        self.assertEqual(resumed["reused_candidates"], 3)
        self.assertEqual(resumed["processed_candidates"], 0)
        self.assertEqual(checksum, file_hash(self.index / "index.faiss"))
        self.assertEqual([entry["id"] for entry in mapping], [0, 1, 2])

    def test_interrupted_index_resumes_and_incomplete_index_is_rejected(self):
        original = SpatialDescriptor.extract
        calls = 0

        def interrupt(descriptor, image):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt()
            return original(descriptor, image)

        with patch.object(SpatialDescriptor, "extract", interrupt), self.assertRaises(KeyboardInterrupt):
            self.build()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_index(self.index, search_device="cpu")
        resumed = self.build()
        self.assertEqual(resumed["reused_candidates"], 1)
        self.assertEqual(resumed["processed_candidates"], 2)

    def test_changes_require_explicit_rebuild(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "changed"):
            self.build(descriptor=SpatialDescriptor(grid_size=4))
        with Image.new("RGB", (50, 70), "red") as image:
            image.save(self.images / "new.png")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.build()
        rebuilt = self.build(rebuild=True)
        self.assertEqual(rebuilt["usable_images"], 4)
        self.assertEqual(rebuilt["reused_candidates"], 0)

    def test_corrupt_images_reported_and_uniform_images_supported(self):
        (self.images / "broken.png").write_bytes(b"broken")
        with Image.new("RGB", (60, 30), "white") as image:
            image.save(self.images / "uniform.png")
        result = self.build()
        self.assertEqual(result["invalid_images"], 1)
        self.assertEqual(result["zero_vectors"], 1)
        errors = (self.index / "errors.jsonl").read_text()
        self.assertIn("broken.png", errors)

    def test_empty_and_all_corrupt_collections_fail(self):
        for path in self.images.iterdir():
            path.unlink()
        with self.assertRaisesRegex(ValueError, "No supported"):
            self.build()
        (self.images / "bad.png").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "No usable"):
            self.build()
        self.assertEqual(json.loads((self.index / "manifest.json").read_text())["status"], "failed")

    def test_tampered_artifacts_rejected(self):
        self.build()
        with (self.index / "index.faiss").open("ab") as stream:
            stream.write(b"damage")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_index(self.index, search_device="cpu")

    def test_build_lock_prevents_loading_or_overwriting_active_index(self):
        self.build()
        with filelock.FileLock(str(self.index / ".index.lock")):
            with self.assertRaisesRegex(ValueError, "currently being rebuilt"):
                load_index(self.index, search_device="cpu")
            with self.assertRaisesRegex(ValueError, "Another process"):
                self.build()

    def test_cpu_fallback_and_explicit_cuda_failure(self):
        with patch.object(faiss, "get_num_gpus", return_value=0, create=True):
            auto = ExactSearch(faiss.IndexFlatIP(3), "auto")
            self.assertEqual(auto.device, "cpu")
            self.assertIsNotNone(auto.fallback_reason)
            with self.assertRaisesRegex(ValueError, "CUDA search unavailable"):
                ExactSearch(faiss.IndexFlatIP(3), "cuda")

    def test_search_rejects_invalid_vectors_and_k(self):
        backend = ExactSearch(faiss.IndexFlatIP(3), "cpu")
        for vectors in (np.ones((1, 3)), np.full((1, 3), np.nan), np.zeros((1, 4))):
            with self.assertRaises(ValueError):
                backend.add(vectors)
        backend.add(np.eye(3, dtype=np.float32))
        for k in (0, 2049):
            with self.assertRaises(ValueError):
                backend.search(np.eye(3, dtype=np.float32), k)

    def test_video_cuts_and_exact_frame_timestamps(self):
        video = self.video()
        info, shots, times = detect_shots(video, min_scene_frames=5)
        self.assertEqual(info["decoded_frames"], 40)
        self.assertEqual([(s["start_frame"], s["end_frame"]) for s in shots], [(0, 20), (20, 40)])
        self.assertAlmostEqual(shots[0]["end_seconds"], 2.0)
        frames = [select_frames(shot, times)[0] for shot in shots]
        self.assertEqual(frames, [10, 30])
        # Seeking can misreport timestamps on some real MP4 files. Require the
        # reader to advance sequentially without CAP_PROP_POS_FRAMES writes.
        original_capture = cv2.VideoCapture

        class NoSeekingCapture:
            def __init__(self, path):
                self.capture = original_capture(path)

            def set(self, prop, value):
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    raise AssertionError("Selected frame extraction must not seek")
                return self.capture.set(prop, value)

            def __getattr__(self, name):
                return getattr(self.capture, name)

        with patch.object(cv2, "VideoCapture", NoSeekingCapture):
            decoded = list(read_selected_frames(video, frames))
        self.assertLess(decoded[0][2].mean(), decoded[1][2].mean())
        for number, pts, _ in decoded:
            self.assertAlmostEqual(pts, times[number], places=3)

    def test_no_cuts_is_one_shot_and_interval_sampling_stays_inside(self):
        _, shots, times = detect_shots(self.video(cuts=False))
        self.assertEqual(len(shots), 1)
        self.assertEqual(select_frames(shots[0], times), [20])
        self.assertEqual(select_frames(shots[0], times, 1), [10, 20, 30])
        self.assertLessEqual(len(select_frames(shots[0], times, 0.001)), 40)
        for value in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                select_frames(shots[0], times, value)

    def test_invalid_video_fails_without_report(self):
        bad = self.root / "invalid.mp4"
        bad.write_bytes(b"not a video")
        with self.assertRaisesRegex(ValueError, "Cannot decode"):
            detect_shots(bad)

    def test_end_to_end_report_paths_scores_and_artifact_separation(self):
        self.build()
        before = {path.name: file_hash(path) for path in self.index.iterdir() if not path.name.startswith(".")}
        summary = query_video(self.video(), self.index, self.output, search_device="cpu", min_scene_frames=5)
        self.assertEqual(summary["query_count"], 2)
        results = json.loads((self.output / "results.json").read_text())
        for shot in results["shots"]:
            for query in shot["queries"]:
                self.assertTrue((self.output / query["thumbnail"]).is_file())
                self.assertTrue(shot["start_seconds"] <= query["timestamp_seconds"] < shot["end_seconds"])
                self.assertEqual(len(query["matches"]), 3)
                for match in query["matches"]:
                    self.assertTrue(Path(match["source_path"]).is_file())
                    self.assertTrue((self.output / match["thumbnail"]).is_file())
                scores = [match["score"] for match in query["matches"]]
                self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIn("Shot 1", (self.output / "report.html").read_text())
        self.assertEqual(before, {path.name: file_hash(path) for path in self.index.iterdir() if not path.name.startswith(".")})
        with self.assertRaisesRegex(ValueError, "not empty"):
            query_video(self.video(), self.index, self.output, search_device="cpu")

    def test_changed_source_thumbnail_is_reported(self):
        self.build()
        (self.images / "image-0.png").unlink()
        summary = query_video(self.video(cuts=False), self.index, self.output, search_device="cpu")
        self.assertEqual(summary["thumbnail_errors"], 1)
        self.assertIn("Source unavailable", (self.output / "report.html").read_text())

    def test_html_escapes_source_names(self):
        (self.images / "image-0.png").rename(self.images / '<img src=x onerror="alert(1)">.png')
        self.build()
        query_video(self.video(cuts=False), self.index, self.output, search_device="cpu")
        html = (self.output / "report.html").read_text()
        self.assertNotIn('<img src=x onerror="alert(1)">', html)
        self.assertIn("&lt;img", html)

    def require_cuda(self, dimension=576):
        try:
            return ExactSearch(faiss.IndexFlatIP(dimension), "cuda")
        except ValueError as error:
            if os.environ.get("IMAGE_EXTRACTION_REQUIRE_CUDA") == "1":
                self.fail(str(error))
            self.skipTest(str(error))

    def test_gpu_matches_cpu_when_available(self):
        gpu = self.require_cuda()
        cpu = ExactSearch(faiss.IndexFlatIP(576), "cpu")
        vectors = np.random.default_rng(41).normal(size=(4096, 576)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        for backend in (cpu, gpu):
            backend.add(vectors)
        for top_k in (10, 2048):
            cpu_scores, cpu_ids = cpu.search(vectors[:4], top_k)
            gpu_scores, gpu_ids = gpu.search(vectors[:4], top_k)
            np.testing.assert_allclose(cpu_scores, gpu_scores, atol=1e-5)
            # Scores separated by <1e-6 may exchange rank across backends.
            for row in range(len(cpu_ids)):
                reference = vectors[gpu_ids[row]] @ vectors[row]
                np.testing.assert_allclose(reference, cpu_scores[row], atol=1e-6)
        portable = gpu.cpu_index()
        np.testing.assert_allclose(portable.reconstruct_n(0, len(vectors)), vectors, atol=1e-6)

    def test_gpu_build_resume_portable_load_and_video_query(self):
        self.require_cuda()
        result = build_index(self.images, self.index, search_device="cuda", batch_size=2)
        self.assertEqual(result["search"]["device"], "cuda")
        self.assertEqual(result["search"]["index_type"], "GpuIndexFlat")
        with patch.object(SpatialDescriptor, "extract", side_effect=AssertionError("decoded on resume")):
            resumed = build_index(self.images, self.index, search_device="cuda")
        self.assertEqual(resumed["reused_candidates"], 3)
        _, mapping, _, cpu = load_index(self.index, search_device="cpu")
        _, _, _, gpu = load_index(self.index, search_device="auto")
        self.assertEqual(gpu.device, "cuda")
        vectors = cpu.cpu_index().reconstruct_n(0, len(mapping))
        cpu_scores, cpu_ids = cpu.search(vectors, 3)
        gpu_scores, gpu_ids = gpu.search(vectors, 3)
        np.testing.assert_allclose(cpu_scores, gpu_scores, atol=1e-5)
        np.testing.assert_array_equal(cpu_ids, gpu_ids)
        summary = query_video(self.video(), self.index, self.output,
                              search_device="cuda", min_scene_frames=5)
        self.assertEqual(summary["search"]["device"], "cuda")
        self.assertEqual(summary["query_count"], 2)
        # A portable CPU-built artifact must also load and search on CUDA.
        cpu_index = self.root / "cpu-index"
        build_index(self.images, cpu_index, search_device="cpu")
        _, _, _, transferred = load_index(cpu_index, search_device="cuda")
        scores, ids = transferred.search(vectors, 3)
        np.testing.assert_allclose(scores, cpu_scores, atol=1e-5)
        np.testing.assert_array_equal(ids, cpu_ids)


if __name__ == "__main__":
    unittest.main()
