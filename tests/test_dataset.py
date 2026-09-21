from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from image_extraction import ImageDataset, ImageLoadError
from image_extraction.__main__ import main


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "images"
        self.root.mkdir()

    def make_image(self, name, *, size=(40, 20), mode="RGB", exif=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.new(mode, size) as image:
            image.save(path, **({"exif": exif} if exif is not None else {}))
        return path

    def test_discovery_is_lazy_sorted_recursive_and_ignores_caches(self):
        self.make_image("nested/z.PNG")
        self.make_image("a.jpg")
        self.make_image(".cache/cached.png")
        self.make_image(".hidden.png")
        (self.root / "README.md").write_text("metadata")
        (self.root / "linked.png").symlink_to(self.root / "a.jpg")
        with patch("image_extraction.dataset.Image.open") as opened:
            dataset = ImageDataset(self.root)
            self.assertEqual(len(dataset), 2)
            opened.assert_not_called()
        self.assertEqual([p.relative_to(self.root).as_posix() for p in dataset.paths], ["a.jpg", "nested/z.PNG"])

    def test_orientation_rgb_and_aspect_preserving_resize(self):
        exif = Image.Exif()
        exif[274] = 6  # Rotate 90 degrees clockwise for display.
        self.make_image("portrait.jpg", size=(40, 20), exif=exif)
        sample = ImageDataset(self.root, max_side=10)[0]
        self.addCleanup(sample.close)
        self.assertEqual(sample.source_size, (20, 40))
        self.assertEqual(sample.image.size, (5, 10))
        self.assertEqual(sample.image.mode, "RGB")
        self.assertEqual(sample.relative_path, "portrait.jpg")
        self.assertIsNone(getattr(sample.image, "fp", None))

    def test_grayscale_and_rgba_convert_without_upscaling(self):
        self.make_image("gray.png", mode="L")
        self.make_image("rgba.png", mode="RGBA")
        for sample in ImageDataset(self.root, max_side=100):
            try:
                self.assertEqual(sample.image.mode, "RGB")
                self.assertEqual(sample.image.size, (40, 20))
            finally:
                sample.close()

    def test_corrupt_files_are_strict_on_access_and_skipped_in_batches(self):
        (self.root / "0-bad.jpg").write_bytes(b"not an image")
        for index in range(5):
            self.make_image(f"{index + 1}.png")
        dataset = ImageDataset(self.root)
        with self.assertRaises(ImageLoadError):
            dataset[0]
        failures = []
        sizes = []
        for batch in dataset.iter_batches(2, on_error=failures.append):
            sizes.append(len(batch))
            for sample in batch:
                sample.close()
        self.assertEqual(sizes, [2, 2, 1])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].path.name, "0-bad.jpg")

    def test_truncated_image_is_rejected_even_if_header_is_readable(self):
        path = self.make_image("truncated.bmp", size=(100, 100))
        path.write_bytes(path.read_bytes()[:100])
        with Image.open(path) as image:
            self.assertEqual(image.size, (100, 100))
        with self.assertRaises(ImageLoadError):
            ImageDataset(self.root)[0]

    def test_empty_missing_and_all_corrupt_collections_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "No supported images"):
            ImageDataset(self.root)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            ImageDataset(self.root / "missing")
        (self.root / "bad.png").write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError, "No usable images"):
            list(ImageDataset(self.root).iter_images(on_error=lambda error: None))

    def test_invalid_batch_and_resize_arguments(self):
        self.make_image("valid.png")
        with self.assertRaisesRegex(ValueError, "max_side"):
            ImageDataset(self.root, max_side=0)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            list(ImageDataset(self.root).iter_batches(0))

    def test_inspect_writes_inventory_errors_and_actual_counts(self):
        self.make_image("valid.png")
        (self.root / "bad.jpg").write_bytes(b"invalid")
        output = self.root.parent / "report"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = main(["inspect", "--images", str(self.root), "--output", str(output)])
        self.assertEqual(result, 0)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual((summary["candidate_images"], summary["usable_images"], summary["invalid_images"]), (2, 1, 1))
        record = json.loads((output / "images.jsonl").read_text())
        self.assertEqual(record["relative_path"], "valid.png")
        self.assertEqual((record["width"], record["height"]), (40, 20))
        self.assertEqual(json.loads((output / "errors.jsonl").read_text())["relative_path"], "bad.jpg")

    def test_inspect_all_corrupt_returns_failure_with_summary(self):
        (self.root / "bad.jpg").write_bytes(b"invalid")
        output = self.root.parent / "report"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = main(["inspect", "--images", str(self.root), "--output", str(output)])
        self.assertEqual(result, 1)
        self.assertEqual(json.loads((output / "summary.json").read_text())["usable_images"], 0)


if __name__ == "__main__":
    unittest.main()
