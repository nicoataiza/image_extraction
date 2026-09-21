import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from image_extraction.download import DATASET_ID, DATASET_REVISION, download_dataset


class DownloadTests(unittest.TestCase):
    def test_pinned_download_records_provenance_and_can_be_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = SimpleNamespace(sha=DATASET_REVISION, siblings=[
                SimpleNamespace(rfilename="car.jpg", size=3),
                SimpleNamespace(rfilename="README.md", size=4),
                SimpleNamespace(rfilename=".gitattributes", size=9),
            ])

            def fetch(**kwargs):
                (root / "car.jpg").write_bytes(b"jpg")
                (root / "README.md").write_text("card")

            with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.snapshot_download", side_effect=fetch) as snapshot:
                api.return_value.dataset_info.return_value = info
                first = download_dataset(root)
                second = download_dataset(root)
                self.assertEqual(snapshot.call_args.kwargs["revision"], DATASET_REVISION)
                self.assertEqual(snapshot.call_args.kwargs["allow_patterns"], ["car.jpg", "README.md"])
                self.assertEqual(first["image_count"], 1)
                self.assertEqual(second["status"], "downloaded")
                self.assertEqual(second["image_bytes"], 3)
                self.assertEqual(json.loads((root / "dataset-source.json").read_text())["repo_id"], DATASET_ID)

    def test_incomplete_download_is_not_marked_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = SimpleNamespace(sha=DATASET_REVISION, siblings=[SimpleNamespace(rfilename="car.jpg", size=3)])
            with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.snapshot_download"):
                api.return_value.dataset_info.return_value = info
                with self.assertRaisesRegex(ValueError, "Download incomplete"):
                    download_dataset(root)
                self.assertEqual(json.loads((root / "dataset-source.json").read_text())["status"], "downloading")

    def test_different_revision_cannot_mix_with_existing_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset-source.json").write_text(json.dumps({"repo_id": DATASET_ID, "revision": "old"}))
            with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.snapshot_download") as snapshot:
                api.return_value.dataset_info.return_value = SimpleNamespace(sha=DATASET_REVISION)
                with self.assertRaisesRegex(ValueError, "another dataset revision"):
                    download_dataset(root)
                snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
