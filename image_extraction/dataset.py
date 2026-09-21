"""Deterministic image discovery and bounded, on-demand decoding."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import logging
import os
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
LOGGER = logging.getLogger(__name__)


class ImageLoadError(ValueError):
    """An image could not be decoded."""

    def __init__(self, path: Path, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Cannot load {path}: {reason}")


@dataclass
class ImageSample:
    """A detached RGB image; the caller owns it and may close it after use."""

    path: Path
    relative_path: str
    image: Image.Image
    source_size: tuple[int, int]  # Display dimensions after EXIF orientation.
    size_bytes: int

    def close(self) -> None:
        self.image.close()


class ImageDataset:
    """Keep only sorted paths in memory; decode images on access.

    Hidden files/directories and symlinks are excluded. Discovery is recursive.
    max_side optionally downsizes without cropping, stretching, or upscaling.
    Integer access is strict; iteration skips decode failures and reports them.
    """

    def __init__(self, root: str | Path, *, max_side: int | None = None):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Image directory does not exist: {self.root}")
        if max_side is not None and max_side <= 0:
            raise ValueError("max_side must be positive")
        self.max_side = max_side
        paths = []

        def scan_error(error: OSError) -> None:
            raise error

        for directory, dirs, files in os.walk(self.root, onerror=scan_error):
            dirs[:] = sorted(
                name for name in dirs
                if not name.startswith(".") and not (Path(directory) / name).is_symlink()
            )
            for name in files:
                path = Path(directory) / name
                if (
                    not name.startswith(".")
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                    and not path.is_symlink()
                    and path.is_file()
                ):
                    paths.append(path)
        self.paths = tuple(sorted(paths, key=lambda p: p.relative_to(self.root).as_posix()))
        if not self.paths:
            raise ValueError(f"No supported images found in {self.root}")

    def __len__(self) -> int:
        """Number of candidate files, including any not yet known to be corrupt."""
        return len(self.paths)

    def __getitem__(self, index: int) -> ImageSample:
        path = self.paths[index]
        try:
            size_bytes = path.stat().st_size
            with Image.open(path) as source:
                # Decode fully: header-only checks miss truncated/corrupt pixel data.
                source.load()
                ImageOps.exif_transpose(source, in_place=True)
                source_size = source.size
                if self.max_side is not None:
                    source.thumbnail((self.max_side, self.max_side), Image.Resampling.LANCZOS)
                image = source.convert("RGB")
            return ImageSample(
                path, path.relative_to(self.root).as_posix(), image, source_size, size_bytes
            )
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            raise ImageLoadError(path, str(error)) from error

    def iter_images(
        self, *, on_error: Callable[[ImageLoadError], None] | None = None
    ) -> Iterator[ImageSample]:
        """Yield valid images, reporting failures; fail if none are usable.

        The iterator does not cache decoded images. Callers must avoid retaining
        samples indefinitely if they want bounded memory use.
        """
        usable = 0
        for index in range(len(self)):
            try:
                sample = self[index]
            except ImageLoadError as error:
                if on_error is None:
                    LOGGER.warning("%s", error)
                else:
                    on_error(error)
                continue
            usable += 1
            yield sample
        if not usable:
            raise ValueError(f"No usable images found in {self.root}")

    def __iter__(self) -> Iterator[ImageSample]:
        return self.iter_images()

    def iter_batches(
        self, batch_size: int = 16, *, on_error: Callable[[ImageLoadError], None] | None = None
    ) -> Iterator[list[ImageSample]]:
        """Yield bounded batches, including a final partial batch.

        Images retain their aspect ratios, so batches are lists, not tensors.
        Callers own the returned images and should close them after processing.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batch = []
        for sample in self.iter_images(on_error=on_error):
            batch.append(sample)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
