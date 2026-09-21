"""Versioned spatial grayscale and unsigned edge-orientation baseline."""

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageFilter, ImageOps

DESCRIPTOR_VERSION = "spatial-gray-edge-v1"


@dataclass(frozen=True)
class SpatialDescriptor:
    """Return a normalized float32 vector without cropping or padding images.

    Cells cover relative positions in the original frame. Grayscale cell means
    and magnitude-weighted edge histograms are independently L2 normalized,
    equally weighted by default, concatenated, and L2 normalized again.
    Uniform images produce a zero vector: they carry no layout evidence.
    """

    grid_size: int = 8
    orientation_bins: int = 8
    max_side: int = 256
    blur_radius: float = 1.0
    intensity_weight: float = 0.5

    def __post_init__(self):
        for name in ("grid_size", "orientation_bins", "max_side"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.blur_radius) or self.blur_radius < 0:
            raise ValueError("blur_radius must be finite and nonnegative")
        if not np.isfinite(self.intensity_weight) or not 0 <= self.intensity_weight <= 1:
            raise ValueError("intensity_weight must be between zero and one")

    @property
    def dimension(self) -> int:
        return self.grid_size ** 2 * (1 + self.orientation_bins)

    def metadata(self) -> dict:
        return {
            "version": DESCRIPTOR_VERSION,
            "parameters": asdict(self),
            "dimension": self.dimension,
            "dtype": "float32",
            "normalization": "block-l2, weighted concatenation, final-l2; uniform=zero",
            "preprocessing": "EXIF orientation, grayscale, aspect-preserving thumbnail, Gaussian blur; no padding",
        }

    def extract(self, image: Image.Image) -> np.ndarray:
        # Work on a detached copy so callers retain ownership of their pixels.
        oriented = ImageOps.exif_transpose(image)
        try:
            gray = oriented.convert("L")
        finally:
            oriented.close()
        try:
            gray.thumbnail((self.max_side, self.max_side), Image.Resampling.LANCZOS)
            with gray.filter(ImageFilter.GaussianBlur(self.blur_radius)) as smoothed:
                pixels = np.asarray(smoothed, dtype=np.float32) / 255.0
        finally:
            gray.close()

        # Remove global brightness and scale contrast before spatial pooling.
        pixels -= pixels.mean()
        deviation = float(pixels.std())
        if deviation < 1e-6:
            return np.zeros(self.dimension, dtype=np.float32)
        pixels /= deviation
        height, width = pixels.shape
        row = np.arange(height) * self.grid_size // height
        column = np.arange(width) * self.grid_size // width
        cells = (row[:, None] * self.grid_size + column[None, :]).ravel()
        cell_count = self.grid_size ** 2
        counts = np.bincount(cells, minlength=cell_count)
        intensity = np.bincount(cells, weights=pixels.ravel(), minlength=cell_count)
        intensity /= np.maximum(counts, 1)

        # Single-pixel dimensions have no derivative along that axis.
        dy = np.gradient(pixels, axis=0) if height > 1 else np.zeros_like(pixels)
        dx = np.gradient(pixels, axis=1) if width > 1 else np.zeros_like(pixels)
        magnitude = np.hypot(dx, dy).ravel()
        angle = (np.arctan2(dy, dx).ravel() % np.pi) * (self.orientation_bins / np.pi)
        lower = np.floor(angle).astype(np.int32)
        fraction = angle - lower
        # Interpolate adjacent bins, wrapping pi back to zero.
        edge_count = cell_count * self.orientation_bins
        edges = np.bincount(
            cells * self.orientation_bins + lower % self.orientation_bins,
            weights=magnitude * (1 - fraction), minlength=edge_count,
        )
        edges += np.bincount(
            cells * self.orientation_bins + (lower + 1) % self.orientation_bins,
            weights=magnitude * fraction, minlength=edge_count,
        )
        intensity = _normalize(intensity) * np.sqrt(self.intensity_weight)
        edges = _normalize(edges) * np.sqrt(1 - self.intensity_weight)
        return _normalize(np.concatenate((intensity, edges))).astype(np.float32)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)
