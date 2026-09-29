"""Pinned, lazy-loaded SigLIP 2 image embeddings for semantic retrieval."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import sys
import time

import numpy as np
from PIL import ImageOps

MODEL_ID = "google/siglip2-base-patch16-naflex"
MODEL_REVISION = "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"
SEMANTIC_VERSION = "siglip2-naflex-image-v1"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "models"


class SemanticDescriptor:
    """Image-only pooled embeddings; model weights load on first extraction.

    NaFlex uses aspect-preserving patch-aligned resizing and masked token padding.
    Device and cache location are runtime choices, not embedding identities.
    """

    dimension = 768

    def __init__(self, *, revision=MODEL_REVISION, max_num_patches=256,
                 device="auto", cache_dir=None):
        if revision != MODEL_REVISION:
            raise ValueError("Unsupported semantic model revision; use the pinned encoder")
        if max_num_patches not in (256, 512, 1024):
            raise ValueError("max_num_patches must be 256, 512, or 1024")
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("extraction device must be auto, cpu, or cuda")
        self.revision = revision
        self.max_num_patches = max_num_patches
        self.requested_device = device
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE).expanduser().resolve()
        self.device = None
        self.fallback_reason = None
        self.load_seconds = 0.0
        self._model = self._processor = None

    def metadata(self):
        try:
            versions = {name: version(name) for name in ("torch", "transformers")}
        except PackageNotFoundError as error:
            raise ValueError("Semantic extraction requires the optional dependencies: pip install -e '.[semantic]'") from error
        return {"version": SEMANTIC_VERSION, "model": MODEL_ID,
                "parameters": {"revision": self.revision, "max_num_patches": self.max_num_patches},
                "dimension": self.dimension, "dtype": "float32", "normalization": "l2",
                "preprocessing": "EXIF orientation, RGB, NaFlex aspect-preserving patch-aligned resize, masked token padding; PIL backend",
                "pooling": "Siglip2VisionModel.pooler_output",
                "inference_dtype": "float32", "attention": "sdpa", "versions": versions}

    def runtime_metadata(self):
        return {"requested_device": self.requested_device, "device": self.device,
                "fallback_reason": self.fallback_reason, "model_load_seconds": self.load_seconds}

    def _load(self):
        if self._model is not None:
            return
        self.metadata()  # Give an actionable dependency error before importing.
        os.environ.setdefault("HF_XET_CACHE", str(self.cache_dir / "xet"))
        import torch
        from transformers import Siglip2ImageProcessorPil, Siglip2VisionModel

        started = time.perf_counter()
        self.device = "cpu"
        if self.requested_device != "cpu":
            try:
                if not torch.cuda.is_available():
                    raise RuntimeError("PyTorch reports CUDA unavailable")
                # Probe execution, not just driver/device enumeration.
                probe = torch.ones((2, 2), device="cuda")
                _ = probe @ probe
                torch.cuda.synchronize()
                self.device = "cuda"
            except RuntimeError as error:
                if self.requested_device == "cuda":
                    raise ValueError(f"CUDA semantic extraction unavailable: {error}") from error
                self.fallback_reason = str(error)
                print(f"Semantic extraction using CPU: {error}", file=sys.stderr)
        common = {"revision": self.revision, "cache_dir": str(self.cache_dir),
                  "trust_remote_code": False}
        processor = Siglip2ImageProcessorPil.from_pretrained(MODEL_ID, **common)
        model = Siglip2VisionModel.from_pretrained(
            MODEL_ID, dtype=torch.float32, attn_implementation="sdpa", use_safetensors=True, **common)
        model.eval().to(self.device)
        if model.config.hidden_size != self.dimension:
            raise ValueError("Semantic model dimension does not match the descriptor")
        self._processor, self._model = processor, model
        self.load_seconds = time.perf_counter() - started

    def extract_batch(self, images):
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)
        self._load()
        import torch

        rgb = []
        try:
            for image in images:
                with ImageOps.exif_transpose(image) as oriented:
                    rgb.append(oriented.convert("RGB"))
            inputs = self._processor(images=rgb, max_num_patches=self.max_num_patches,
                                     return_tensors="pt").to(self.device)
            with torch.inference_mode():
                features = self._model(**inputs).pooler_output.float()
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
                # Moving back to CPU synchronizes GPU extraction before checkpointing/timing.
                vectors = features.cpu().numpy().astype(np.float32, copy=True)
        finally:
            for image in rgb:
                image.close()
        if (vectors.shape != (len(images), self.dimension) or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
            raise ValueError("Semantic encoder returned invalid or zero embeddings")
        return vectors

    def extract(self, image):
        return self.extract_batch([image])[0]
