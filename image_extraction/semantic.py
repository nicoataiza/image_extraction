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
        self._choose_device(torch)
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

    def _choose_device(self, torch):
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

    def _features(self, inputs):
        return self._model(**inputs).pooler_output

    def encode_text(self, prompts):
        """Unit text vectors in this index's image space, plus runtime provenance."""
        from .labelling import encode_prompts
        return encode_prompts([{"prompts": list(prompts)}], device=self.requested_device, cache_dir=self.cache_dir)

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
                features = self._features(inputs).float()
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


FGCLIP2_MODEL_ID = "qihoo360/fg-clip2-base"
FGCLIP2_REVISION = "430fbc8a912c86fd4de601381b6245a0edab22f0"
FGCLIP2_VERSION = "fgclip2-naflex-image-v1"


class FgClip2Descriptor(SemanticDescriptor):
    """FG-CLIP 2 (fine-grained, SigLIP 2 NaFlex architecture) global image embeddings.

    Model code is vendored and reviewed (image_extraction/vendor/fgclip2), so remote
    code stays disabled. A fixed patch budget keeps batching and replay deterministic;
    the model card's size-dependent budget is not used.
    """

    def __init__(self, *, revision=FGCLIP2_REVISION, max_num_patches=256, device="auto", cache_dir=None):
        if revision != FGCLIP2_REVISION:
            raise ValueError("Unsupported FG-CLIP 2 revision; use the pinned encoder")
        if max_num_patches not in (256, 576, 1024):
            raise ValueError("FG-CLIP 2 max_num_patches must be 256, 576, or 1024")
        super().__init__(max_num_patches=256, device=device, cache_dir=cache_dir)
        self.revision, self.max_num_patches = revision, max_num_patches
        self._tokenizer = None

    def metadata(self):
        from .indexing import file_hash
        base = super().metadata()
        vendor = Path(__file__).resolve().parent / "vendor" / "fgclip2"
        return {**base, "version": FGCLIP2_VERSION, "model": FGCLIP2_MODEL_ID,
                "pooling": "Fgclip2Model.get_image_features (vision attention-pooling head)",
                "vendored_code_sha256": {name: file_hash(vendor / name)
                                         for name in ("configuration_fgclip2.py", "modeling_fgclip2.py")}}

    def _load(self):
        if self._model is not None:
            return
        self.metadata()
        os.environ.setdefault("HF_XET_CACHE", str(self.cache_dir / "xet"))
        import torch
        from transformers import AutoTokenizer, Siglip2ImageProcessorPil
        from .vendor.fgclip2 import Fgclip2Config, Fgclip2Model

        started = time.perf_counter()
        self._choose_device(torch)
        common = {"revision": self.revision, "cache_dir": str(self.cache_dir)}
        config = Fgclip2Config.from_pretrained(FGCLIP2_MODEL_ID, **common)
        config._attn_implementation = "sdpa"
        # Not Fgclip2Model.from_pretrained: under Transformers 5 the vendored (4.57-era)
        # _init_weights re-randomizes loaded tensors without reporting it. Build on CPU,
        # then require an exact, complete checkpoint match.
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        model = Fgclip2Model(config)
        weights = load_file(hf_hub_download(FGCLIP2_MODEL_ID, "model.safetensors", **common))
        model.load_state_dict(weights, strict=True)
        state = model.state_dict()
        if any(not torch.equal(state[name], tensor.to(state[name].dtype)) for name, tensor in weights.items()):
            raise ValueError("FG-CLIP 2 weights did not load exactly")
        del weights, state
        model.eval().to(self.device)
        if model.config.vision_config.hidden_size != self.dimension:
            raise ValueError("FG-CLIP 2 dimension does not match the descriptor")
        self._processor = Siglip2ImageProcessorPil.from_pretrained(FGCLIP2_MODEL_ID, **common)
        self._tokenizer = AutoTokenizer.from_pretrained(FGCLIP2_MODEL_ID, trust_remote_code=False, **common)
        self._model = model
        self.load_seconds = time.perf_counter() - started

    def _features(self, inputs):
        return self._model.get_image_features(**inputs)

    def encode_text(self, prompts, batch_size=32):
        self._load()
        import torch
        prompts = [prompt.lower() for prompt in prompts]
        chunks = []
        for start in range(0, len(prompts), batch_size):
            # Model card: lowercase, padded to 64 tokens, "short" text walk.
            inputs = self._tokenizer(prompts[start:start + batch_size], padding="max_length", max_length=64,
                                     truncation=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                vectors = self._model.get_text_features(**inputs, walk_type="short").float()
                chunks.append(torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy())
        vectors = np.concatenate(chunks)
        if (vectors.shape != (len(prompts), self.dimension) or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
            raise ValueError("FG-CLIP 2 returned invalid text embeddings")
        return vectors, {"requested_device": self.requested_device, "device": self.device,
                         "fallback_reason": self.fallback_reason, "tokenizer": type(self._tokenizer).__name__,
                         "padding": "max_length", "max_length": 64, "lowercase": True,
                         "walk_type": "short", "dtype": "float32"}
