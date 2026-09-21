"""Exact FAISS inner-product search with explicit CPU/CUDA selection."""

import sys

import numpy as np


def require_faiss():
    try:
        import faiss
    except ImportError as error:
        raise ValueError("FAISS is required. Install .[retrieval] for CPU support, or GPU-enabled FAISS separately.") from error
    return faiss


class ExactSearch:
    """Own the FAISS index and GPU resources for their entire shared lifetime."""

    def __init__(self, cpu_index, device: str = "auto"):
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("search device must be auto, cpu, or cuda")
        self.faiss = require_faiss()
        self.index = cpu_index
        self.resources = None
        self.device = "cpu"
        self.requested = device
        self.fallback_reason = None
        if device != "cpu":
            try:
                if not hasattr(self.faiss, "StandardGpuResources") or self.faiss.get_num_gpus() < 1:
                    raise RuntimeError("GPU-enabled FAISS and a visible CUDA device are required")
                resources = self.faiss.StandardGpuResources()
                resources.setTempMemory(128 * 1024 * 1024)
                # Probe actual kernels, not just driver/device visibility.
                probe = self.faiss.index_cpu_to_gpu(resources, 0, self.faiss.IndexFlatIP(cpu_index.d))
                vector = np.zeros((1, cpu_index.d), dtype=np.float32)
                vector[0, 0] = 1
                probe.add(vector)
                scores, ids = probe.search(vector, 1)
                if ids[0, 0] != 0 or not np.isclose(scores[0, 0], 1):
                    raise RuntimeError("CUDA FAISS search probe returned an incorrect result")
                del probe
                self.index = self.faiss.index_cpu_to_gpu(resources, 0, cpu_index)
                self.resources = resources
                self.device = "cuda"
            except (RuntimeError, AttributeError) as error:
                if device == "cuda":
                    raise ValueError(f"CUDA search unavailable: {error}") from error
                self.fallback_reason = str(error)
        message = f"Search backend: {self.device}"
        if self.fallback_reason:
            message += f" (auto fallback: {self.fallback_reason})"
        print(message, file=sys.stderr)

    def metadata(self) -> dict:
        return {"requested_device": self.requested, "device": self.device,
                "fallback_reason": self.fallback_reason, "faiss_version": self.faiss.__version__,
                "metric": "inner_product", "precision": "float32", "exact": True}

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(self._vectors(vectors))

    def search(self, vectors: np.ndarray, top_k: int = 10):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.index.ntotal == 0:
            raise ValueError("Cannot search an empty index")
        # GPU FAISS limits k to 2048. Keep both backends' public interface equal.
        if top_k > 2048:
            raise ValueError("top_k must be at most 2048")
        scores, ids = self.index.search(self._vectors(vectors), min(top_k, self.index.ntotal))
        if not np.isfinite(scores).all() or np.any(ids < 0):
            raise ValueError("FAISS returned invalid search results")
        # Stable presentation among ties returned by FAISS (ties at k can differ).
        for row in range(len(ids)):
            order = np.lexsort((ids[row], -scores[row]))
            scores[row], ids[row] = scores[row, order], ids[row, order]
        return scores, ids

    def cpu_index(self):
        return self.faiss.index_gpu_to_cpu(self.index) if self.device == "cuda" else self.index

    def _vectors(self, vectors):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.index.d or not np.isfinite(vectors).all():
            raise ValueError("Expected a finite descriptor matrix with the index dimension")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.all((norms == 0) | np.isclose(norms, 1, atol=1e-5)):
            raise ValueError("Descriptors must be unit length or zero")
        return vectors
