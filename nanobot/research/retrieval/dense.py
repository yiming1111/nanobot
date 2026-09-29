"""BGE-M3 embeddings backed by a persistent FAISS cosine index."""

from __future__ import annotations

import json
import os
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Sequence, cast

from nanobot.research.models import SearchHit


def _configure_faiss_loader() -> None:
    """Skip unavailable SIMD probes in generic-only Windows FAISS wheels."""
    if sys.platform != "win32" or os.environ.get("FAISS_OPT_LEVEL"):
        return
    spec = find_spec("faiss")
    locations = spec.submodule_search_locations if spec is not None else None
    if not locations:
        return
    package_dir = Path(next(iter(locations)))
    optimized_patterns = (
        "swigfaiss_avx*",
        "_swigfaiss_avx*",
        "swigfaiss_sve*",
        "_swigfaiss_sve*",
    )
    optimized_wrapper_found = any(
        next(package_dir.glob(pattern), None) is not None for pattern in optimized_patterns
    )
    if not optimized_wrapper_found:
        os.environ["FAISS_OPT_LEVEL"] = "generic"


def _research_dependency_error(package: str) -> RuntimeError:
    return RuntimeError(
        f"The optional research dependency '{package}' is not installed. "
        "Install the project with `pip install -e .[research]`."
    )


class BGEM3Encoder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        use_fp16: bool = False,
        batch_size: int = 8,
        max_length: int = 1024,
    ) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise _research_dependency_error("FlagEmbedding") from exc
        self._model = BGEM3FlagModel(model_name, devices=device, use_fp16=use_fp16)
        self.batch_size = batch_size
        self.max_length = max_length

    def encode(self, texts: Sequence[str]) -> Any:
        if not texts:
            try:
                import numpy as np
            except ImportError as exc:
                raise _research_dependency_error("numpy") from exc
            return np.empty((0, 0), dtype="float32")
        result = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return result["dense_vecs"]


class FaissDenseIndex:
    def __init__(self, index: Any, item_ids: list[str]) -> None:
        self._index = index
        self.item_ids = item_ids

    @staticmethod
    def _modules() -> tuple[Any, Any]:
        try:
            _configure_faiss_loader()
            import faiss
        except ImportError as exc:
            raise _research_dependency_error("faiss-cpu") from exc
        try:
            import numpy as np
        except ImportError as exc:
            raise _research_dependency_error("numpy") from exc
        return faiss, np

    @classmethod
    def build(cls, item_ids: list[str], vectors: Any) -> "FaissDenseIndex":
        faiss, np = cls._modules()
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[0] != len(item_ids) or not item_ids:
            raise ValueError("dense vectors must be a non-empty matrix aligned with item_ids")
        matrix = matrix.copy()
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index=index, item_ids=list(item_ids))

    def search(
        self,
        query_vector: Any,
        *,
        top_k: int = 20,
        allowed_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        faiss, np = self._modules()
        if allowed_ids is not None and not allowed_ids:
            return []
        query = np.asarray(query_vector, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = query.copy()
        faiss.normalize_L2(query)
        search_k = len(self.item_ids) if allowed_ids is not None else min(top_k, len(self.item_ids))
        scores, indexes = self._index.search(query, search_k)
        results: list[SearchHit] = []
        for score, index in zip(scores[0], indexes[0], strict=True):
            if int(index) < 0:
                continue
            item_id = self.item_ids[int(index)]
            if allowed_ids is not None and item_id not in allowed_ids:
                continue
            results.append(SearchHit(item_id=item_id, score=float(score)))
            if len(results) >= top_k:
                break
        return results

    def save(self, index_path: Path, ids_path: Path) -> None:
        faiss, _np = self._modules()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        ids_path.write_text(json.dumps(self.item_ids, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path, ids_path: Path) -> "FaissDenseIndex":
        faiss, _np = cls._modules()
        if not index_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"dense index not found: {index_path}")
        index = faiss.read_index(str(index_path))
        item_ids = cast(list[str], json.loads(ids_path.read_text(encoding="utf-8")))
        if index.ntotal != len(item_ids):
            raise ValueError("FAISS index and id mapping are inconsistent")
        return cls(index=index, item_ids=item_ids)
