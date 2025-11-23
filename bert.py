from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional at import time
    SentenceTransformer = None  # type: ignore[assignment]


_model: SentenceTransformer | None = None  # type: ignore[misc]


def _get_model() -> "SentenceTransformer":
    """
    Lazy-load Sentence-BERT model.
    """
    global _model
    if _model is None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required for bert.py. "
                "Please install it via `pip install sentence-transformers`."
            )
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def encode_docs(
    docs: Iterable[Tuple[str, str]],
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """
    Encode documents into dense embeddings.

    Args:
        docs: iterable of (doc_id, text)
        batch_size: encoding batch size

    Returns:
        dict: doc_id -> L2-normalized embedding (np.ndarray, shape [dim])
    """
    model = _get_model()

    doc_ids: List[str] = []
    texts: List[str] = []
    for doc_id, text in docs:
        doc_ids.append(doc_id)
        texts.append(text)

    if not doc_ids:
        return {}

    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    doc_vectors: Dict[str, np.ndarray] = {}
    for i, doc_id in enumerate(doc_ids):
        vec = embeddings[i]
        # Safety: ensure L2-normalized
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        doc_vectors[doc_id] = vec.astype(np.float32)

    return doc_vectors


def build_user_vector_bert(
    doc_vectors: Dict[str, np.ndarray],
    history_ids: Sequence[str],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """
    Build user vector as weighted average of history document embeddings.
    """
    if not history_ids:
        # empty history -> zero vector (if we know dimension)
        if doc_vectors:
            any_vec = next(iter(doc_vectors.values()))
            return np.zeros_like(any_vec, dtype=np.float32)
        return np.zeros((0,), dtype=np.float32)

    if weights is None:
        weights = [1.0] * len(history_ids)
    if len(weights) != len(history_ids):
        raise ValueError("weights and history_ids must have the same length")

    acc_vec: np.ndarray | None = None
    total_weight = 0.0

    for doc_id, w in zip(history_ids, weights):
        vec = doc_vectors.get(doc_id)
        if vec is None:
            continue
        if acc_vec is None:
            acc_vec = vec.astype(np.float32) * w
        else:
            acc_vec += vec.astype(np.float32) * w
        total_weight += w

    if acc_vec is None:
        # no valid history docs
        if doc_vectors:
            any_vec = next(iter(doc_vectors.values()))
            return np.zeros_like(any_vec, dtype=np.float32)
        return np.zeros((0,), dtype=np.float32)

    if total_weight > 0:
        acc_vec /= total_weight

    # L2-normalize
    norm = np.linalg.norm(acc_vec)
    if norm > 0:
        acc_vec = acc_vec / norm
    return acc_vec.astype(np.float32)


def cosine_dense(vec_user: np.ndarray, vec_doc: np.ndarray) -> float:
    """
    Cosine similarity for dense vectors.
    Assumes vectors are already L2-normalized.
    """
    if vec_user.shape != vec_doc.shape:
        raise ValueError("Vector shapes do not match")
    if vec_user.size == 0 or vec_doc.size == 0:
        return 0.0
    return float(np.dot(vec_user, vec_doc))


def bert_similarity(user_vec: np.ndarray, doc_vec: np.ndarray) -> float:
    return cosine_dense(user_vec, doc_vec)


__all__ = [
    "encode_docs",
    "build_user_vector_bert",
    "cosine_dense",
    "bert_similarity",
]


