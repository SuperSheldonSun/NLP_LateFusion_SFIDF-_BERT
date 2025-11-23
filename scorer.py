from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import sparse

from bert import bert_similarity, build_user_vector_bert
from sfidf import build_user_vector_sfidf, cosine_sparse


# ----------------------------------------------------------------------
# Helper scorers for different modes
# ----------------------------------------------------------------------

def score_sfidf_session(
    history_ids: Sequence[str],
    candidate_ids: Sequence[str],
    doc_vectors: Dict[str, sparse.csr_matrix],
) -> List[Tuple[str, float]]:
    """
    SF-IDF-only scoring (synset-based or combined, depending on doc_vectors passed in).
    """
    user_vec = build_user_vector_sfidf(doc_vectors, history_ids)
    scores: List[Tuple[str, float]] = []
    for doc_id in candidate_ids:
        doc_vec = doc_vectors.get(doc_id)
        if doc_vec is None:
            sim = 0.0
        else:
            sim = cosine_sparse(user_vec, doc_vec)
        scores.append((doc_id, sim))
    return scores


def score_bert_session(
    history_ids: Sequence[str],
    candidate_ids: Sequence[str],
    doc_vectors: Dict[str, np.ndarray],
) -> List[Tuple[str, float]]:
    """
    BERT-only scoring.
    """
    user_vec = build_user_vector_bert(doc_vectors, history_ids)
    scores: List[Tuple[str, float]] = []
    for doc_id in candidate_ids:
        doc_vec = doc_vectors.get(doc_id)
        if doc_vec is None:
            sim = 0.0
        else:
            sim = bert_similarity(user_vec, doc_vec)
        scores.append((doc_id, sim))
    return scores


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v > min_v:
        return (arr - min_v) / (max_v - min_v)
    # constant array -> all zeros
    return np.zeros_like(arr)


@dataclass
class LateFusionScorer:
    """
    Late Fusion scorer combining BERT and SF-IDF+ similarities:

        score = lambda * sim_bert + (1 - lambda) * sim_sfidf_plus
    """

    lambda_fusion: float
    bert_vectors: Dict[str, np.ndarray]
    sfidf_plus_vectors: Dict[str, sparse.csr_matrix]
    normalize: bool = True

    def score_candidates(
        self,
        history_ids: Sequence[str],
        candidate_ids: Sequence[str],
    ) -> List[Tuple[str, float]]:
        # BERT part
        bert_scores = score_bert_session(history_ids, candidate_ids, self.bert_vectors)
        sim_bert = np.array([s for _, s in bert_scores], dtype=np.float32)

        # SF-IDF+ part
        sfidf_scores = score_sfidf_session(history_ids, candidate_ids, self.sfidf_plus_vectors)
        sim_sfidf = np.array([s for _, s in sfidf_scores], dtype=np.float32)

        if self.normalize:
            sim_bert = _minmax_normalize(sim_bert)
            sim_sfidf = _minmax_normalize(sim_sfidf)

        fused = self.lambda_fusion * sim_bert + (1.0 - self.lambda_fusion) * sim_sfidf

        results: List[Tuple[str, float]] = []
        for doc_id, score in zip(candidate_ids, fused.tolist()):
            results.append((doc_id, float(score)))
        return results


__all__ = [
    "score_sfidf_session",
    "score_bert_session",
    "LateFusionScorer",
]


