import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import sparse


# ----------------------------------------------------------------------
# NLP utilities: tokenization, POS, lemmatization, WordNet, NER
# ----------------------------------------------------------------------

try:
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
except Exception as e:  # pragma: no cover - import-time fallback
    nltk = None
    pos_tag = None
    word_tokenize = None
    wn = None
    WordNetLemmatizer = None

_lemmatizer = WordNetLemmatizer() if WordNetLemmatizer is not None else None


def _ensure_nltk_resources() -> None:
    """
    Ensure necessary NLTK resources are available.
    This function does not call nltk.download() automatically, but raises
    a clear error message if resources are missing.
    """
    if nltk is None or pos_tag is None or word_tokenize is None or wn is None or _lemmatizer is None:
        raise RuntimeError(
            "NLTK and WordNet are required for sfidf.py. "
            "Please install nltk and download 'punkt', 'averaged_perceptron_tagger', and 'wordnet'."
        )


def _map_pos_tag(tag: str) -> str:
    """Map Penn Treebank POS tags to WordNet POS tags."""
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("N"):
        return "n"
    if tag.startswith("R"):
        return "r"
    return "n"


def preprocess(text: str) -> Tuple[List[str], List[str]]:
    """
    Tokenization + POS tagging + lemmatization.

    Returns:
        tokens: lemmatized tokens (lowercase)
        pos_tags: WordNet-style POS tags aligned with tokens
    """
    _ensure_nltk_resources()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if re.search(r"\w", t)]
    tagged = pos_tag(tokens)

    lemmas: List[str] = []
    pos_tags: List[str] = []
    for token, tag in tagged:
        wn_pos = _map_pos_tag(tag)
        lemma = _lemmatizer.lemmatize(token.lower(), pos=wn_pos) if _lemmatizer else token.lower()
        lemmas.append(lemma)
        pos_tags.append(wn_pos)
    return lemmas, pos_tags


def tokens_to_synsets(tokens: Sequence[str], pos_tags: Sequence[str]) -> List[str]:
    """
    Map tokens to WordNet synset IDs (as strings).
    Use the first synset as a simple heuristic.
    """
    _ensure_nltk_resources()
    synsets: List[str] = []
    for tok, pos in zip(tokens, pos_tags):
        if not tok:
            continue
        syn_list = wn.synsets(tok, pos=pos)
        if syn_list:
            synsets.append(syn_list[0].name())
    return synsets


# NER using spaCy if available, otherwise a simple regex fallback.
try:
    import spacy

    try:
        _spacy_nlp = spacy.load("en_core_web_sm")
    except Exception:
        _spacy_nlp = None
except Exception:  # pragma: no cover - import-time fallback
    spacy = None
    _spacy_nlp = None


def extract_entities(text: str) -> List[str]:
    """
    Named Entity Recognition.

    Priority:
    1. spaCy en_core_web_sm if available;
    2. Fallback: simple regex-based heuristic for capitalized phrases.
    """
    if _spacy_nlp is not None:
        doc = _spacy_nlp(text)
        return [ent.text for ent in doc.ents]

    # Fallback: very simple heuristic - consecutive capitalized words
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
    return candidates


# ----------------------------------------------------------------------
# Corpus-level statistics & document vectors
# ----------------------------------------------------------------------

@dataclass
class CorpusStats:
    df_synset: Dict[str, int]
    df_entity: Dict[str, int]
    num_docs: int


def compute_corpus_stats(
    docs: Iterable[Tuple[str, str]]
) -> Tuple[CorpusStats, Dict[str, List[str]], Dict[str, List[str]]]:
    """
    First pass over corpus:
    - build document frequency for synsets and entities
    - store per-document synsets/entities for later use

    Args:
        docs: iterable of (doc_id, text)

    Returns:
        corpus_stats, doc_synsets, doc_entities
    """
    df_synset: Dict[str, int] = defaultdict(int)
    df_entity: Dict[str, int] = defaultdict(int)
    doc_synsets: Dict[str, List[str]] = {}
    doc_entities: Dict[str, List[str]] = {}

    num_docs = 0
    for doc_id, text in docs:
        num_docs += 1

        tokens, pos_tags = preprocess(text)
        syns = tokens_to_synsets(tokens, pos_tags)
        ents = extract_entities(text)

        doc_synsets[doc_id] = syns
        doc_entities[doc_id] = ents

        for s in set(syns):
            df_synset[s] += 1
        for e in set(ents):
            df_entity[e] += 1

    stats = CorpusStats(df_synset=dict(df_synset), df_entity=dict(df_entity), num_docs=num_docs)
    return stats, doc_synsets, doc_entities


def compute_synset_weights(synsets: Sequence[str]) -> Dict[str, float]:
    """
    Compute per-document synset frequency (sf) weights.
    """
    counts = Counter(synsets)
    return {s: float(c) for s, c in counts.items()}


def compute_entity_weights(entities: Sequence[str]) -> Dict[str, float]:
    """
    Compute per-document entity term-frequency (tf) weights.
    """
    counts = Counter(entities)
    return {e: float(c) for e, c in counts.items()}


def _tfidf_weights(
    freq_map: Dict[str, float],
    df_map: Dict[str, int],
    num_docs: int,
) -> Dict[str, float]:
    """Generic TF-IDF-like weighting."""
    weights: Dict[str, float] = {}
    for key, tf in freq_map.items():
        df = df_map.get(key, 0)
        idf = math.log(num_docs / (df + 1.0))
        weights[key] = tf * idf
    return weights


def _l2_normalize_sparse(vec: sparse.csr_matrix) -> sparse.csr_matrix:
    if vec.nnz == 0:
        return vec
    norm = sparse.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def build_sfidf_vectors(
    stats: CorpusStats,
    doc_synsets: Dict[str, List[str]],
    doc_entities: Dict[str, List[str]],
    alpha: float = 0.5,
) -> Tuple[
    Dict[str, sparse.csr_matrix],
    Dict[str, sparse.csr_matrix],
    Dict[str, sparse.csr_matrix],
]:
    """
    Build SF-IDF (synset-only) and SF-IDF+ (synset+entity) vectors for all docs.

    Returns:
        sfidf_vectors: doc_id -> sparse vector (synset-only)
        entity_vectors: doc_id -> sparse vector (entity-only)
        sfidf_plus_vectors: doc_id -> sparse vector (combined)
    """
    # Build vocab indices
    synset_vocab = {s: i for i, s in enumerate(stats.df_synset.keys())}
    offset = len(synset_vocab)
    entity_vocab = {e: offset + i for i, e in enumerate(stats.df_entity.keys())}
    dim = len(synset_vocab) + len(entity_vocab)

    sfidf_vectors: Dict[str, sparse.csr_matrix] = {}
    entity_vectors: Dict[str, sparse.csr_matrix] = {}
    sfidf_plus_vectors: Dict[str, sparse.csr_matrix] = {}

    for doc_id, syns in doc_synsets.items():
        ents = doc_entities.get(doc_id, [])

        syn_freq = compute_synset_weights(syns)
        ent_freq = compute_entity_weights(ents)

        syn_weights = _tfidf_weights(syn_freq, stats.df_synset, stats.num_docs)
        ent_weights = _tfidf_weights(ent_freq, stats.df_entity, stats.num_docs)

        # SF-IDF (synset-only)
        syn_indices = []
        syn_values = []
        for s, w in syn_weights.items():
            idx = synset_vocab.get(s)
            if idx is None:
                continue
            syn_indices.append(idx)
            syn_values.append(w)

        if syn_indices:
            syn_vec = sparse.csr_matrix(
                (np.array(syn_values, dtype=np.float32), ([0] * len(syn_indices), syn_indices)),
                shape=(1, dim),
            )
        else:
            syn_vec = sparse.csr_matrix((1, dim), dtype=np.float32)

        syn_vec = _l2_normalize_sparse(syn_vec)
        sfidf_vectors[doc_id] = syn_vec

        # Entity TF-IDF
        ent_indices = []
        ent_values = []
        for e, w in ent_weights.items():
            idx = entity_vocab.get(e)
            if idx is None:
                continue
            ent_indices.append(idx)
            ent_values.append(w)

        if ent_indices:
            ent_vec = sparse.csr_matrix(
                (np.array(ent_values, dtype=np.float32), ([0] * len(ent_indices), ent_indices)),
                shape=(1, dim),
            )
        else:
            ent_vec = sparse.csr_matrix((1, dim), dtype=np.float32)

        ent_vec = _l2_normalize_sparse(ent_vec)
        entity_vectors[doc_id] = ent_vec

        # SF-IDF+ = alpha * syn + (1-alpha) * ent, then L2-normalize
        plus_vec = syn_vec.multiply(alpha) + ent_vec.multiply(1.0 - alpha)
        plus_vec = _l2_normalize_sparse(plus_vec)
        sfidf_plus_vectors[doc_id] = plus_vec
    return sfidf_vectors, entity_vectors, sfidf_plus_vectors


def combine_sfidf_components(
    syn_vectors: Dict[str, sparse.csr_matrix],
    entity_vectors: Dict[str, sparse.csr_matrix],
    alpha: float,
) -> Dict[str, sparse.csr_matrix]:
    """
    Recompose SF-IDF+ vectors from stored synset-only and entity-only vectors.
    """
    combined: Dict[str, sparse.csr_matrix] = {}
    doc_ids = set(syn_vectors.keys()) | set(entity_vectors.keys())
    for doc_id in doc_ids:
        syn_vec = syn_vectors.get(doc_id)
        ent_vec = entity_vectors.get(doc_id)
        if syn_vec is None and ent_vec is None:
            continue
        base = syn_vec if syn_vec is not None else ent_vec
        assert base is not None  # for type checker
        zero = sparse.csr_matrix(base.shape, dtype=base.dtype)
        syn_part = syn_vec.multiply(alpha) if syn_vec is not None else zero
        ent_part = ent_vec.multiply(1.0 - alpha) if ent_vec is not None else zero
        combined_vec = _l2_normalize_sparse(syn_part + ent_part)
        combined[doc_id] = combined_vec
    return combined


# ----------------------------------------------------------------------
# User vectors & similarity
# ----------------------------------------------------------------------

def build_user_vector_sfidf(
    doc_vectors: Dict[str, sparse.csr_matrix],
    history_ids: Sequence[str],
    weights: Sequence[float] | None = None,
) -> sparse.csr_matrix:
    """
    Build user vector as weighted average of history document vectors.
    """
    if not history_ids:
        # empty history -> zero vector with same dim as any doc vector (if available)
        if doc_vectors:
            any_vec = next(iter(doc_vectors.values()))
            return sparse.csr_matrix((1, any_vec.shape[1]), dtype=any_vec.dtype)
        return sparse.csr_matrix((1, 0), dtype=np.float32)

    if weights is None:
        weights = [1.0] * len(history_ids)
    if len(weights) != len(history_ids):
        raise ValueError("weights and history_ids must have the same length")

    acc_vec: sparse.csr_matrix | None = None
    total_weight = 0.0

    for doc_id, w in zip(history_ids, weights):
        vec = doc_vectors.get(doc_id)
        if vec is None:
            continue
        if acc_vec is None:
            acc_vec = vec.multiply(w)
        else:
            acc_vec = acc_vec + vec.multiply(w)
        total_weight += w

    if acc_vec is None:
        # no valid history doc vectors
        if doc_vectors:
            any_vec = next(iter(doc_vectors.values()))
            return sparse.csr_matrix((1, any_vec.shape[1]), dtype=any_vec.dtype)
        return sparse.csr_matrix((1, 0), dtype=np.float32)

    if total_weight > 0:
        acc_vec = acc_vec / total_weight

    return _l2_normalize_sparse(acc_vec)


def cosine_sparse(vec_user: sparse.csr_matrix, vec_doc: sparse.csr_matrix) -> float:
    """
    Cosine similarity for L2-normalized sparse vectors.
    """
    if vec_user.shape[1] != vec_doc.shape[1]:
        raise ValueError("Vector dimensions do not match")
    if vec_user.nnz == 0 or vec_doc.nnz == 0:
        return 0.0
    # Both are 1 x D, so dot product is a 1x1 matrix
    sim = vec_user.multiply(vec_doc).sum()
    return float(sim)


def sfidf_similarity(user_vec: sparse.csr_matrix, doc_vec: sparse.csr_matrix) -> float:
    return cosine_sparse(user_vec, doc_vec)


def sfidf_plus_similarity(user_vec: sparse.csr_matrix, doc_vec: sparse.csr_matrix) -> float:
    return cosine_sparse(user_vec, doc_vec)


__all__ = [
    "CorpusStats",
    "preprocess",
    "tokens_to_synsets",
    "extract_entities",
    "compute_corpus_stats",
    "compute_synset_weights",
    "compute_entity_weights",
    "build_sfidf_vectors",
    "combine_sfidf_components",
    "build_user_vector_sfidf",
    "cosine_sparse",
    "sfidf_similarity",
    "sfidf_plus_similarity",
]


