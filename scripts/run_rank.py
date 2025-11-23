import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import MindDataset
from scorer import LateFusionScorer, score_bert_session, score_sfidf_session


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    data_cfg = cfg.get("data", {})
    paths_cfg = cfg.get("paths", {})
    fusion_cfg = cfg.get("fusion", {})
    if "root_dir" in data_cfg:
        args.root_dir = data_cfg["root_dir"]
    if "outputs_dir" in paths_cfg:
        args.output_dir = paths_cfg["outputs_dir"]
    if "lambda_fusion" in fusion_cfg:
        args.lambda_fusion = fusion_cfg["lambda_fusion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ranking for MIND-small with multiple modes.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Project root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to store ranking TSV files.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tfidf", "sfidf", "sfidf_plus", "bert", "late_fusion"],
        required=True,
        help="Scoring mode.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev"],
        help="Data split to rank on.",
    )
    parser.add_argument(
        "--lambda_fusion",
        type=float,
        default=0.5,
        help="Lambda for Late Fusion: score = lambda * sim_bert + (1-lambda) * sim_sfidf_plus.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to config.yaml.",
    )
    args = parser.parse_args()
    if args.config:
        cfg = load_config(args.config)
        apply_config(args, cfg)
    return args


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------------------
# TF-IDF baseline helpers
# ----------------------------------------------------------------------

def build_tfidf_index(dataset: MindDataset) -> Tuple[TfidfVectorizer, sparse.csr_matrix, Dict[str, int]]:
    """
    Build a TF-IDF index over all train+dev news.
    Returns:
        vectorizer, doc_matrix (num_docs x dim), doc_id -> row_index
    """
    doc_ids: List[str] = []
    texts: List[str] = []
    seen = set()
    for split in ["train", "dev"]:
        for doc_id, text, _ in dataset.iter_docs(split):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc_ids.append(doc_id)
            texts.append(text)

    vectorizer = TfidfVectorizer(max_features=100000)
    doc_matrix = vectorizer.fit_transform(texts)
    doc_id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    return vectorizer, doc_matrix, doc_id_to_index


def score_tfidf_session(
    history_ids: List[str],
    candidate_ids: List[str],
    doc_matrix: sparse.csr_matrix,
    doc_id_to_index: Dict[str, int],
) -> List[Tuple[str, float]]:
    """
    TF-IDF baseline: user profile = average TF-IDF of history docs,
    score = cosine(user, doc).
    """
    if not history_ids:
        # no history -> zero scores
        return [(doc_id, 0.0) for doc_id in candidate_ids]

    indices = [doc_id_to_index[hid] for hid in history_ids if hid in doc_id_to_index]
    if not indices:
        return [(doc_id, 0.0) for doc_id in candidate_ids]

    # Average history vectors
    hist_mat = doc_matrix[indices, :]
    user_vec = hist_mat.mean(axis=0)
    user_vec = sparse.csr_matrix(user_vec)
    # L2-normalize
    norm = sparse.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm

    scores: List[Tuple[str, float]] = []
    for doc_id in candidate_ids:
        idx = doc_id_to_index.get(doc_id)
        if idx is None:
            sim = 0.0
        else:
            doc_vec = doc_matrix[idx, :]
            if doc_vec.nnz == 0:
                sim = 0.0
            else:
                sim = float(user_vec.multiply(doc_vec).sum())
        scores.append((doc_id, sim))
    return scores


def load_vectors(root_dir: str, vectors_dir: str = "vectors"):
    """
    Load SF-IDF, SF-IDF+ and BERT vectors from pickles.
    """
    base = os.path.join(root_dir, vectors_dir)
    sfidf_path = os.path.join(base, "sfidf_vectors.pkl")
    sfidf_plus_path = os.path.join(base, "sfidf_plus_vectors.pkl")
    bert_path = os.path.join(base, "bert_vectors.pkl")

    sfidf_vectors = {}
    sfidf_plus_vectors = {}
    bert_vectors = {}

    if os.path.isfile(sfidf_path):
        with open(sfidf_path, "rb") as f:
            sfidf_vectors = pickle.load(f)
    if os.path.isfile(sfidf_plus_path):
        with open(sfidf_plus_path, "rb") as f:
            sfidf_plus_vectors = pickle.load(f)
    if os.path.isfile(bert_path):
        with open(bert_path, "rb") as f:
            bert_vectors = pickle.load(f)

    return sfidf_vectors, sfidf_plus_vectors, bert_vectors


def main() -> None:
    args = parse_args()

    root_dir = os.path.abspath(args.root_dir)
    output_dir = os.path.join(root_dir, args.output_dir)
    ensure_dir(output_dir)

    dataset = MindDataset(root_dir=root_dir)

    # Prepare resources depending on mode
    if args.mode == "tfidf":
        vectorizer, doc_matrix, doc_id_to_index = build_tfidf_index(dataset)
        sfidf_vectors = sfidf_plus_vectors = bert_vectors = None  # type: ignore[assignment]
        fusion_scorer = None  # type: ignore[assignment]
    else:
        sfidf_vectors, sfidf_plus_vectors, bert_vectors = load_vectors(root_dir)
        vectorizer = doc_matrix = doc_id_to_index = None  # type: ignore[assignment]

        if args.mode == "late_fusion":
            fusion_scorer = LateFusionScorer(
                lambda_fusion=args.lambda_fusion,
                bert_vectors=bert_vectors,
                sfidf_plus_vectors=sfidf_plus_vectors,
                normalize=True,
            )
        else:
            fusion_scorer = None  # type: ignore[assignment]

    output_path = os.path.join(output_dir, f"{args.mode}_{args.split}_rank.tsv")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for user_id, session_id, history_ids, cand_ids, labels in dataset.iter_sessions(args.split):
            if args.mode == "tfidf":
                scores = score_tfidf_session(history_ids, cand_ids, doc_matrix, doc_id_to_index)  # type: ignore[arg-type]
            elif args.mode == "sfidf":
                scores = score_sfidf_session(history_ids, cand_ids, sfidf_vectors)  # type: ignore[arg-type]
            elif args.mode == "sfidf_plus":
                scores = score_sfidf_session(history_ids, cand_ids, sfidf_plus_vectors)  # type: ignore[arg-type]
            elif args.mode == "bert":
                scores = score_bert_session(history_ids, cand_ids, bert_vectors)  # type: ignore[arg-type]
            elif args.mode == "late_fusion":
                scores = fusion_scorer.score_candidates(history_ids, cand_ids)  # type: ignore[union-attr]
            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            # Align scores with labels and write TSV
            score_dict = dict(scores)
            for doc_id, label in zip(cand_ids, labels):
                score = score_dict.get(doc_id, 0.0)
                out_f.write(f"{user_id}\t{session_id}\t{doc_id}\t{score:.6f}\t{label}\n")

    print(f"Saved ranking results to {output_path}")


if __name__ == "__main__":
    main()


