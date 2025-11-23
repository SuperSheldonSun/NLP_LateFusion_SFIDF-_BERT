import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from scipy import sparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import MindDataset
from sfidf import CorpusStats, build_sfidf_vectors, compute_corpus_stats


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    data_cfg = cfg.get("data", {})
    paths_cfg = cfg.get("paths", {})
    sfidf_cfg = cfg.get("sfidf", {})
    if "root_dir" in data_cfg:
        args.root_dir = data_cfg["root_dir"]
    if "vectors_dir" in paths_cfg:
        args.output_dir = paths_cfg["vectors_dir"]
    if "alpha" in sfidf_cfg:
        args.alpha = sfidf_cfg["alpha"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SF-IDF and SF-IDF+ vectors for MIND-small.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Project root directory (containing data/MINDsmall_train and data/MINDsmall_dev).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vectors",
        help="Directory to store SF-IDF vectors.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for synset vs entity in SF-IDF+ (alpha * synset + (1-alpha) * entity).",
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


def build_corpus(
    dataset: MindDataset,
) -> Tuple[
    CorpusStats,
    Dict[str, sparse.csr_matrix],
    Dict[str, sparse.csr_matrix],
    Dict[str, sparse.csr_matrix],
]:
    """
    Build SF-IDF and SF-IDF+ vectors over the union of train+dev news.
    """
    all_docs = {}
    for split in ["train", "dev"]:
        for doc_id, text, _ in dataset.iter_docs(split):
            # If a doc_id appears in both, keep the first occurrence
            all_docs.setdefault(doc_id, text)

    # First pass: corpus stats + per-doc synsets/entities
    stats, doc_synsets, doc_entities = compute_corpus_stats(all_docs.items())

    # Second pass: build vectors
    sfidf_vectors, entity_vectors, sfidf_plus_vectors = build_sfidf_vectors(
        stats=stats,
        doc_synsets=doc_synsets,
        doc_entities=doc_entities,
        alpha=args.alpha,  # type: ignore[name-defined]
    )

    return stats, sfidf_vectors, entity_vectors, sfidf_plus_vectors


def main() -> None:
    global args
    args = parse_args()

    root_dir = os.path.abspath(args.root_dir)
    output_dir = os.path.join(root_dir, args.output_dir)
    ensure_dir(output_dir)

    dataset = MindDataset(root_dir=root_dir)

    print("Building SF-IDF / SF-IDF+ vectors over train+dev...")
    stats, sfidf_vectors, entity_vectors, sfidf_plus_vectors = build_corpus(dataset)

    # Save stats and vectors as pickles
    stats_path = os.path.join(output_dir, "sfidf_stats.pkl")
    sfidf_path = os.path.join(output_dir, "sfidf_vectors.pkl")
    sfidf_entity_path = os.path.join(output_dir, "sfidf_entity_vectors.pkl")
    sfidf_plus_path = os.path.join(output_dir, "sfidf_plus_vectors.pkl")

    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    with open(sfidf_path, "wb") as f:
        pickle.dump(sfidf_vectors, f)
    with open(sfidf_entity_path, "wb") as f:
        pickle.dump(entity_vectors, f)
    with open(sfidf_plus_path, "wb") as f:
        pickle.dump(sfidf_plus_vectors, f)

    print(f"Saved corpus stats to {stats_path}")
    print(f"Saved SF-IDF vectors to {sfidf_path}")
    print(f"Saved entity TF-IDF vectors to {sfidf_entity_path}")
    print(f"Saved SF-IDF+ vectors to {sfidf_plus_path}")


if __name__ == "__main__":
    main()


