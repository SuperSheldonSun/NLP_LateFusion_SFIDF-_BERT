import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import MindDataset
from bert import encode_docs


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    data_cfg = cfg.get("data", {})
    paths_cfg = cfg.get("paths", {})
    if "root_dir" in data_cfg:
        args.root_dir = data_cfg["root_dir"]
    if "vectors_dir" in paths_cfg:
        args.output_dir = paths_cfg["vectors_dir"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Sentence-BERT vectors for MIND-small.")
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
        help="Directory to store BERT vectors.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for encoding.",
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


def main() -> None:
    args = parse_args()

    root_dir = os.path.abspath(args.root_dir)
    output_dir = os.path.join(root_dir, args.output_dir)
    ensure_dir(output_dir)

    dataset = MindDataset(root_dir=root_dir)

    # Union of train + dev docs
    all_docs = {}
    for split in ["train", "dev"]:
        for doc_id, text, _ in dataset.iter_docs(split):
            all_docs.setdefault(doc_id, text)

    print(f"Encoding {len(all_docs)} documents with Sentence-BERT...")
    doc_vectors = encode_docs(all_docs.items(), batch_size=args.batch_size)

    output_path = os.path.join(output_dir, "bert_vectors.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(doc_vectors, f)

    print(f"Saved BERT vectors to {output_path}")


if __name__ == "__main__":
    main()


