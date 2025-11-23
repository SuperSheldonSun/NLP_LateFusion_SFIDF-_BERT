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
    if "cached_dir" in paths_cfg:
        args.output_dir = paths_cfg["cached_dir"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cached docs and sessions for MIND-small.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Project root directory (containing data/MINDsmall_train and data/MINDsmall_dev).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cached",
        help="Directory to store cached docs and sessions.",
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


def cache_split(dataset: MindDataset, split: str, output_dir: str) -> None:
    docs = []
    for doc_id, text, meta in dataset.iter_docs(split):
        docs.append((doc_id, text, meta))

    sessions = []
    for user_id, session_id, history_ids, cand_ids, labels in dataset.iter_sessions(split):
        sessions.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "history": history_ids,
                "candidates": cand_ids,
                "labels": labels,
            }
        )

    docs_path = os.path.join(output_dir, f"docs_{split}.pkl")
    sessions_path = os.path.join(output_dir, f"sessions_{split}.pkl")

    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)
    with open(sessions_path, "wb") as f:
        pickle.dump(sessions, f)


def main() -> None:
    args = parse_args()
    root_dir = os.path.abspath(args.root_dir)
    output_dir = os.path.join(root_dir, args.output_dir)

    ensure_dir(output_dir)

    dataset = MindDataset(root_dir=root_dir)

    for split in ["train", "dev"]:
        print(f"Caching split '{split}'...")
        cache_split(dataset, split, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()


