import argparse
import json
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple

import yaml


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    eval_cfg = cfg.get("eval", {})
    if "k_list" in eval_cfg:
        args.k_list = eval_cfg["k_list"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ranking results (Hit@K, nDCG@K, MRR).")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to TSV ranking file: user_id \\t session_id \\t doc_id \\t score \\t label.",
    )
    parser.add_argument(
        "--k_list",
        type=int,
        nargs="+",
        default=[5, 10],
        help="List of K values for Hit@K and nDCG@K.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to save metrics as JSON.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="",
        help="Optional mode name (e.g., tfidf, sfidf, sfidf_plus, bert, late_fusion) stored in JSON output.",
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


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def hit_at_k(labels: List[int], k: int) -> float:
    return 1.0 if any(labels[:k]) else 0.0


def dcg_at_k(labels: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(labels[:k], start=1):
        if rel > 0:
            dcg += (2**rel - 1) / (log2(i + 1))
    return dcg


def log2(x: float) -> float:
    from math import log

    return log(x, 2.0)


def ndcg_at_k(labels: List[int], k: int) -> float:
    dcg = dcg_at_k(labels, k)
    sorted_labels = sorted(labels, reverse=True)
    idcg = dcg_at_k(sorted_labels, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr(labels: List[int]) -> float:
    for i, rel in enumerate(labels, start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def load_sessions(tsv_path: str) -> Dict[str, List[Tuple[float, int]]]:
    """
    Load ranking results and group by session_id.

    Each line: user_id \\t session_id \\t doc_id \\t score \\t label
    We only need (score, label) per session for metrics.
    """
    sessions: DefaultDict[str, List[Tuple[float, int]]] = defaultdict(list)
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            _, session_id, _, score_str, label_str = parts
            try:
                score = float(score_str)
                label = int(label_str)
            except ValueError:
                continue
            sessions[session_id].append((score, label))
    return sessions


def evaluate(tsv_path: str, k_list: List[int]) -> Dict[str, float]:
    sessions = load_sessions(tsv_path)
    if not sessions:
        raise ValueError(f"No valid sessions found in {tsv_path}")

    k_list = sorted(set(k_list))

    hit_sums = {k: 0.0 for k in k_list}
    ndcg_sums = {k: 0.0 for k in k_list}
    mrr_sum = 0.0
    num_sessions = 0

    for _, items in sessions.items():
        # items: list of (score, label)
        # sort by score descending
        items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
        labels_sorted = [label for _, label in items_sorted]

        for k in k_list:
            hit_sums[k] += hit_at_k(labels_sorted, k)
            ndcg_sums[k] += ndcg_at_k(labels_sorted, k)
        mrr_sum += mrr(labels_sorted)
        num_sessions += 1

    metrics: Dict[str, float] = {}
    for k in k_list:
        metrics[f"Hit@{k}"] = hit_sums[k] / num_sessions
        metrics[f"nDCG@{k}"] = ndcg_sums[k] / num_sessions
    metrics["MRR"] = mrr_sum / num_sessions

    return metrics


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.input, args.k_list)

    print(f"Results for file: {args.input}")
    if args.mode:
        print(f"Mode: {args.mode}")
    for key in sorted(metrics.keys()):
        print(f"{key}: {metrics[key]:.6f}")

    if args.output_json:
        out = {
            "input": args.input,
            "mode": args.mode,
            "k_list": args.k_list,
            "metrics": metrics,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()


