import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import MindDataset
from eval import evaluate
from scorer import LateFusionScorer
from sfidf import combine_sfidf_components

DEFAULT_ALPHA_VALUES = [0.3, 0.5, 0.7]
DEFAULT_LAMBDA_VALUES = [0.3, 0.5, 0.7]
DEFAULT_BASELINE_METRICS = [
    "outputs/metrics_tfidf_dev.json",
    "outputs/metrics_sfidf_plus_dev.json",
]
DEFAULT_RESULTS_JSON_TEMPLATE = "grid_results_{split}.json"

def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    data_cfg = cfg.get("data", {})
    paths_cfg = cfg.get("paths", {})
    grid_cfg = cfg.get("grid_search", {})
    eval_cfg = cfg.get("eval", {})
    if "root_dir" in data_cfg:
        args.root_dir = data_cfg["root_dir"]
    if "vectors_dir" in paths_cfg:
        args.vectors_dir = paths_cfg["vectors_dir"]
    if "outputs_dir" in paths_cfg:
        args.output_dir = paths_cfg["outputs_dir"]
    if "alpha_values" in grid_cfg:
        args.alpha_values = grid_cfg["alpha_values"]
    if "lambda_values" in grid_cfg:
        args.lambda_values = grid_cfg["lambda_values"]
    if "primary_metric" in grid_cfg:
        args.primary_metric = grid_cfg["primary_metric"]
    if "k_list" in eval_cfg:
        args.k_list = eval_cfg["k_list"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search Late Fusion (alpha, lambda).")
    parser.add_argument("--root_dir", type=str, default=".", help="Project root directory.")
    parser.add_argument(
        "--vectors_dir",
        type=str,
        default="vectors",
        help="Directory containing precomputed vector pickles.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to store ranking TSV files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--alpha_values",
        type=float,
        nargs="+",
        default=None,
        help="List of alpha values for SF-IDF+ recomposition.",
    )
    parser.add_argument(
        "--lambda_values",
        type=float,
        nargs="+",
        default=None,
        help="List of lambda values for Late Fusion.",
    )
    parser.add_argument(
        "--k_list",
        type=int,
        nargs="+",
        default=[5, 10],
        help="K values for evaluation metrics.",
    )
    parser.add_argument(
        "--primary_metric",
        type=str,
        default="nDCG@10",
        help="Metric key used for selecting best combo.",
    )
    parser.add_argument(
        "--baseline_metrics",
        type=str,
        nargs="*",
        default=None,
        help="Optional paths to JSON metric files for other baselines.",
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default="",
        help="Optional output path to dump all grid results as JSON.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to config.yaml. CLI args override config values.",
    )
    args = parser.parse_args()
    if args.config:
        cfg = load_config(args.config)
        apply_config(args, cfg)
    if args.alpha_values is None:
        args.alpha_values = DEFAULT_ALPHA_VALUES.copy()
    if args.lambda_values is None:
        args.lambda_values = DEFAULT_LAMBDA_VALUES.copy()
    if args.baseline_metrics is None:
        args.baseline_metrics = DEFAULT_BASELINE_METRICS.copy()
    if not args.results_json:
        args.results_json = DEFAULT_RESULTS_JSON_TEMPLATE.format(split=args.split)
    return args


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_pickle(path: str, expected: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {expected} at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_vectors(root_dir: str, vectors_dir: str):
    base = os.path.join(root_dir, vectors_dir)
    sfidf_path = os.path.join(base, "sfidf_vectors.pkl")
    sfidf_entity_path = os.path.join(base, "sfidf_entity_vectors.pkl")
    bert_path = os.path.join(base, "bert_vectors.pkl")
    sfidf_vectors = load_pickle(sfidf_path, "SF-IDF vectors")
    entity_vectors = load_pickle(sfidf_entity_path, "entity TF-IDF vectors")
    bert_vectors = load_pickle(bert_path, "BERT vectors")
    return sfidf_vectors, entity_vectors, bert_vectors


def run_late_fusion_ranking(
    dataset: MindDataset,
    split: str,
    scorer: LateFusionScorer,
    output_path: str,
) -> None:
    with open(output_path, "w", encoding="utf-8") as out_f:
        for user_id, session_id, history_ids, cand_ids, labels in dataset.iter_sessions(split):
            scores = scorer.score_candidates(history_ids, cand_ids)
            score_dict = dict(scores)
            for doc_id, label in zip(cand_ids, labels):
                score = score_dict.get(doc_id, 0.0)
                out_f.write(f"{user_id}\t{session_id}\t{doc_id}\t{score:.6f}\t{label}\n")


def format_float(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p") if text else "0"


def load_baseline_metrics(paths: Sequence[str]) -> List[Dict[str, Any]]:
    baselines: List[Dict[str, Any]] = []
    for path in paths:
        if not os.path.isfile(path):
            print(f"[Baseline] Skip missing metrics file: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        baselines.append(
            {
                "label": data.get("mode", os.path.basename(path)),
                "mode": data.get("mode", "baseline"),
                "alpha": None,
                "lambda": None,
                "metrics": data.get("metrics", {}),
                "source": path,
            }
        )
    return baselines


def print_table(
    late_results: List[Dict[str, Any]],
    baselines: List[Dict[str, Any]],
    metric_order: Sequence[str],
    primary_metric: str,
) -> None:
    combined = baselines + late_results
    combined.sort(key=lambda x: x.get("metrics", {}).get(primary_metric, float("-inf")), reverse=True)
    header = ["Label", "alpha", "lambda"] + list(metric_order)
    widths = [max(len(h), 12) for h in header]

    def fmt_row(values):
        cells = []
        for value, width in zip(values, widths):
            cells.append(str(value).ljust(width))
        return " | ".join(cells)

    print("\n=== Metrics Overview ===")
    print(fmt_row(header))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for entry in combined:
        label = entry.get("label", "late_fusion")
        alpha = entry.get("alpha", "-")
        lamb = entry.get("lambda", "-")
        metrics = entry.get("metrics", {})
        row = [label, alpha, lamb]
        for key in metric_order:
            value = metrics.get(key, "-")
            row.append(f"{value:.4f}" if isinstance(value, (int, float)) else value)
        print(fmt_row(row))


def main() -> None:
    args = parse_args()
    root_dir = os.path.abspath(args.root_dir)
    vectors_dir = os.path.join(root_dir, args.vectors_dir)
    output_dir = os.path.join(root_dir, args.output_dir, "late_fusion_grid")
    ensure_dir(output_dir)

    dataset = MindDataset(root_dir=root_dir)
    sfidf_vectors, entity_vectors, bert_vectors = load_vectors(root_dir, args.vectors_dir)

    grid_results: List[Dict[str, Any]] = []
    metric_keys: List[str] = []

    for alpha in args.alpha_values:
        sfidf_plus_vectors = combine_sfidf_components(sfidf_vectors, entity_vectors, alpha)
        for lambda_fusion in args.lambda_values:
            scorer = LateFusionScorer(
                lambda_fusion=lambda_fusion,
                bert_vectors=bert_vectors,
                sfidf_plus_vectors=sfidf_plus_vectors,
                normalize=True,
            )
            fname = f"late_fusion_{args.split}_alpha{format_float(alpha)}_lambda{format_float(lambda_fusion)}.tsv"
            output_path = os.path.join(output_dir, fname)
            print(f"[Grid] alpha={alpha:.3f}, lambda={lambda_fusion:.3f} -> {output_path}")
            run_late_fusion_ranking(dataset, args.split, scorer, output_path)
            metrics = evaluate(output_path, args.k_list)
            metric_keys = sorted(metrics.keys())
            grid_results.append(
                {
                    "label": f"late_fusion_a{alpha:.3f}_l{lambda_fusion:.3f}",
                    "mode": "late_fusion",
                    "alpha": round(alpha, 6),
                    "lambda": round(lambda_fusion, 6),
                    "metrics": metrics,
                    "tsv_path": output_path,
                }
            )
            print(f"[Metrics] {metrics}")

    baselines = load_baseline_metrics(args.baseline_metrics)
    metric_order = metric_keys if metric_keys else ["Hit@5", "Hit@10", "nDCG@5", "nDCG@10", "MRR"]
    print_table(grid_results, baselines, metric_order, args.primary_metric)

    results_json_path = args.results_json
    if results_json_path:
        if not os.path.isabs(results_json_path):
            results_json_path = os.path.join(output_dir, results_json_path)

    if results_json_path:
        payload = {
            "grid_results": grid_results,
            "baselines": baselines,
            "primary_metric": args.primary_metric,
            "split": args.split,
            "k_list": args.k_list,
        }
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Grid] Saved aggregated results to {results_json_path}")


if __name__ == "__main__":
    main()

