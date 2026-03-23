# special-token/run_eval_grid.py


# === IMPORTS ===

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


# === IO HELPERS ===

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(row: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# === PARSING HELPERS ===

def parse_csv_arg(value: str) -> Optional[List[str]]:
    if not value:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items if items else None


def parse_int_list(values: List[int]) -> Optional[List[int]]:
    return values if values else None


# === RUN DISCOVERY / FILTERING ===

def get_candidate_run_dirs(runs_root: Path) -> List[Path]:
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root does not exist: {runs_root}")

    run_dirs = [p for p in runs_root.iterdir() if p.is_dir() and (p / "run_summary.json").exists()]
    return sorted(run_dirs)


def run_matches_filters(
    run_summary: Dict[str, Any],
    personas: Optional[List[str]],
    styles: Optional[List[str]],
    topics: Optional[List[str]],
    token_counts: Optional[List[int]],
    token_placements: Optional[List[str]],
    position_modes: Optional[List[str]],
    include_baseline: bool,
    include_trained: bool,
) -> bool:
    cfg = run_summary["config"]

    base_persona_id = str(cfg["base_persona_id"])
    style_id = str(cfg["style_id"])
    held_out_topic_id = str(cfg["held_out_topic_id"])
    num_special_tokens = int(cfg["num_special_tokens"])
    token_placement = str(cfg["token_placement"])
    position_mode = str(cfg["position_mode"])

    is_baseline = num_special_tokens == 0

    if is_baseline and not include_baseline:
        return False
    if (not is_baseline) and not include_trained:
        return False

    if personas is not None and base_persona_id not in personas:
        return False
    if styles is not None and style_id not in styles:
        return False
    if topics is not None and held_out_topic_id not in topics:
        return False
    if token_counts is not None and num_special_tokens not in token_counts:
        return False
    if token_placements is not None and token_placement not in token_placements:
        return False
    if position_modes is not None and position_mode not in position_modes:
        return False

    return True


# === SUMMARY HELPERS ===

def load_eval_summary(eval_dir: Path) -> Dict[str, Any]:
    summary_path = eval_dir / "eval_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing eval summary: {summary_path}")
    return load_json(summary_path)


def flatten_eval_summary(eval_summary: Dict[str, Any]) -> Dict[str, Any]:
    train_config = eval_summary["train_config"]
    bucket_summaries = eval_summary["bucket_summaries"]
    deltas = eval_summary["matched_vs_control_deltas"]

    row: Dict[str, Any] = {
        "run_name": eval_summary["run_name"],
        "base_persona_id": train_config["base_persona_id"],
        "style_id": train_config["style_id"],
        "held_out_topic_id": train_config["held_out_topic_id"],
        "num_special_tokens": train_config["num_special_tokens"],
        "token_placement": train_config["token_placement"],
        "position_mode": train_config["position_mode"],
        "model_name": train_config["model_name"],
        "is_baseline": int(train_config["num_special_tokens"]) == 0,
    }

    for bucket_name, bucket in bucket_summaries.items():
        prefix = bucket_name
        row[f"{prefix}__n_examples_raw"] = bucket["n_examples_raw"]
        row[f"{prefix}__n_examples_used"] = bucket["n_examples_used"]
        row[f"{prefix}__n_examples_dropped"] = bucket["n_examples_dropped"]
        row[f"{prefix}__mean_teacher_forced_loss"] = bucket["mean_teacher_forced_loss"]
        row[f"{prefix}__mean_generation_cosine_similarity"] = bucket["mean_generation_cosine_similarity"]
        row[f"{prefix}__exact_match_rate"] = bucket["exact_match_rate"]

    for k, v in deltas.items():
        row[k] = v

    return row


# === CLI ===

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--examples_path", type=str, default="data/examples.jsonl")
    parser.add_argument("--runs_root", type=str, default="data/runs")
    parser.add_argument("--evals_root", type=str, default="data/evals")

    parser.add_argument("--personas", type=str, default="")
    parser.add_argument("--styles", type=str, default="")
    parser.add_argument("--topics", type=str, default="")

    parser.add_argument("--token_counts", type=int, nargs="*", default=[])
    parser.add_argument("--token_placements", type=str, nargs="*", default=[])
    parser.add_argument("--position_modes", type=str, nargs="*", default=[])

    parser.add_argument("--include_baseline", action="store_true")
    parser.add_argument("--include_trained", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_runs", type=int, default=None)

    parser.add_argument("--generation_max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--sentence_model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--max_examples_per_bucket", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# === MAIN ===

def main() -> None:
    args = parse_args()

    personas = parse_csv_arg(args.personas)
    styles = parse_csv_arg(args.styles)
    topics = parse_csv_arg(args.topics)

    token_counts = parse_int_list(args.token_counts)
    token_placements = args.token_placements if args.token_placements else None
    position_modes = args.position_modes if args.position_modes else None

    if not args.include_baseline and not args.include_trained:
        raise ValueError("At least one of --include_baseline or --include_trained must be set.")

    repo_root = Path(args.repo_root)
    runs_root = repo_root / args.runs_root
    evals_root = repo_root / args.evals_root
    evals_root.mkdir(parents=True, exist_ok=True)

    candidate_run_dirs = get_candidate_run_dirs(runs_root)
    selected_run_dirs: List[Path] = []

    for run_dir in candidate_run_dirs:
        run_summary = load_json(run_dir / "run_summary.json")

        if run_matches_filters(
            run_summary=run_summary,
            personas=personas,
            styles=styles,
            topics=topics,
            token_counts=token_counts,
            token_placements=token_placements,
            position_modes=position_modes,
            include_baseline=args.include_baseline,
            include_trained=args.include_trained,
        ):
            selected_run_dirs.append(run_dir)

    if args.max_runs is not None:
        selected_run_dirs = selected_run_dirs[:args.max_runs]

    print("=" * 80)
    print("=== RUNNING EVAL GRID ===")
    print(f"[info] candidate_runs={len(candidate_run_dirs)}")
    print(f"[info] selected_runs={len(selected_run_dirs)}")
    print(f"[info] personas={personas}")
    print(f"[info] styles={styles}")
    print(f"[info] topics={topics}")
    print(f"[info] token_counts={token_counts}")
    print(f"[info] token_placements={token_placements}")
    print(f"[info] position_modes={position_modes}")
    print("=" * 80)

    summary_path = evals_root / "eval_runs_summary.jsonl"

    for idx, run_dir in enumerate(selected_run_dirs, start=1):
        run_name = run_dir.name
        eval_dir = evals_root / run_name
        eval_summary_path = eval_dir / "eval_summary.json"

        if args.skip_existing and eval_summary_path.exists():
            print(f"[skip {idx}/{len(selected_run_dirs)}] {run_name} already evaluated")
            continue

        cmd = [
            "python",
            "-u",
            "evaluate_special_token.py",
            "--repo_root", str(args.repo_root),
            "--examples_path", str(args.examples_path),
            "--runs_root", str(args.runs_root),
            "--evals_root", str(args.evals_root),
            "--run_name", run_name,
            "--generation_max_new_tokens", str(args.generation_max_new_tokens),
            "--sentence_model_name", str(args.sentence_model_name),
            "--seed", str(args.seed),
            "--save_per_example",
        ]

        if personas is not None:
            cmd.extend(["--allowed_personas", ",".join(personas)])
        if styles is not None:
            cmd.extend(["--allowed_styles", ",".join(styles)])
        if args.max_examples_per_bucket is not None:
            cmd.extend(["--max_examples_per_bucket", str(args.max_examples_per_bucket)])

        print(f"[run {idx}/{len(selected_run_dirs)}] evaluating {run_name}")
        print("[cmd]", " ".join(cmd))

        subprocess.run(cmd, check=True, cwd=repo_root)

        eval_summary = load_eval_summary(eval_dir)
        flat_row = flatten_eval_summary(eval_summary)
        append_jsonl(flat_row, summary_path)

    print("=" * 80)
    print("=== EVAL GRID COMPLETED ===")
    print(f"[info] summary_path={summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()