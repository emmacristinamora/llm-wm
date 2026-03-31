# special-token/run_macro_grid.py

# === IMPORTS ===

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import List


# === IO / LOGGING HELPERS ===

def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(row, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# === PARSING HELPERS ===

def parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def build_python_command(
    train_script_path: Path,
    repo_root: Path,
    examples_path: str,
    runs_root: str,
    base_persona_id: str,
    style_id: str,
    held_out_topic_id: str,
    num_special_tokens: int,
    token_placement: str,
    position_mode: str,
    default_chat_template: bool,
    use_examples_percentage: float,
    model_name: str,
    max_length: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    grad_accum_steps: int,
    max_grad_norm: float,
    eval_every_steps: int,
    save_per_eval: bool,
    seed: int,
    use_fp16: bool,
    use_bf16: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        str(train_script_path),
        "--repo_root", str(repo_root),
        "--examples_path", examples_path,
        "--runs_root", runs_root,
        "--base_persona_id", base_persona_id,
        "--style_id", style_id,
        "--held_out_topic_id", held_out_topic_id,
        "--num_special_tokens", str(num_special_tokens),
        "--token_placement", token_placement,
        "--position_mode", position_mode,
        "--use_examples_percentage", str(use_examples_percentage),
        "--model_name", model_name,
        "--max_length", str(max_length),
        "--batch_size", str(batch_size),
        "--num_epochs", str(num_epochs),
        "--learning_rate", str(learning_rate),
        "--weight_decay", str(weight_decay),
        "--warmup_ratio", str(warmup_ratio),
        "--grad_accum_steps", str(grad_accum_steps),
        "--max_grad_norm", str(max_grad_norm),
        "--eval_every_steps", str(eval_every_steps),
        "--seed", str(seed),
    ]

    if default_chat_template:
        cmd.append("--default_chat_template")

    if save_per_eval:
        cmd.append("--save_per_eval")

    if use_fp16:
        cmd.append("--use_fp16")
    if use_bf16:
        cmd.append("--use_bf16")

    return cmd


# === RUN EXECUTION ===

def run_one_command(
    cmd: List[str],
    run_index: int,
    total_runs: int,
    dry_run: bool,
) -> int:
    print("=" * 100)
    print(f"[RUN {run_index}/{total_runs}]")
    print(" ".join(cmd))
    print("=" * 100)

    if dry_run:
        return 0

    completed = subprocess.run(cmd)
    return int(completed.returncode)


# === MAIN ===

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--train_script", type=str, default="train_special_token.py")
    parser.add_argument("--examples_path", type=str, default="data/examples.jsonl")
    parser.add_argument("--runs_root", type=str, default="data/runs")

    parser.add_argument("--personas", type=str, required=True)
    parser.add_argument("--styles", type=str, required=True)
    parser.add_argument("--topics", type=str, required=True)

    # baseline handled separately
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--baseline_token_placement", type=str, default="after_context")
    parser.add_argument("--baseline_position_mode", type=str, default="default")

    # micro grid for non-baseline runs
    parser.add_argument("--token_counts", type=int, nargs="+", default=[1, 3, 5, 10, 15])
    parser.add_argument("--num_epochs", type=int, nargs="+", default=[3])
    parser.add_argument(
        "--token_placements",
        type=str,
        nargs="+",
        default=["after_context"],
        choices=["before_context", "after_context"],
    )
    parser.add_argument(
        "--position_modes",
        type=str,
        nargs="+",
        default=["default", "shared_position"],
        choices=["default", "shared_position"],
    )
    parser.add_argument(
        "--default_chat_template",
        action="store_true",
        help="Whether to use the default chat template (with system/assistant/user roles) or a simplified template."
    )
    parser.add_argument(
        "--use_examples_percentage",
        type=float,
        nargs="+",
        default=[1.0],
        help="Percentage of available examples to use for training."
    )
    # shared training args
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--eval_every_steps", type=int, default=20)
    parser.add_argument("--save_per_eval", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.use_fp16 and args.use_bf16:
        raise ValueError("Use at most one of --use_fp16 and --use_bf16.")

    repo_root = Path(args.repo_root).resolve()
    train_script_path = repo_root / args.train_script

    if not train_script_path.exists():
        raise FileNotFoundError(f"Could not find training script: {train_script_path}")

    personas = parse_csv_arg(args.personas)
    styles = parse_csv_arg(args.styles)
    topics = parse_csv_arg(args.topics)

    if len(personas) == 0:
        raise ValueError("No personas provided.")
    if len(styles) == 0:
        raise ValueError("No styles provided.")
    if len(topics) == 0:
        raise ValueError("No topics provided.")

    macro_buckets = list(itertools.product(personas, styles, topics))

    nonbaseline_micro_configs = []
    for num_special_tokens in args.token_counts:
        if num_special_tokens <= 0:
            raise ValueError("token_counts should contain only positive values. Baseline is handled separately.")
        for token_placement in args.token_placements:
            for position_mode in args.position_modes:
                for example_percentage in args.use_examples_percentage:
                    for num_epochs in args.num_epochs:
                        nonbaseline_micro_configs.append(
                            {
                                "num_special_tokens": num_special_tokens,
                                "token_placement": token_placement,
                                "position_mode": position_mode,
                                "use_examples_percentage": example_percentage,
                                "num_epochs": num_epochs,
                            }
                        )

    planned_runs = []
    for base_persona_id, style_id, held_out_topic_id in macro_buckets:
        if args.run_baseline:
            planned_runs.append(
                {
                    "base_persona_id": base_persona_id,
                    "style_id": style_id,
                    "held_out_topic_id": held_out_topic_id,
                    "num_special_tokens": 0,
                    "token_placement": args.baseline_token_placement,
                    "position_mode": args.baseline_position_mode,
                    "use_examples_percentage": 1.0,
                    "is_baseline": True,
                    "num_epochs": 0,
                }
            )

        for micro in nonbaseline_micro_configs:
            planned_runs.append(
                {
                    "base_persona_id": base_persona_id,
                    "style_id": style_id,
                    "held_out_topic_id": held_out_topic_id,
                    "num_special_tokens": micro["num_special_tokens"],
                    "num_epochs": micro["num_epochs"],
                    "token_placement": micro["token_placement"],
                    "position_mode": micro["position_mode"],
                    "use_examples_percentage": micro["use_examples_percentage"],
                    "is_baseline": False,
                }
            )

    manifest = {
        "repo_root": str(repo_root),
        "train_script": str(train_script_path),
        "examples_path": args.examples_path,
        "runs_root": args.runs_root,
        "personas": personas,
        "styles": styles,
        "topics": topics,
        "run_baseline": args.run_baseline,
        "baseline_token_placement": args.baseline_token_placement,
        "baseline_position_mode": args.baseline_position_mode,
        "token_counts": args.token_counts,
        "token_placements": args.token_placements,
        "position_modes": args.position_modes,
        "default_chat_template": args.default_chat_template,
        "use_examples_percentage": args.use_examples_percentage,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "grad_accum_steps": args.grad_accum_steps,
        "max_grad_norm": args.max_grad_norm,
        "eval_every_steps": args.eval_every_steps,
        "save_per_eval" : args.save_per_eval,
        "seed": args.seed,
        "use_fp16": args.use_fp16,
        "use_bf16": args.use_bf16,
        "dry_run": args.dry_run,
        "fail_fast": args.fail_fast,
        "n_macro_buckets": len(macro_buckets),
        "n_nonbaseline_micro_configs": len(nonbaseline_micro_configs),
        "n_total_runs": len(planned_runs),
        "planned_runs": planned_runs,
    }

    orchestration_dir = repo_root / args.runs_root / "_orchestration"
    save_json(manifest, orchestration_dir / "macro_manifest.json")

    print("=" * 100)
    print("=== MACRO GRID RUNNER ===")
    print(json.dumps(
        {
            "personas": personas,
            "styles": styles,
            "topics": topics,
            "run_baseline": args.run_baseline,
            "n_macro_buckets": len(macro_buckets),
            "n_nonbaseline_micro_configs": len(nonbaseline_micro_configs),
            "n_total_runs": len(planned_runs),
            "dry_run": args.dry_run,
        },
        indent=2,
        ensure_ascii=False,
    ))
    print("=" * 100)

    successes = 0
    failures = 0

    for run_index, run_cfg in enumerate(planned_runs, start=1):
        cmd = build_python_command(
            train_script_path=train_script_path,
            repo_root=repo_root,
            examples_path=args.examples_path,
            runs_root=args.runs_root,
            base_persona_id=run_cfg["base_persona_id"],
            style_id=run_cfg["style_id"],
            held_out_topic_id=run_cfg["held_out_topic_id"],
            num_special_tokens=run_cfg["num_special_tokens"],
            token_placement=run_cfg["token_placement"],
            position_mode=run_cfg["position_mode"],
            default_chat_template=args.default_chat_template,
            use_examples_percentage=run_cfg["use_examples_percentage"],
            model_name=args.model_name,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_epochs=run_cfg["num_epochs"],
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            eval_every_steps=args.eval_every_steps,
            save_per_eval=args.save_per_eval,
            seed=args.seed,
            use_fp16=args.use_fp16,
            use_bf16=args.use_bf16,
        )

        return_code = run_one_command(
            cmd=cmd,
            run_index=run_index,
            total_runs=len(planned_runs),
            dry_run=args.dry_run,
        )

        log_row = {
            "run_index": run_index,
            "total_runs": len(planned_runs),
            "return_code": return_code,
            **run_cfg,
        }
        append_jsonl(log_row, orchestration_dir / "macro_execution_log.jsonl")

        if return_code == 0:
            successes += 1
        else:
            failures += 1
            if args.fail_fast:
                print(f"Stopping early because run {run_index} failed.")
                break

    print("=" * 100)
    print("=== MACRO GRID FINISHED ===")
    print(json.dumps(
        {
            "successes": successes,
            "failures": failures,
            "attempted_runs": successes + failures,
            "planned_runs": len(planned_runs),
        },
        indent=2,
        ensure_ascii=False,
    ))
    print("=" * 100)


if __name__ == "__main__":
    main()