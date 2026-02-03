# main.py

# === 1. IMPORTS ===

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

# internal
from src.generate_transcript import generate_full_transcript


# === 2. UTILS ===

# define config path 
DEFAULT_CONFIG_PATH = Path("config/settings.yaml")

# load config function
def load_config(path: Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    Args:
        path (Path): Path to the YAML config file.
    Returns:
        Dict[str, Any]: The loaded configuration as a dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"!! Config file not found at '{path}'.")
    
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"!! Loaded config from '{path}' is not a dictionary at the top level.")
    
    # fail fast checks
    for k in ["global", "model", "generation", "styles", "investigator", "experiments"]:
        if k not in cfg:
            raise KeyError(f"Missing required top-level config key: '{k}'")

    if not isinstance(cfg["experiments"], dict) or len(cfg["experiments"]) == 0:
        raise ValueError("Config key 'experiments' must be a non-empty mapping of experiments.")

    if "seeds" not in cfg["global"] or not isinstance(cfg["global"]["seeds"], list):
        raise ValueError("global.seeds must exist and be a list (e.g., [1,2,3]).")

    return cfg

# function to parse CLI args
def parse_args(available_experiments: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LLM-vs-LLM experiments")

    p.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config (default: config/settings.yaml)",
    )

    p.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=(
            "Experiment id to run (e.g. 'baseline'). "
            "If omitted, you must pass --all_experiments."
        ),
    )

    p.add_argument(
        "--all_experiments",
        action="store_true",
        help="Run all experiments defined under config['experiments'].",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run a single seed. If omitted, runs all seeds in global.seeds.",
    )

    p.add_argument(
        "--list_experiments",
        action="store_true",
        help="Print available experiments and exit.",
    )

    args = p.parse_args()

    # validate args
    if args.list_experiments:
        print("Available experiments:")
        for e in available_experiments:
            print(" -", e)
        raise SystemExit(0)

    if not args.all_experiments and not args.experiment:
        raise SystemExit(
            "You must pass either --experiment <id> or --all_experiments. "
            "Use --list_experiments to see options."
        )

    if args.experiment and args.experiment not in available_experiments:
        raise SystemExit(
            f"Unknown experiment '{args.experiment}'. "
            f"Valid: {available_experiments}. "
            "Use --list_experiments."
        )

    return args


# === 3. MAIN FUNCTION ===

def main() -> None:
    # load config
    cfg = load_config(DEFAULT_CONFIG_PATH)
    available_experiments = list(cfg["experiments"].keys())

    # parse args
    args = parse_args(available_experiments)

    # if user gave a different config path, reload and recompute available experiments
    cfg_path = Path(args.config)
    if cfg_path != DEFAULT_CONFIG_PATH:
        cfg = load_config(cfg_path)
        available_experiments = list(cfg["experiments"].keys())
        if args.experiment and args.experiment not in available_experiments:
            raise SystemExit(
                f"Unknown experiment '{args.experiment}' in config '{cfg_path}'. "
                f"Valid: {available_experiments}. "
                "Use --list_experiments."
            )
        
    # determine experiments to run
    if args.all_experiments:
        experiment_ids = available_experiments
    else:
        experiment_ids = [args.experiment]

    # determine seeds to run
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = cfg["global"]["seeds"]

    # run experiments
    for exp_id in experiment_ids:
        for seed in seeds:
            print(f"=== Running experiment '{exp_id}' with seed {seed} ===")
            generate_full_transcript(
                cfg = cfg,
                experiment_id = exp_id,
                seed = seed,
            )
    print("=== All experiments completed ===")

if __name__ == "__main__":
    main()