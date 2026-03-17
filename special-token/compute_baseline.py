"""
Compute baseline eval loss (no special tokens) for each persona.
This gives the raw model's ability to predict user messages
from conversation context alone.

Usage:
    python special-token/compute_baseline.py \
    --test_examples_path llm-vs-llm/data/special_token/test_examples.jsonl \
    --synthetic_user_ids bp_career_builder__st_formal bp_tech_starter__st_formal bp_traveler__st_skeptical \
    --model_name Qwen/Qwen2.5-0.5B \
    --output_path llm-vs-llm/data/special-token/baseline_results.json
"""

import argparse
import json
import torch
from pathlib import Path

import sys
sys.path.insert(0, "special-token")
from control_token_train_updated import (
    load_examples_from_jsonl,
    load_model_and_tokenizer,
    set_seed,
    build_inputs_for_example,
    compute_loss,
)


def compute_baseline_eval_loss(
    test_examples_path: str,
    synthetic_user_id: str,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    seed: int = 0,
):
    set_seed(seed)

    eval_examples = load_examples_from_jsonl(
        path=Path(test_examples_path),
        synthetic_user_id=synthetic_user_id,
    )

    if len(eval_examples) == 0:
        print(f"No eval examples found for {synthetic_user_id}")
        return None

    model, tokenizer = load_model_and_tokenizer(model_name)

    # Build eval batches with NO special tokens (empty list)
    eval_batches = [
        build_inputs_for_example(
            model=model,
            tokenizer=tokenizer,
            example=ex,
            special_tokens=[],  # <-- no special tokens
            special_position_mode="default",
        )
        for ex in eval_examples
    ]

    # Compute mean eval loss
    total_loss = 0.0
    valid = 0
    for batch in eval_batches:
        loss = compute_loss(model, batch)
        if not (loss != loss):  # skip NaN
            total_loss += loss
            valid += 1

    if valid == 0:
        print(f"All losses were NaN for {synthetic_user_id}")
        return None

    mean_loss = total_loss / valid
    return mean_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_examples_path", type=str, required=True)
    parser.add_argument("--synthetic_user_ids", type=str, nargs="+", required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_path", type=str, default="baseline_results.json")
    args = parser.parse_args()

    results = {}
    for user_id in args.synthetic_user_ids:
        print(f"\n{'='*60}")
        print(f"Computing baseline for: {user_id}")
        print(f"{'='*60}")

        loss = compute_baseline_eval_loss(
            test_examples_path=args.test_examples_path,
            synthetic_user_id=user_id,
            model_name=args.model_name,
        )

        if loss is not None:
            results[user_id] = round(loss, 4)
            print(f"Baseline eval loss: {loss:.4f}")
        else:
            results[user_id] = None
            print(f"Failed to compute baseline")

        # Free GPU memory between personas
        torch.cuda.empty_cache()

    # Print summary table
    print(f"\n{'='*60}")
    print("BASELINE RESULTS (no special tokens)")
    print(f"{'='*60}")
    for user_id, loss in results.items():
        print(f"  {user_id}: {loss}")

    # Save results
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()