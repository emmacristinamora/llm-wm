"""
Cross-user evaluation for special tokens.

For each user A:
  1. Train special tokens on user A's training data
  2. Evaluate on user A's test data (same-user, should improve)
  3. Evaluate on user B and C's test data (cross-user, should NOT improve if tokens are user-specific)

This script does NOT modify any existing code. It imports from control_token_train_updated.

Usage:
    python cross_user_eval.py \
        --train_examples_path /path/to/train_examples.jsonl \
        --test_examples_path /path/to/test_examples.jsonl \
        --synthetic_user_ids bp_career_builder__st_formal bp_tech_starter__st_formal bp_traveler__st_skeptical \
        --model_name Qwen/Qwen2.5-0.5B \
        --token_count 10 \
        --train_steps 100 \
        --num_train_conversations 8 \
        --output_path cross_user_results.json
"""

import argparse
import json
import torch
from pathlib import Path
from dataclasses import asdict

# ---- Import everything from your existing training script ----
# Adjust this import path to match where your script lives
import sys
sys.path.insert(0, "special-token")  # adjust if needed

from control_token_train_updated import (
    load_examples_from_jsonl,
    load_model_and_tokenizer,
    set_seed,
    build_special_tokens,
    resize_vocab_for_special_tokens,
    freeze_all_but_input_embeddings,
    register_embedding_grad_mask,
    build_inputs_for_example,
    select_train_examples,
    train_step,
    compute_loss,
    TrainConfig,
)


def train_and_cross_evaluate(
    train_examples_path: str,
    test_examples_path: str,
    train_user_id: str,
    all_user_ids: list,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    token_count: int = 10,
    train_steps: int = 100,
    lr: float = 5e-5,
    num_train_conversations: int = 8,
    seed: int = 0,
):
    """
    Train special tokens on train_user_id, then evaluate on ALL users' test sets.
    Returns dict: {user_id: eval_loss} for each user in all_user_ids.
    """
    set_seed(seed)

    # Load training data for the train user
    train_examples_all = load_examples_from_jsonl(
        path=Path(train_examples_path),
        synthetic_user_id=train_user_id,
    )

    train_examples = select_train_examples(
        train_examples_all=train_examples_all,
        train_mode="multi_conversation",
        num_train_conversations=num_train_conversations,
    )

    if len(train_examples) == 0:
        print(f"No training examples for {train_user_id}")
        return None

    print(f"\n{'='*60}")
    print(f"Training on: {train_user_id} ({len(train_examples)} examples)")
    print(f"{'='*60}")

    # Load model and add special tokens
    model, tokenizer = load_model_and_tokenizer(model_name)
    special_tokens = build_special_tokens(token_count)
    trainable_token_ids = resize_vocab_for_special_tokens(model, tokenizer, special_tokens)

    # Initialize special token embeddings to mean
    with torch.no_grad():
        embeddings = model.get_input_embeddings()
        mean_emb = embeddings.weight[:-len(special_tokens)].float().mean(dim=0, keepdim=True)
        mean_emb = mean_emb.to(embeddings.weight.dtype)
        for tid in trainable_token_ids:
            embeddings.weight[tid].copy_(mean_emb.squeeze())

    freeze_all_but_input_embeddings(model)
    hook = register_embedding_grad_mask(model, trainable_token_ids)

    # Build training batches
    train_batches = [
        build_inputs_for_example(
            model=model,
            tokenizer=tokenizer,
            example=ex,
            special_tokens=special_tokens,
        )
        for ex in train_examples
    ]

    # Train
    optimizer = torch.optim.AdamW(
        [model.get_input_embeddings().weight],
        lr=lr,
        weight_decay=0.0,
        eps=1e-8,
    )

    for step in range(train_steps):
        batch = train_batches[step % len(train_batches)]
        loss = train_step(model, optimizer, batch)
        if step % 20 == 0 or step == train_steps - 1:
            print(f"  step {step}: train_loss = {loss:.4f}")

    # Now evaluate on ALL users' test sets
    cross_eval_results = {}

    for eval_user_id in all_user_ids:
        eval_examples = load_examples_from_jsonl(
            path=Path(test_examples_path),
            synthetic_user_id=eval_user_id,
        )

        if len(eval_examples) == 0:
            print(f"  No eval examples for {eval_user_id}")
            cross_eval_results[eval_user_id] = None
            continue

        eval_batches = [
            build_inputs_for_example(
                model=model,
                tokenizer=tokenizer,
                example=ex,
                special_tokens=special_tokens,
            )
            for ex in eval_examples
        ]

        total_loss = 0.0
        valid = 0
        for b in eval_batches:
            l = compute_loss(model, b)
            if l == l:  # not NaN
                total_loss += l
                valid += 1

        if valid > 0:
            mean_loss = total_loss / valid
            cross_eval_results[eval_user_id] = round(mean_loss, 4)
            tag = "SAME" if eval_user_id == train_user_id else "CROSS"
            print(f"  [{tag}] eval on {eval_user_id}: {mean_loss:.4f} ({len(eval_examples)} examples)")
        else:
            cross_eval_results[eval_user_id] = None

    hook.remove()
    torch.cuda.empty_cache()

    return cross_eval_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_examples_path", type=str, required=True)
    parser.add_argument("--test_examples_path", type=str, required=True)
    parser.add_argument("--synthetic_user_ids", type=str, nargs="+", required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--token_count", type=int, default=10)
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_train_conversations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="cross_user_results.json")
    args = parser.parse_args()

    all_results = {}

    for train_user_id in args.synthetic_user_ids:
        result = train_and_cross_evaluate(
            train_examples_path=args.train_examples_path,
            test_examples_path=args.test_examples_path,
            train_user_id=train_user_id,
            all_user_ids=args.synthetic_user_ids,
            model_name=args.model_name,
            token_count=args.token_count,
            train_steps=args.train_steps,
            lr=args.lr,
            num_train_conversations=args.num_train_conversations,
            seed=args.seed,
        )
        all_results[train_user_id] = result

    # Print summary table
    print(f"\n{'='*60}")
    print("CROSS-USER EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Trained on':<35} {'Evaluated on':<35} {'Eval Loss'}")
    print("-" * 80)
    for train_uid, evals in all_results.items():
        if evals is None:
            continue
        for eval_uid, loss in evals.items():
            tag = "SAME" if train_uid == eval_uid else "CROSS"
            print(f"{train_uid:<35} {eval_uid:<35} {loss}  [{tag}]")

    # Save
    with open(args.output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()