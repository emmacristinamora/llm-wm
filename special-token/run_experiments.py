# === IMPORTS ===

import json
from pathlib import Path
from typing import Any, Dict, List

from src.transcripts_to_examples import load_jsonl, prepare_examples
from src.train import TrainConfig, run_training


# === PATHS ===

REPO_ROOT = Path(__file__).resolve().parent
TRANSCRIPTS_PATH = REPO_ROOT / "synthetic-conversations" / "data" / "transcripts.jsonl"
RESULTS_PATH = REPO_ROOT / "special_token_experiments.jsonl"


# === IO ===

def append_jsonl(row: Dict[str, Any], path: Path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# === TRANSCRIPT IDS ===

def get_transcript_id(transcript: Dict[str, Any], idx: int) -> str:
    if transcript.get("conversation_id") is not None:
        return str(transcript["conversation_id"])
    if transcript.get("transcript_id") is not None:
        return str(transcript["transcript_id"])
    if transcript.get("id") is not None:
        return str(transcript["id"])
    return f"transcript_{idx}"


def attach_transcript_ids(transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    transcripts_with_ids = []

    for idx, transcript in enumerate(transcripts):
        transcript_copy = dict(transcript)
        transcript_copy["_resolved_transcript_id"] = get_transcript_id(transcript, idx)
        transcripts_with_ids.append(transcript_copy)

    return transcripts_with_ids


def filter_transcripts_by_ids(
    transcripts: List[Dict[str, Any]],
    transcript_ids: List[str],
) -> List[Dict[str, Any]]:
    selected_ids = set(str(x) for x in transcript_ids)
    filtered = []

    for transcript in transcripts:
        if str(transcript["_resolved_transcript_id"]) in selected_ids:
            filtered.append(transcript)

    return filtered


# === EXAMPLE PREPARATION ===

def build_examples_for_transcripts(
    transcripts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    examples = prepare_examples(
        transcripts=transcripts,
        min_context_messages=2,
        min_user_turn_number=2,
        history_window=None,
        only_user_targets=True,
    )

    return examples


# === MANUAL SPLIT ===

def build_train_val_split_by_transcript(
    transcripts: List[Dict[str, Any]],
    train_transcript_ids: List[str],
    val_transcript_ids: List[str],
) -> Dict[str, Any]:
    train_transcripts = filter_transcripts_by_ids(
        transcripts=transcripts,
        transcript_ids=train_transcript_ids,
    )
    val_transcripts = filter_transcripts_by_ids(
        transcripts=transcripts,
        transcript_ids=val_transcript_ids,
    )

    train_examples = build_examples_for_transcripts(train_transcripts)
    val_examples = build_examples_for_transcripts(val_transcripts)

    return {
        "split_mode": "by_transcript",
        "train_transcript_ids": train_transcript_ids,
        "val_transcript_ids": val_transcript_ids,
        "train_examples": train_examples,
        "val_examples": val_examples,
    }


# === MANUAL EXPERIMENTS ===

EXPERIMENTS = [
    {
        "config_type": "baseline",
        "num_special_tokens": 0,
        "token_placement": "after_context",
        "position_mode": "default",
        "weight_decay": 0.0,
    },
    {
        "config_type": "token_1_after_default_wd0",
        "num_special_tokens": 1,
        "token_placement": "after_context",
        "position_mode": "default",
        "weight_decay": 0.0,
    },
    {
        "config_type": "token_1_before_default_wd0",
        "num_special_tokens": 1,
        "token_placement": "before_context",
        "position_mode": "default",
        "weight_decay": 0.0,
    },
    {
        "config_type": "token_1_after_shared_wd0",
        "num_special_tokens": 1,
        "token_placement": "after_context",
        "position_mode": "shared_position",
        "weight_decay": 0.0,
    },
    {
        "config_type": "token_1_after_default_wd1e4",
        "num_special_tokens": 1,
        "token_placement": "after_context",
        "position_mode": "default",
        "weight_decay": 1e-4,
    },
    {
        "config_type": "token_3_after_default_wd0",
        "num_special_tokens": 3,
        "token_placement": "after_context",
        "position_mode": "default",
        "weight_decay": 0.0,
    },
]


# === RESULT COMPACTION ===

def compact_result_row(
    experiment: Dict[str, Any],
    split_info: Dict[str, Any],
    train_config: TrainConfig,
    results: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "config_type": experiment["config_type"],
        "model_name": train_config.model_name,
        "split_mode": split_info["split_mode"],
        "train_transcript_ids": split_info["train_transcript_ids"],
        "val_transcript_ids": split_info["val_transcript_ids"],
        "num_special_tokens": train_config.num_special_tokens,
        "token_placement": train_config.token_placement,
        "position_mode": train_config.position_mode,
        "weight_decay": train_config.weight_decay,
        "batch_size": train_config.batch_size,
        "num_epochs": train_config.num_epochs,
        "learning_rate": train_config.learning_rate,
        "max_length": train_config.max_length,
        "train_size": results.get("n_train_examples"),
        "val_size": results.get("n_val_examples"),
        "best_val_loss": results.get("best_val_loss"),
        "final_val_loss": results.get("final_val_loss"),
        "val_mean_cosine": (
            results.get("val_cosine_metrics", {}).get("mean_cosine_similarity")
            if results.get("val_cosine_metrics") is not None else None
        ),
        "is_baseline": results.get("is_baseline", False),
    }

    return row


# === MAIN ===

def main():
    print(f"Loading transcripts from: {TRANSCRIPTS_PATH}")
    transcripts = load_jsonl(str(TRANSCRIPTS_PATH))
    transcripts = attach_transcript_ids(transcripts)

    if len(transcripts) == 0:
        raise ValueError("No transcripts found.")

    available_ids = [t["_resolved_transcript_id"] for t in transcripts]
    print(f"Found {len(transcripts)} transcripts")
    print("Available transcript IDs:")
    for transcript_id in available_ids:
        print(f"  - {transcript_id}")

    # === MANUAL TRAIN / VAL SPLIT ===
    # Edit these if you want a different split.
    train_transcript_ids = [
        available_ids[0],
        available_ids[1],
        available_ids[2],
        available_ids[3],
        available_ids[4],
        available_ids[5],
        available_ids[6],
        available_ids[7]
    ]
    val_transcript_ids = [
        available_ids[8],
        available_ids[9],
    ]

    split_info = build_train_val_split_by_transcript(
        transcripts=transcripts,
        train_transcript_ids=train_transcript_ids,
        val_transcript_ids=val_transcript_ids,
    )

    train_examples = split_info["train_examples"]
    val_examples = split_info["val_examples"]

    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples: {len(val_examples)}")

    if len(train_examples) == 0:
        raise ValueError("No training examples found.")
    if len(val_examples) == 0:
        raise ValueError("No validation examples found.")

    print(f"Running {len(EXPERIMENTS)} experiments")

    for exp_idx, experiment in enumerate(EXPERIMENTS, start=1):
        print("=" * 80)
        print(f"Experiment {exp_idx}/{len(EXPERIMENTS)}")
        print(f"config_type: {experiment['config_type']}")

        # override only some parameters for testing
        train_config = TrainConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            num_special_tokens=experiment["num_special_tokens"],
            token_placement=experiment["token_placement"],
            position_mode=experiment["position_mode"],
            weight_decay=experiment["weight_decay"],
            batch_size=1,
            num_epochs=3,
            learning_rate=5e-3,
            max_length=1024,
            eval_every_steps=20,
            generation_max_new_tokens=80,
            max_generation_examples=100,
            do_sample=False,
            use_fp16=True,
        )

        results = run_training(
            config=train_config,
            train_examples=train_examples,
            val_examples=val_examples,
            test_examples=[],
        )

        result_row = compact_result_row(
            experiment=experiment,
            split_info=split_info,
            train_config=train_config,
            results=results,
        )

        append_jsonl(result_row, RESULTS_PATH)

        print("Saved result row:")
        print(json.dumps(result_row, indent=2, ensure_ascii=False))

    print("=== All experiments completed. ===")


if __name__ == "__main__":
    main()