# src/transcripts_to_examples.py

# === IMPORTS ===

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# === IO ===

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} line {line_num}: {e}")
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# === TRANSCRIPT HELPERS ===

def get_transcript_id(transcript: Dict[str, Any], idx: int) -> str:
    for key in ["conversation_id", "transcript_id", "id"]:
        if transcript.get(key) is not None:
            return str(transcript[key])
    return f"transcript_{idx}"


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = msg.get("content")

        if role not in {"system", "user", "assistant"}:
            continue
        if content is None:
            continue

        content = str(content).strip()
        if not content:
            continue

        normalized.append(
            {
                "turn_index": msg_idx,
                "role": role,
                "content": content,
            }
        )

    return normalized


def count_user_turns(messages: List[Dict[str, Any]]) -> int:
    return sum(1 for msg in messages if msg["role"] == "user")


def assign_turn_bucket(user_turn_number: int, total_user_turns: int) -> str:
    if total_user_turns <= 0:
        return "unknown"

    ratio = user_turn_number / total_user_turns

    if ratio <= 1/3:
        return "early"
    if ratio <= 2/3:
        return "middle"
    return "late"


# === MAIN PREPARATION ===

def transcript_to_examples(
    transcript: Dict[str, Any],
    transcript_idx: int,
    min_context_messages: int = 2,
    min_user_turn_number: int = 2,
    history_window: Optional[int] = None,
    only_user_targets: bool = True,
) -> List[Dict[str, Any]]:
    transcript_id = get_transcript_id(transcript, transcript_idx)
    raw_messages = transcript.get("messages", [])
    messages = normalize_messages(raw_messages)

    total_user_turns = count_user_turns(messages)
    examples = []

    for target_idx, target_msg in enumerate(messages):
        if only_user_targets and target_msg["role"] != "user":
            continue

        if target_idx < min_context_messages:
            continue

        user_turn_number = sum(1 for m in messages[: target_idx + 1] if m["role"] == "user")
        assistant_turn_number = sum(1 for m in messages[: target_idx + 1] if m["role"] == "assistant")

        if user_turn_number < min_user_turn_number:
            continue

        if history_window is None:
            context_messages = messages[:target_idx]
        else:
            context_messages = messages[max(0, target_idx - history_window):target_idx]

        example = {
            "example_id": f"{transcript_id}__target_{target_idx}",
            "transcript_id": transcript_id,
            "target_turn_index": target_idx,
            "context_messages": context_messages,
            "target_message": target_msg["content"],
            "target_role": target_msg["role"],
            "context_length_messages": len(context_messages),
            "user_turn_number": user_turn_number,
            "assistant_turn_number": assistant_turn_number,
            "total_user_turns_in_transcript": total_user_turns,
            "user_turn_bucket": assign_turn_bucket(user_turn_number, total_user_turns),
            "persona_id": transcript.get("persona_id"),
            "topic_id": transcript.get("profile", {}).get("topic_id"),
            "profile": transcript.get("profile"),
            "metadata": {
                "seed_prompt": transcript.get("seed_prompt"),
                "experiment_index": transcript.get("experiment_index"),
                "replicate_index": transcript.get("replicate_index"),
                "user_model_name": transcript.get("user_model_name"),
                "assistant_model_name": transcript.get("assistant_model_name"),
            },
        }

        examples.append(example)

    return examples


def prepare_examples(
    transcripts: List[Dict[str, Any]],
    min_context_messages: int = 2,
    min_user_turn_number: int = 2,
    history_window: Optional[int] = None,
    only_user_targets: bool = True,
) -> List[Dict[str, Any]]:
    all_examples = []

    for transcript_idx, transcript in enumerate(transcripts):
        transcript_examples = transcript_to_examples(
            transcript=transcript,
            transcript_idx=transcript_idx,
            min_context_messages=min_context_messages,
            min_user_turn_number=min_user_turn_number,
            history_window=history_window,
            only_user_targets=only_user_targets,
        )
        all_examples.extend(transcript_examples)

    return all_examples


# === FILTERING ===

def filter_examples_by_transcript_ids(
    examples: List[Dict[str, Any]],
    transcript_ids: List[str],
) -> List[Dict[str, Any]]:
    transcript_id_set = {str(x) for x in transcript_ids}
    return [ex for ex in examples if str(ex["transcript_id"]) in transcript_id_set]


def select_single_transcript(
    examples: List[Dict[str, Any]],
    transcript_id: str,
) -> List[Dict[str, Any]]:
    return filter_examples_by_transcript_ids(examples, [transcript_id])


def get_available_transcript_ids(transcripts: List[Dict[str, Any]]) -> List[str]:
    return [get_transcript_id(transcript, idx) for idx, transcript in enumerate(transcripts)]