# special-token/transcripts_to_examples.py


# === IMPORTS ===

import json
from pathlib import Path
from typing import Any, Dict, List


# === PATHS ===

REPO_ROOT = Path(__file__).resolve().parent
TRANSCRIPTS_PATH = REPO_ROOT / "data" / "transcripts.jsonl"
EXAMPLES_PATH = REPO_ROOT / "data" / "examples.jsonl"


# === IO ===

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# === HELPERS ===

def get_transcript_id(transcript: Dict[str, Any], idx: int) -> str:
    if transcript.get("conversation_id") is not None:
        return str(transcript["conversation_id"])
    if transcript.get("transcript_id") is not None:
        return str(transcript["transcript_id"])
    if transcript.get("id") is not None:
        return str(transcript["id"])
    return f"transcript_{idx}"


def get_messages_field(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ["messages", "conversation", "chat", "turns"]:
        value = transcript.get(key)
        if isinstance(value, list):
            return value

    raise ValueError(
        "Could not find a list of messages in transcript. Expected one of: "
        "'messages', 'conversation', 'chat', 'turns'."
    )


def normalize_message(message: Dict[str, Any]) -> Dict[str, str]:
    role = message.get("role")
    content = message.get("content")

    if role is None:
        raise ValueError(f"Message missing role: {message}")
    if content is None:
        raise ValueError(f"Message missing content: {message}")

    return {
        "role": str(role).strip().lower(),
        "content": str(content),
    }


# === METADATA EXTRACTION ===

def extract_profile_fields(transcript: Dict[str, Any]) -> Dict[str, Any]:
    profile = transcript.get("profile", {}) if isinstance(transcript.get("profile"), dict) else {}

    return {
        "persona_id": transcript.get("persona_id"),
        "base_persona_id": (
            profile.get("base_persona_id")
            if profile.get("base_persona_id") is not None
            else transcript.get("base_persona_id")
        ),
        "style_id": (
            profile.get("style_id")
            if profile.get("style_id") is not None
            else transcript.get("style_id")
        ),
        "topic_id": (
            profile.get("topic_id")
            if profile.get("topic_id") is not None
            else transcript.get("topic_id")
        ),
        "init_idx": (
            profile.get("init_idx")
            if profile.get("init_idx") is not None
            else transcript.get("init_idx")
        ),
    }


# === EXAMPLE BUILDING ===

def prepare_examples(
    transcripts: List[Dict[str, Any]],
    min_context_messages: int = 4,
    min_user_turn_number: int = 3,
    history_window: int = None,
    only_user_targets: bool = True,
    ) -> List[Dict[str, Any]]:
    """
    Convert transcripts to examples for model training. 
    We consider an example to be a single target message and its preceding context messages.
    Args:
    - transcripts: List of transcript dictionaries.
    - min_context_messages: Minimum number of messages in the context for an example to be created (set at 2)
    - min_user_turn_number: Minimum number of user turns that must have occurred in the transcript before we start creating examples.
    - history_window: If set, limits the number of context messages to the most recent N
    - only_user_targets: If True, only create examples where the target message is from the user.
    """
    examples: List[Dict[str, Any]] = []

    for transcript_idx, transcript in enumerate(transcripts):
        transcript_id = get_transcript_id(transcript, transcript_idx)
        metadata = extract_profile_fields(transcript)
        raw_messages = get_messages_field(transcript)
        messages = [normalize_message(message) for message in raw_messages]

        user_turn_number = 0

        for message_idx, message in enumerate(messages):
            role = message["role"]

            if only_user_targets and role != "user":
                continue

            if role == "user":
                user_turn_number += 1

            context_messages = messages[:message_idx]
            target_message = message["content"].strip()

            if len(context_messages) < min_context_messages:
                continue

            if role == "user" and user_turn_number < min_user_turn_number:
                continue

            if history_window is not None:
                context_messages = context_messages[-history_window:]

            example = {
                "example_id": f"{transcript_id}__msg_{message_idx}",
                "transcript_id": transcript_id,
                "target_message_index": message_idx,
                "target_role": role,
                "user_turn_number": user_turn_number if role == "user" else None,
                "context_messages": context_messages,
                "target_message": target_message,
                **metadata,
            }

            examples.append(example)

    return examples


# === MAIN ===

def main() -> None:
    print(f"Loading transcripts from: {TRANSCRIPTS_PATH}")
    transcripts = load_jsonl(TRANSCRIPTS_PATH)

    if len(transcripts) == 0:
        raise ValueError("No transcripts found.")

    examples = prepare_examples(
        transcripts=transcripts,
        min_context_messages=4,
        min_user_turn_number=3,
        history_window=None,
        only_user_targets=True,
    )

    if len(examples) == 0:
        raise ValueError("No examples were created.")

    write_jsonl(examples, EXAMPLES_PATH)

    print(f"Loaded transcripts: {len(transcripts)}")
    print(f"Created examples: {len(examples)}")
    print(f"Saved to: {EXAMPLES_PATH}")

    print("Sample example:")
    print(json.dumps(examples[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
