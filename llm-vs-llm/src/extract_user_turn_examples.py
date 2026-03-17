import json
import argparse
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_synthetic_user_id(conv: dict) -> str:
    profile = conv["profile"]
    return f"{profile['base_persona_id']}__{profile['style_id']}"


def extract_examples_from_conversation(conv: dict):
    """
    Extract examples where target is a user turn, excluding the first seed user turn.
    Assumes alternating user/assistant messages starting with user.
    """
    messages = conv["messages"]
    examples = []

    for idx, msg in enumerate(messages):
        if msg["role"] != "user":
            continue

        # skip first user turn (seed prompt)
        if idx == 0:
            continue

        context_messages = messages[:idx]
        target_message = msg["content"]

        examples.append({
            "conversation_id": conv["conversation_id"],
            "persona_id": conv["persona_id"],
            "synthetic_user_id": make_synthetic_user_id(conv),
            "base_persona_id": conv["profile"]["base_persona_id"],
            "style_id": conv["profile"]["style_id"],
            "target_turn_index": idx,
            "context_messages": context_messages,
            "target_message": target_message,
        })

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations_path", type=str, required=True)
    parser.add_argument("--splits_path", type=str, required=True)
    parser.add_argument("--output_train_path", type=str, required=True)
    parser.add_argument("--output_test_path", type=str, required=True)
    args = parser.parse_args()

    splits = read_json(Path(args.splits_path))
    convs_by_id = {}

    for conv in read_jsonl(Path(args.conversations_path)):
        convs_by_id[conv["conversation_id"]] = conv

    train_rows = []
    test_rows = []

    for synthetic_user_id, split in splits.items():
        for item in split["train"]:
            conv = convs_by_id[item["conversation_id"]]
            train_rows.extend(extract_examples_from_conversation(conv))

        for item in split["test"]:
            conv = convs_by_id[item["conversation_id"]]
            test_rows.extend(extract_examples_from_conversation(conv))

    write_jsonl(Path(args.output_train_path), train_rows)
    write_jsonl(Path(args.output_test_path), test_rows)

    print(f"Wrote {len(train_rows)} train examples -> {args.output_train_path}")
    print(f"Wrote {len(test_rows)} test examples -> {args.output_test_path}")


if __name__ == "__main__":
    main()