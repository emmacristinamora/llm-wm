import json
import argparse
from pathlib import Path
from collections import defaultdict


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_synthetic_user_id(conv: dict) -> str:
    profile = conv["profile"]
    return f"{profile['base_persona_id']}__{profile['style_id']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to conversation jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to grouped users json",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    grouped = defaultdict(list)

    for conv in read_jsonl(input_path):
        synthetic_user_id = make_synthetic_user_id(conv)
        grouped[synthetic_user_id].append({
            "conversation_id": conv["conversation_id"],
            "persona_id": conv["persona_id"],
            "base_persona_id": conv["profile"]["base_persona_id"],
            "style_id": conv["profile"]["style_id"],
            "experiment_idx": conv["profile"].get("experiment_idx"),
            "replicate_index": conv.get("replicate_index"),
        })

    grouped = dict(sorted(grouped.items(), key=lambda x: x[0]))
    write_json(output_path, grouped)

    print(f"Wrote grouped users to {output_path}")
    print(f"Number of synthetic users: {len(grouped)}")
    for user_id, convs in grouped.items():
        print(f"{user_id}: {len(convs)} conversations")


if __name__ == "__main__":
    main()