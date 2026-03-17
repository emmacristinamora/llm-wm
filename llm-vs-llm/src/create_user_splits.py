import json
import argparse
import random
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouped_users_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of conversations per user for test",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    grouped = read_json(Path(args.grouped_users_path))
    out = {}

    for synthetic_user_id, convs in grouped.items():
        convs = list(convs)
        rng.shuffle(convs)

        n = len(convs)
        n_test = max(1, round(n * args.test_size)) if n >= 2 else 0
        n_test = min(n_test, n - 1) if n >= 2 else 0

        test_convs = convs[:n_test]
        train_convs = convs[n_test:]

        out[synthetic_user_id] = {
            "train": train_convs,
            "test": test_convs,
        }

    write_json(Path(args.output_path), out)

    print(f"Wrote user splits to {args.output_path}")
    for user_id, split in out.items():
        print(
            f"{user_id}: train={len(split['train'])}, test={len(split['test'])}"
        )


if __name__ == "__main__":
    main()