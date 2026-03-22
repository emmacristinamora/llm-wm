# special-token/evaluate_special_token.py

# === IMPORTS ===

import argparse
import json
import random
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# === CONFIG DATACLASS ===

@dataclass
class EvalConfig:
    # paths
    repo_root: str = "."
    examples_path: str = "data/examples.jsonl"
    runs_root: str = "data/runs"
    evals_root: str = "data/evals"
    run_name: str = ""

    # evaluation-time generation config
    generation_max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0

    # sentence similarity model
    sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # compute / batching
    max_length: int = 1024
    batch_size: int = 1
    max_examples_per_bucket: Optional[int] = None

    # reproducibility / precision
    seed: int = 42
    use_fp16: bool = False
    use_bf16: bool = False

    # saving
    save_per_example: bool = True


# === REPRODUCIBILITY HELPERS ===

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# === IO HELPERS ===

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return rows


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    sanitized_obj = sanitize_for_json(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitized_obj, f, indent=2, ensure_ascii=False)


def append_jsonl(row: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]

    return obj


def build_run_dir(config: EvalConfig) -> Path:
    if not config.run_name:
        raise ValueError("config.run_name is empty.")

    return Path(config.repo_root) / config.runs_root / config.run_name


def build_eval_dir(config: EvalConfig) -> Path:
    if not config.run_name:
        raise ValueError("config.run_name is empty.")

    return Path(config.repo_root) / config.evals_root / config.run_name


def load_training_artifacts(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    run_summary_path = run_dir / "run_summary.json"
    embedding_path = run_dir / "special_token_embeddings.pt"

    if not run_summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {run_summary_path}")

    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing learned embedding file: {embedding_path}")

    run_summary = load_json(run_summary_path)
    embedding_artifact = torch.load(embedding_path, map_location="cpu")

    return run_summary, embedding_artifact


# === EXAMPLE SCHEMA HELPERS ===

def validate_example_schema(example: Dict[str, Any]) -> None:
    required_keys = [
        "example_id",
        "transcript_id",
        "context_messages",
        "target_message",
        "base_persona_id",
        "style_id",
        "topic_id",
    ]

    missing = [key for key in required_keys if key not in example]
    if missing:
        raise ValueError(f"Example is missing required keys: {missing}")


# === PROMPT / FORMATTING HELPERS ===

def format_message(role: str, content: str) -> str:
    role = role.lower().strip()

    if role == "system":
        prefix = "System"
    elif role == "user":
        prefix = "User"
    elif role == "assistant":
        prefix = "Assistant"
    else:
        prefix = role.capitalize()

    return f"{prefix}: {content.strip()}"


def render_context_messages(messages: List[Dict[str, str]]) -> str:
    """
    Important:
    This mirrors training exactly. We render ONLY context_messages.
    We do NOT insert any separate system prompt here.
    """
    rendered: List[str] = []

    for message in messages:
        role = message["role"]
        content = message["content"]
        rendered.append(format_message(role=role, content=content))

    return "\n".join(rendered).strip()


def make_special_tokens(base: str, n: int) -> List[str]:
    if n <= 0:
        return []

    if n == 1:
        return [base]

    return [f"{base}{i}" for i in range(n)]


def build_prompt_text(
    example: Dict[str, Any],
    special_tokens: List[str],
    token_placement: str,
) -> str:
    """
    This must stay aligned with training.

    Current logic:
    - render example["context_messages"]
    - insert learned special token text either before or after context
    - end with 'User:'

    No explicit system prompt is added here.
    """
    context_text = render_context_messages(example["context_messages"])
    special_text = " ".join(special_tokens).strip()

    if len(special_tokens) == 0:
        conditioned_context = context_text
    else:
        if token_placement == "before_context":
            conditioned_context = f"{special_text}\n{context_text}".strip()
        elif token_placement == "after_context":
            conditioned_context = f"{context_text}\n{special_text}".strip()
        else:
            raise ValueError(f"Unsupported token_placement: {token_placement}")

    prompt_text = f"{conditioned_context}\nUser:"
    return prompt_text


def build_full_text(
    example: Dict[str, Any],
    special_tokens: List[str],
    token_placement: str,
) -> str:
    prompt_text = build_prompt_text(
        example=example,
        special_tokens=special_tokens,
        token_placement=token_placement,
    )
    target_text = example["target_message"].strip()
    return f"{prompt_text} {target_text}"


# === TENSORIZATION HELPERS ===

def build_training_example_tensors(
    tokenizer,
    example: Dict[str, Any],
    special_tokens: List[str],
    token_placement: str,
    max_length: int,
) -> Optional[Dict[str, Any]]:
    """
    Build one teacher-forced evaluation example.

    Same logic as training:
    - tokenize prompt and target separately
    - preserve target under truncation
    - supervise only the target tokens
    """
    prompt_text = build_prompt_text(
        example=example,
        special_tokens=special_tokens,
        token_placement=token_placement,
    )
    target_text = " " + example["target_message"].strip()

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    if len(target_ids) == 0:
        return None

    bos_ids: List[int] = []
    if tokenizer.bos_token_id is not None:
        bos_ids = [tokenizer.bos_token_id]

    available_for_prompt = max_length - len(bos_ids) - len(target_ids)

    if available_for_prompt < 0:
        return None

    if len(prompt_ids) > available_for_prompt:
        prompt_ids = prompt_ids[-available_for_prompt:] if available_for_prompt > 0 else []

    input_ids_list = bos_ids + prompt_ids + target_ids
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()
    prompt_token_count = len(bos_ids) + len(prompt_ids)
    labels[:prompt_token_count] = -100

    if (labels != -100).sum().item() == 0:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "metadata": {
            "example_id": example.get("example_id"),
            "transcript_id": example.get("transcript_id"),
            "topic_id": example.get("topic_id"),
            "base_persona_id": example.get("base_persona_id"),
            "style_id": example.get("style_id"),
            "user_turn_number": example.get("user_turn_number"),
            "target_message_index": example.get("target_message_index"),
            "target_message": example.get("target_message"),
            "prompt_text": prompt_text,
        },
        "raw_example": example,
    }


# === SPLIT / BUCKET HELPERS ===

def filter_examples(
    examples: List[Dict[str, Any]],
    base_persona_id: Optional[str] = None,
    style_id: Optional[str] = None,
    topic_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []

    for example in examples:
        validate_example_schema(example)

        if base_persona_id is not None and str(example["base_persona_id"]) != str(base_persona_id):
            continue
        if style_id is not None and str(example["style_id"]) != str(style_id):
            continue
        if topic_id is not None and str(example["topic_id"]) != str(topic_id):
            continue

        filtered.append(example)

    return filtered


def build_evaluation_buckets(
    examples: List[Dict[str, Any]],
    target_base_persona_id: str,
    target_style_id: str,
    held_out_topic_id: str,
    max_examples_per_bucket: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build the four evaluation conditions for one trained embedding.

    matched:
        same persona, same style, same held-out topic

    same_persona_diff_style:
        same persona, different style, same held-out topic

    diff_persona_same_style:
        different persona, same style, same held-out topic

    diff_persona_diff_style:
        different persona, different style, same held-out topic
    """
    topic_examples = filter_examples(
        examples=examples,
        topic_id=held_out_topic_id,
    )

    matched: List[Dict[str, Any]] = []
    same_persona_diff_style: List[Dict[str, Any]] = []
    diff_persona_same_style: List[Dict[str, Any]] = []
    diff_persona_diff_style: List[Dict[str, Any]] = []

    for example in topic_examples:
        persona = str(example["base_persona_id"])
        style = str(example["style_id"])

        same_persona = persona == str(target_base_persona_id)
        same_style = style == str(target_style_id)

        if same_persona and same_style:
            matched.append(example)
        elif same_persona and not same_style:
            same_persona_diff_style.append(example)
        elif not same_persona and same_style:
            diff_persona_same_style.append(example)
        else:
            diff_persona_diff_style.append(example)

    if max_examples_per_bucket is not None:
        matched = matched[:max_examples_per_bucket]
        same_persona_diff_style = same_persona_diff_style[:max_examples_per_bucket]
        diff_persona_same_style = diff_persona_same_style[:max_examples_per_bucket]
        diff_persona_diff_style = diff_persona_diff_style[:max_examples_per_bucket]

    return {
        "matched": matched,
        "same_persona_diff_style": same_persona_diff_style,
        "diff_persona_same_style": diff_persona_same_style,
        "diff_persona_diff_style": diff_persona_diff_style,
    }


# === DATASET / COLLATE ===

class NextUserTurnDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer,
        special_tokens: List[str],
        token_placement: str,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.token_placement = token_placement
        self.max_length = max_length

        processed_items: List[Dict[str, Any]] = []
        dropped_examples = 0

        for example in examples:
            item = build_training_example_tensors(
                tokenizer=tokenizer,
                example=example,
                special_tokens=special_tokens,
                token_placement=token_placement,
                max_length=max_length,
            )
            if item is None:
                dropped_examples += 1
            else:
                processed_items.append(item)

        self.items = processed_items
        self.dropped_examples = dropped_examples

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate_batch(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("Received empty batch in collate_batch.")

    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids_list: List[torch.Tensor] = []
    attention_mask_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    metadata_list: List[Dict[str, Any]] = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids = torch.cat(
            [
                item["input_ids"],
                torch.full((pad_len,), pad_token_id, dtype=torch.long),
            ],
            dim=0,
        )
        attention_mask = torch.cat(
            [
                item["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long),
            ],
            dim=0,
        )
        labels = torch.cat(
            [
                item["labels"],
                torch.full((pad_len,), -100, dtype=torch.long),
            ],
            dim=0,
        )

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
        metadata_list.append(item["metadata"])

    labels_tensor = torch.stack(labels_list, dim=0)
    if (labels_tensor != -100).sum().item() == 0:
        raise ValueError("Batch has no supervised tokens after padding/masking.")

    return {
        "input_ids": torch.stack(input_ids_list, dim=0),
        "attention_mask": torch.stack(attention_mask_list, dim=0),
        "labels": labels_tensor,
        "metadata": metadata_list,
    }


# === MODEL HELPERS ===

def prepare_model_and_tokenizer_from_train_config(
    train_config: Dict[str, Any],
    eval_config: EvalConfig,
):
    model_name = train_config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if eval_config.use_bf16:
        model_dtype = torch.bfloat16
    elif eval_config.use_fp16:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_dtype,
    )

    special_tokens = make_special_tokens(
        base=train_config["special_token_base"],
        n=int(train_config["num_special_tokens"]),
    )

    if len(special_tokens) > 0:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        model.resize_token_embeddings(len(tokenizer))
        special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    else:
        special_token_ids = []

    return model, tokenizer, special_tokens, special_token_ids


def load_learned_special_token_embeddings(
    embedding_artifact: Dict[str, Any],
) -> torch.Tensor:
    if "embeddings" not in embedding_artifact:
        raise ValueError("Embedding artifact does not contain 'embeddings'.")

    learned_rows = embedding_artifact["embeddings"]

    if not isinstance(learned_rows, torch.Tensor):
        raise ValueError("Loaded learned embeddings are not a torch.Tensor.")

    return learned_rows


def inject_special_token_embeddings(
    model,
    special_token_ids: List[int],
    learned_rows: torch.Tensor,
) -> None:
    if len(special_token_ids) == 0:
        return

    if learned_rows.shape[0] != len(special_token_ids):
        raise ValueError(
            f"Mismatch between number of learned rows ({learned_rows.shape[0]}) "
            f"and number of special token ids ({len(special_token_ids)})."
        )

    embedding_weight = model.get_input_embeddings().weight

    with torch.no_grad():
        for i, token_id in enumerate(special_token_ids):
            embedding_weight[token_id].copy_(learned_rows[i].to(embedding_weight.device, dtype=embedding_weight.dtype))


def build_position_ids_with_shared_special_tokens(
    input_ids: torch.Tensor,
    special_token_ids: List[int],
) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    position_ids = (
        torch.arange(seq_len, device=input_ids.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .clone()
    )

    if len(special_token_ids) == 0:
        return position_ids

    special_token_id_set = set(int(x) for x in special_token_ids)

    for batch_idx in range(batch_size):
        shared_position = None

        for token_idx in range(seq_len):
            token_id = int(input_ids[batch_idx, token_idx].item())

            if token_id in special_token_id_set:
                if shared_position is None:
                    shared_position = int(position_ids[batch_idx, token_idx].item())
                position_ids[batch_idx, token_idx] = shared_position

    return position_ids


def build_forward_kwargs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor],
    position_mode: str,
    special_token_ids: List[int],
) -> Dict[str, torch.Tensor]:
    forward_kwargs: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    if labels is not None:
        forward_kwargs["labels"] = labels

    if position_mode == "shared_position":
        forward_kwargs["position_ids"] = build_position_ids_with_shared_special_tokens(
            input_ids=input_ids,
            special_token_ids=special_token_ids,
        )
    elif position_mode != "default":
        raise ValueError(f"Unsupported position_mode: {position_mode}")

    return forward_kwargs


# === CLI ===

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--examples_path", type=str, default="data/examples.jsonl")
    parser.add_argument("--runs_root", type=str, default="data/runs")
    parser.add_argument("--evals_root", type=str, default="data/evals")
    parser.add_argument("--run_name", type=str, required=True)

    parser.add_argument("--generation_max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument(
        "--sentence_model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_examples_per_bucket", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--save_per_example", action="store_true")

    return parser.parse_args()


# === PROBABILISTIC EVAL HELPERS ===

@torch.no_grad()
def compute_example_loss(
    model,
    batch: Dict[str, Any],
    device: torch.device,
    position_mode: str,
    special_token_ids: List[int],
) -> float:
    model.eval()

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    forward_kwargs = build_forward_kwargs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        position_mode=position_mode,
        special_token_ids=special_token_ids,
    )

    outputs = model(**forward_kwargs)
    loss = outputs.loss

    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError("NaN/Inf evaluation loss encountered.")

    return float(loss.item())


@torch.no_grad()
def run_bucket_loss(
    model,
    dataset: NextUserTurnDataset,
    tokenizer,
    device: torch.device,
    position_mode: str,
    special_token_ids: List[int],
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compute teacher-forced loss example by example so we can keep exact per-example rows.
    """
    if len(dataset) == 0:
        return float("nan"), []

    per_example_rows: List[Dict[str, Any]] = []
    total_loss = 0.0

    for item in dataset.items:
        batch = collate_batch([item], tokenizer.pad_token_id)
        example_loss = compute_example_loss(
            model=model,
            batch=batch,
            device=device,
            position_mode=position_mode,
            special_token_ids=special_token_ids,
        )

        metadata = item["metadata"]
        row = {
            "example_id": metadata.get("example_id"),
            "transcript_id": metadata.get("transcript_id"),
            "topic_id": metadata.get("topic_id"),
            "base_persona_id": metadata.get("base_persona_id"),
            "style_id": metadata.get("style_id"),
            "user_turn_number": metadata.get("user_turn_number"),
            "target_message_index": metadata.get("target_message_index"),
            "teacher_forced_loss": example_loss,
        }
        per_example_rows.append(row)
        total_loss += example_loss

    mean_loss = total_loss / len(per_example_rows)
    return mean_loss, per_example_rows


# === GENERATION HELPERS ===

@torch.no_grad()
def generate_next_user_turn(
    model,
    tokenizer,
    example: Dict[str, Any],
    special_tokens: List[str],
    token_placement: str,
    position_mode: str,
    special_token_ids: List[int],
    generation_max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> str:
    """
    Important:
    This mirrors training-time prompt construction:
    - ONLY context_messages
    - NO explicit external system prompt
    - special tokens inserted before/after context
    - trailing 'User:'
    """
    prompt_text = build_prompt_text(
        example=example,
        special_tokens=special_tokens,
        token_placement=token_placement,
    )

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    bos_ids: List[int] = []
    if tokenizer.bos_token_id is not None:
        bos_ids = [tokenizer.bos_token_id]

    input_ids = torch.tensor(
        [bos_ids + prompt_ids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)

    gen_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": generation_max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    if position_mode == "shared_position":
        gen_kwargs["position_ids"] = build_position_ids_with_shared_special_tokens(
            input_ids=input_ids,
            special_token_ids=special_token_ids,
        )

    out = model.generate(**gen_kwargs)
    generated_ids = out[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return generated_text


def load_sentence_similarity_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@torch.no_grad()
def compute_sentence_cosine_similarity(
    sentence_model,
    text_a: str,
    text_b: str,
) -> float:
    import torch.nn.functional as F

    embeddings = sentence_model.encode(
        [text_a, text_b],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    sim = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    return float(sim.item())


def evaluate_generation_bucket(
    model,
    tokenizer,
    examples: List[Dict[str, Any]],
    special_tokens: List[str],
    token_placement: str,
    position_mode: str,
    special_token_ids: List[int],
    generation_max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    sentence_model,
    device: torch.device,
) -> Tuple[float, List[Dict[str, Any]]]:
    if len(examples) == 0:
        return float("nan"), []

    per_example_rows: List[Dict[str, Any]] = []
    total_cosine = 0.0

    for example in examples:
        generated_text = generate_next_user_turn(
            model=model,
            tokenizer=tokenizer,
            example=example,
            special_tokens=special_tokens,
            token_placement=token_placement,
            position_mode=position_mode,
            special_token_ids=special_token_ids,
            generation_max_new_tokens=generation_max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )

        gold_text = str(example["target_message"]).strip()
        cosine_similarity = compute_sentence_cosine_similarity(
            sentence_model=sentence_model,
            text_a=generated_text,
            text_b=gold_text,
        )

        row = {
            "example_id": example.get("example_id"),
            "transcript_id": example.get("transcript_id"),
            "topic_id": example.get("topic_id"),
            "base_persona_id": example.get("base_persona_id"),
            "style_id": example.get("style_id"),
            "user_turn_number": example.get("user_turn_number"),
            "target_message_index": example.get("target_message_index"),
            "generated_text": generated_text,
            "gold_text": gold_text,
            "generation_cosine_similarity": cosine_similarity,
            "exact_match": int(generated_text.strip() == gold_text.strip()),
        }
        per_example_rows.append(row)
        total_cosine += cosine_similarity

    mean_cosine = total_cosine / len(per_example_rows)
    return mean_cosine, per_example_rows


# === BUCKET EVALUATOR ===

def evaluate_one_bucket(
    bucket_name: str,
    bucket_examples: List[Dict[str, Any]],
    model,
    tokenizer,
    special_tokens: List[str],
    token_placement: str,
    position_mode: str,
    special_token_ids: List[int],
    sentence_model,
    eval_config: EvalConfig,
    device: torch.device,
) -> Dict[str, Any]:
    dataset = NextUserTurnDataset(
        examples=bucket_examples,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        token_placement=token_placement,
        max_length=eval_config.max_length,
    )

    mean_loss, loss_rows = run_bucket_loss(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        position_mode=position_mode,
        special_token_ids=special_token_ids,
    )

    kept_examples_for_generation = [item["raw_example"] for item in dataset.items]

    mean_cosine, generation_rows = evaluate_generation_bucket(
        model=model,
        tokenizer=tokenizer,
        examples=kept_examples_for_generation,
        special_tokens=special_tokens,
        token_placement=token_placement,
        position_mode=position_mode,
        special_token_ids=special_token_ids,
        generation_max_new_tokens=eval_config.generation_max_new_tokens,
        do_sample=eval_config.do_sample,
        temperature=eval_config.temperature,
        top_p=eval_config.top_p,
        sentence_model=sentence_model,
        device=device,
    )

    generation_by_id = {row["example_id"]: row for row in generation_rows}
    per_example_rows: List[Dict[str, Any]] = []

    for loss_row in loss_rows:
        example_id = loss_row["example_id"]
        merged = dict(loss_row)
        if example_id in generation_by_id:
            merged.update(
                {
                    "generated_text": generation_by_id[example_id]["generated_text"],
                    "gold_text": generation_by_id[example_id]["gold_text"],
                    "generation_cosine_similarity": generation_by_id[example_id]["generation_cosine_similarity"],
                    "exact_match": generation_by_id[example_id]["exact_match"],
                }
            )
        per_example_rows.append(merged)

    rows_with_exact_match = [
        row for row in per_example_rows
        if "exact_match" in row
    ]

    exact_match_rate = (
        sum(int(row["exact_match"]) for row in rows_with_exact_match) / len(rows_with_exact_match)
        if len(rows_with_exact_match) > 0 else float("nan")
    )

    return {
        "bucket_name": bucket_name,
        "n_examples_raw": len(bucket_examples),
        "n_examples_used": len(per_example_rows),
        "n_examples_dropped": dataset.dropped_examples,
        "mean_teacher_forced_loss": mean_loss,
        "mean_generation_cosine_similarity": mean_cosine,
        "exact_match_rate": exact_match_rate,
        "per_example_rows": per_example_rows,
    }


# === MAIN EVALUATION ===

def compute_bucket_deltas(bucket_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    deltas: Dict[str, Any] = {}

    matched = bucket_results["matched"]

    control_bucket_names = [
        "same_persona_diff_style",
        "diff_persona_same_style",
        "diff_persona_diff_style",
    ]

    for control_name in control_bucket_names:
        control = bucket_results[control_name]

        deltas[f"loss_delta_matched_minus_{control_name}"] = (
            matched["mean_teacher_forced_loss"] - control["mean_teacher_forced_loss"]
        )
        deltas[f"cosine_delta_matched_minus_{control_name}"] = (
            matched["mean_generation_cosine_similarity"] - control["mean_generation_cosine_similarity"]
        )
        deltas[f"exact_match_delta_matched_minus_{control_name}"] = (
            matched["exact_match_rate"] - control["exact_match_rate"]
        )

    return deltas


def run_evaluation(config: EvalConfig) -> Dict[str, Any]:
    set_seed(config.seed)

    if not config.run_name:
        raise ValueError("run_name must be provided for evaluation.")

    run_dir = build_run_dir(config)
    eval_dir = build_eval_dir(config)
    eval_dir.mkdir(parents=True, exist_ok=True)

    run_summary, embedding_artifact = load_training_artifacts(run_dir)
    train_config = run_summary["config"]

    examples_path = Path(config.repo_root) / config.examples_path
    examples = load_jsonl(examples_path)

    target_base_persona_id = str(train_config["base_persona_id"])
    target_style_id = str(train_config["style_id"])
    held_out_topic_id = str(train_config["held_out_topic_id"])
    token_placement = str(train_config["token_placement"])
    position_mode = str(train_config["position_mode"])

    model, tokenizer, special_tokens, special_token_ids = prepare_model_and_tokenizer_from_train_config(
        train_config=train_config,
        eval_config=config,
    )

    learned_rows = load_learned_special_token_embeddings(embedding_artifact=embedding_artifact)
    inject_special_token_embeddings(
        model=model,
        special_token_ids=special_token_ids,
        learned_rows=learned_rows,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    sentence_model = load_sentence_similarity_model(config.sentence_model_name)

    buckets = build_evaluation_buckets(
        examples=examples,
        target_base_persona_id=target_base_persona_id,
        target_style_id=target_style_id,
        held_out_topic_id=held_out_topic_id,
        max_examples_per_bucket=config.max_examples_per_bucket,
    )

    bucket_results: Dict[str, Dict[str, Any]] = {}
    all_per_example_rows: List[Dict[str, Any]] = []

    for bucket_name, bucket_examples in buckets.items():
        print(f"[eval] run={config.run_name} | bucket={bucket_name} | n_raw={len(bucket_examples)}")

        bucket_result = evaluate_one_bucket(
            bucket_name=bucket_name,
            bucket_examples=bucket_examples,
            model=model,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            token_placement=token_placement,
            position_mode=position_mode,
            special_token_ids=special_token_ids,
            sentence_model=sentence_model,
            eval_config=config,
            device=device,
        )

        bucket_results[bucket_name] = bucket_result

        for row in bucket_result["per_example_rows"]:
            enriched_row = dict(row)
            enriched_row["run_name"] = config.run_name
            enriched_row["bucket_name"] = bucket_name
            enriched_row["trained_base_persona_id"] = target_base_persona_id
            enriched_row["trained_style_id"] = target_style_id
            enriched_row["trained_held_out_topic_id"] = held_out_topic_id
            enriched_row["num_special_tokens"] = int(train_config["num_special_tokens"])
            enriched_row["token_placement"] = token_placement
            enriched_row["position_mode"] = position_mode
            all_per_example_rows.append(enriched_row)

        print(
            f"[eval] bucket={bucket_name} "
            f"| n_used={bucket_result['n_examples_used']} "
            f"| mean_loss={bucket_result['mean_teacher_forced_loss']:.6f} "
            f"| mean_cosine={bucket_result['mean_generation_cosine_similarity']:.6f} "
            f"| exact_match={bucket_result['exact_match_rate']:.6f}"
        )

    deltas = compute_bucket_deltas(bucket_results)

    summary = {
        "eval_config": asdict(config),
        "run_name": config.run_name,
        "train_config": train_config,
        "special_tokens": special_tokens,
        "special_token_ids": special_token_ids,
        "held_out_topic_id": held_out_topic_id,
        "trained_base_persona_id": target_base_persona_id,
        "trained_style_id": target_style_id,
        "bucket_summaries": {
            bucket_name: {
                "bucket_name": bucket_result["bucket_name"],
                "n_examples_raw": bucket_result["n_examples_raw"],
                "n_examples_used": bucket_result["n_examples_used"],
                "n_examples_dropped": bucket_result["n_examples_dropped"],
                "mean_teacher_forced_loss": bucket_result["mean_teacher_forced_loss"],
                "mean_generation_cosine_similarity": bucket_result["mean_generation_cosine_similarity"],
                "exact_match_rate": bucket_result["exact_match_rate"],
            }
            for bucket_name, bucket_result in bucket_results.items()
        },
        "matched_vs_control_deltas": deltas,
    }

    save_json(summary, eval_dir / "eval_summary.json")

    if config.save_per_example:
        per_example_path = eval_dir / "per_example_results.jsonl"
        for row in all_per_example_rows:
            append_jsonl(row, per_example_path)

    return summary


# === CLI MAIN ===

def main() -> None:
    args = parse_args()

    if args.use_fp16 and args.use_bf16:
        raise ValueError("Use at most one of --use_fp16 and --use_bf16.")

    config = EvalConfig(
        repo_root=args.repo_root,
        examples_path=args.examples_path,
        runs_root=args.runs_root,
        evals_root=args.evals_root,
        run_name=args.run_name,
        generation_max_new_tokens=args.generation_max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        sentence_model_name=args.sentence_model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_examples_per_bucket=args.max_examples_per_bucket,
        seed=args.seed,
        use_fp16=args.use_fp16,
        use_bf16=args.use_bf16,
        save_per_example=args.save_per_example,
    )

    print("=" * 80)
    print("=== RUNNING SPECIAL TOKEN EVALUATION ===")
    print(json.dumps(asdict(config), indent=2, ensure_ascii=False))

    summary = run_evaluation(config)

    print("=== EVALUATION COMPLETED ===")
    print(json.dumps(summary["bucket_summaries"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["matched_vs_control_deltas"], indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()