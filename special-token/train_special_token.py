# special-token/train_special_token.py

# === IMPORTS ===

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# === CONFIG DATACLASS ===

@dataclass
class TrainConfig:
    # paths
    repo_root: str = "."
    examples_path: str = "data/examples.jsonl"
    runs_root: str = "data/runs"
    run_name: str = ""

    # macro experiment config
    base_persona_id: str = ""
    style_id: str = ""
    held_out_topic_id: str = ""

    # micro experiment config
    num_special_tokens: int = 1
    special_token_base: str = "<st>"
    token_placement: str = "after_context"      # before_context | after_context
    position_mode: str = "default"              # default | shared_position
    default_chat_template: bool = True
    use_examples_percentage: float = 1.0

    # model / optimization
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_length: int = 1024
    batch_size: int = 1
    num_epochs: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    grad_accum_steps: int = 1
    max_grad_norm: float = 0.5
    eval_every_steps: int = 20
    save_per_epoch: bool = False # Whether to save the embedding checkpoint after every epoch

    # reproducibility / precision
    seed: int = 42
    use_fp16: bool = False
    use_bf16: bool = False


# === REPRODUCIBILITY HELPERS ===

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# === IO HELPERS ===

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

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(row: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_run_name(config: TrainConfig) -> str:
    # import datetime
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{config.base_persona_id}"
        f"__{config.style_id}"
        f"__heldout_{config.held_out_topic_id}"
        f"__tok{config.num_special_tokens}"
        f"__{config.num_epochs}ep"
        f"__{config.token_placement}"
        f"__{config.position_mode}"
        f"__{'template' if config.default_chat_template else 'notemplate'}"
        f"__useexamples{int(config.use_examples_percentage * 100)}"
    #    f"__{timestamp}"
    )


def build_run_dir(config: TrainConfig) -> Path:
    if not config.run_name:
        raise ValueError("config.run_name is empty. Set it once before building run_dir.")

    repo_root = Path(config.repo_root)
    runs_root = repo_root / config.runs_root
    return runs_root / config.run_name


# === SPLIT HELPERS ===

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


def filter_examples_for_bucket(
    examples: List[Dict[str, Any]],
    base_persona_id: str,
    style_id: str,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []

    for example in examples:
        validate_example_schema(example)

        if str(example["base_persona_id"]) != str(base_persona_id):
            continue
        if str(example["style_id"]) != str(style_id):
            continue

        filtered.append(example)

    return filtered


def build_leave_one_topic_out_split(
    examples: List[Dict[str, Any]],
    base_persona_id: str,
    style_id: str,
    held_out_topic_id: str,
) -> Dict[str, Any]:
    bucket_examples = filter_examples_for_bucket(
        examples=examples,
        base_persona_id=base_persona_id,
        style_id=style_id,
    )

    train_examples: List[Dict[str, Any]] = []
    val_examples: List[Dict[str, Any]] = []

    for example in bucket_examples:
        topic_id = str(example["topic_id"])

        if topic_id == str(held_out_topic_id):
            val_examples.append(example)
        else:
            train_examples.append(example)

    train_topic_ids = sorted({str(x["topic_id"]) for x in train_examples})
    val_topic_ids = sorted({str(x["topic_id"]) for x in val_examples})
    transcript_ids_in_bucket = sorted({str(x["transcript_id"]) for x in bucket_examples})

    return {
        "split_mode": "leave_one_topic_out",
        "base_persona_id": str(base_persona_id),
        "style_id": str(style_id),
        "held_out_topic_id": str(held_out_topic_id),
        "train_topic_ids": train_topic_ids,
        "val_topic_ids": val_topic_ids,
        "bucket_transcript_ids": transcript_ids_in_bucket,
        "train_examples": train_examples,
        "val_examples": val_examples,
    }


# === PROMPT / FORMATTING HELPERS ===

def format_message(role: str, content: str, default_chat_template: bool) -> str:
    role = role.lower().strip()

    if role == "system":
        prefix = "<|im_start|>system\n" if default_chat_template else "System"
    elif role == "user":
        prefix = "<|im_start|>user\n" if default_chat_template else "User"
    elif role == "assistant":
        prefix = "<|im_start|>assistant\n" if default_chat_template else "Assistant"
    else:
        prefix = role.capitalize()

    return f"{prefix}{content}<|im_end|>" if default_chat_template else f"{prefix}: {content.strip()}"


def render_context_messages(messages: List[Dict[str, str]], default_chat_template: bool) -> str:
    rendered = []

    for message in messages:
        role = message["role"]
        content = message["content"]
        rendered.append(format_message(role=role, content=content, default_chat_template=default_chat_template))

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
    default_chat_template: bool
) -> str:
    context_text = render_context_messages(example["context_messages"], default_chat_template=default_chat_template)
    special_text = "".join(special_tokens)

    if len(special_tokens) == 0:
        conditioned_context = f"{context_text}\n"
    else:
        if token_placement == "before_context":
            conditioned_context = f"{special_text}\n{context_text}\n".strip()
        elif token_placement == "after_context":
            conditioned_context = f"{context_text}\n{special_text}\n".strip()
        else:
            raise ValueError(f"Unsupported token_placement: {token_placement}")

    prompt_text = f"{conditioned_context}<|im_start|>user\n" if default_chat_template else f"{conditioned_context}User:"
    return prompt_text


def build_full_text(
    example: Dict[str, Any],
    special_tokens: List[str],
    token_placement: str,
    default_chat_template: bool
) -> str:
    prompt_text = build_prompt_text(
        example=example,
        special_tokens=special_tokens,
        token_placement=token_placement,
        default_chat_template=default_chat_template
    )

    target_text = example["target_message"].strip()
    # full_text = f"{prompt_text}{target_text}<|im_end|>\n" if default_chat_template else f"{prompt_text}{target_text}"
    # I am leaving <|im_end|>\n out, since we are only interested in scoring the actual reply.
    full_text = f"{prompt_text}{target_text}"
    return full_text


# === TENSORIZATION HELPERS ===

def build_training_example_tensors(
    tokenizer,
    example: Dict[str, Any],
    special_tokens: List[str],
    token_placement: str,
    max_length: int,
    default_chat_template: bool
) -> Optional[Dict[str, Any]]:
    """
    Build one supervised example while preserving the target under truncation.

    Strategy:
    - tokenize prompt and target separately, without automatic special tokens
    - if needed, truncate prompt from the LEFT
    - keep the target intact whenever possible
    - mask the prompt tokens in labels, supervise only the target tokens
    """
    prompt_text = build_prompt_text(
        example=example,
        special_tokens=special_tokens,
        token_placement=token_placement,
        default_chat_template=default_chat_template
    )

    target_text =  example["target_message"].strip()

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
        default_chat_template: bool,
        use_examples_percentage: float,
        
    ):
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.token_placement = token_placement
        self.max_length = max_length

        processed_items: List[Dict[str, Any]] = []
        dropped_examples = 0

        if use_examples_percentage < 1.0:
            examples = random.sample(examples, int(len(examples) * use_examples_percentage))

        for example in examples:
            item = build_training_example_tensors(
                tokenizer=tokenizer,
                example=example,
                special_tokens=special_tokens,
                token_placement=token_placement,
                max_length=max_length,
                default_chat_template=default_chat_template
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

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    metadata_list = []

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

def prepare_model_and_tokenizer(config: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.use_bf16:
        model_dtype = torch.bfloat16
    elif config.use_fp16:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=model_dtype,
    )

    special_tokens = make_special_tokens(
        base=config.special_token_base,
        n=config.num_special_tokens,
    )

    if len(special_tokens) > 0:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        model.resize_token_embeddings(len(tokenizer))
        special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    else:
        special_token_ids = []

    return model, tokenizer, special_tokens, special_token_ids


def freeze_all_parameters(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def enable_input_embeddings(model) -> None:
    model.get_input_embeddings().weight.requires_grad = True


def initialize_special_token_embeddings(
    model,
    special_token_ids: List[int],
) -> None:
    if len(special_token_ids) == 0:
        return

    embedding_weight = model.get_input_embeddings().weight

    with torch.no_grad():
        special_ids = torch.tensor(special_token_ids, device=embedding_weight.device)
        mask = torch.ones(embedding_weight.shape[0], dtype=torch.bool, device=embedding_weight.device)
        mask[special_ids] = False
        mean_embedding = embedding_weight[mask].mean(dim=0)

        for token_id in special_token_ids:
            embedding_weight[token_id].copy_(mean_embedding)


def register_embedding_gradient_mask(model, special_token_ids: List[int]):
    if len(special_token_ids) == 0:
        return None

    embedding_weight = model.get_input_embeddings().weight
    special_token_ids_tensor = torch.tensor(special_token_ids, dtype=torch.long)

    def grad_hook(grad: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(grad)
        mask[special_token_ids_tensor.to(grad.device)] = 1.0
        return grad * mask

    return embedding_weight.register_hook(grad_hook)


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


def get_special_token_embedding_rows(
    model,
    special_token_ids: List[int],
) -> Optional[torch.Tensor]:
    if len(special_token_ids) == 0:
        return None

    embedding_weight = model.get_input_embeddings().weight
    row_ids = torch.tensor(special_token_ids, device=embedding_weight.device)
    return embedding_weight[row_ids]


def save_special_token_artifacts(
    run_dir: Path,
    config: TrainConfig,
    special_tokens: List[str],
    special_token_ids: List[int],
    model,
    tokenizer,
    train_history: List[Dict[str, Any]],
    best_val_loss: float,
    final_val_loss: float,
    val_losses: List[float],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "config": asdict(config),
        "special_tokens": special_tokens,
        "special_token_ids": special_token_ids,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "train_history": train_history,
        "val_losses": val_losses,
    }

    if extra_metadata is not None:
        artifacts["metadata"] = extra_metadata

    save_json(artifacts, run_dir / "run_summary.json")

    if len(special_token_ids) > 0:
        special_rows = get_special_token_embedding_rows(model, special_token_ids)
        if special_rows is not None:
            torch.save(
                {
                    "special_tokens": special_tokens,
                    "special_token_ids": special_token_ids,
                    "embeddings": special_rows.detach().cpu(),
                },
                run_dir / "special_token_embeddings.pt",
            )

    tokenizer.save_pretrained(run_dir / "tokenizer")


# === TRAINING HELPERS ===

@torch.no_grad()
def run_validation_loss(
    model,
    val_loader: DataLoader,
    device: torch.device,
    position_mode: str,
    special_token_ids: List[int],
) -> float:
    model.eval()

    total_loss = 0.0
    total_examples = 0

    for batch in val_loader:
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
            raise ValueError("NaN/Inf validation loss encountered.")

        batch_size = input_ids.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def build_optimizer(
    model,
    config: TrainConfig,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [model.get_input_embeddings().weight],
        lr=config.learning_rate,
        weight_decay=0.0,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    num_train_batches: int,
):
    num_update_steps_per_epoch = max(
        1,
        math.ceil(num_train_batches / config.grad_accum_steps),
    )
    total_train_steps = max(1, config.num_epochs * num_update_steps_per_epoch)
    warmup_steps = int(config.warmup_ratio * total_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    return scheduler, total_train_steps


def maybe_add_special_token_l2_penalty(
    loss: torch.Tensor,
    model,
    special_token_ids: List[int],
    weight_decay: float,
) -> torch.Tensor:
    if weight_decay <= 0 or len(special_token_ids) == 0:
        return loss

    special_rows = get_special_token_embedding_rows(model, special_token_ids)
    if special_rows is None:
        return loss

    penalty = 0.5 * weight_decay * (special_rows ** 2).sum()
    return loss + penalty


def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config: TrainConfig,
    special_token_ids: List[int],
    val_loader: DataLoader,
    train_history: List[Dict[str, Any]],
    global_step: int,
    best_val_loss: float,
    best_embedding_state,
):
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        forward_kwargs = build_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_mode=config.position_mode,
            special_token_ids=special_token_ids,
        )

        outputs = model(**forward_kwargs)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"NaN/Inf training loss encountered at global_step={global_step}, batch_step={step}."
            )

        base_loss_value = float(loss.item())

        loss = maybe_add_special_token_l2_penalty(
            loss=loss,
            model=model,
            special_token_ids=special_token_ids,
            weight_decay=config.weight_decay,
        )

        full_loss_value = float(loss.item())
        loss = loss / config.grad_accum_steps

        loss.backward()

        if (step + 1) % config.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [model.get_input_embeddings().weight],
                config.max_grad_norm,
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % config.eval_every_steps == 0 or global_step == 1:
                val_loss = run_validation_loss(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    position_mode=config.position_mode,
                    special_token_ids=special_token_ids,
                )

                log_row = {
                    "global_step": global_step,
                    "base_train_loss": base_loss_value,
                    "full_train_loss": full_loss_value,
                    "val_loss": val_loss,
                }
                train_history.append(log_row)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_embedding_state = (
                        model.get_input_embeddings().weight.detach().cpu().clone()
                    )

    # flush any remaining accumulated gradients from a partial last batch
    remaining = (step + 1) % config.grad_accum_steps
    if remaining != 0:
        torch.nn.utils.clip_grad_norm_(
            [model.get_input_embeddings().weight],
            config.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1    

        if global_step % config.eval_every_steps == 0 or global_step == 1:
            val_loss = run_validation_loss(
                model=model,
                val_loader=val_loader,
                device=device,
                position_mode=config.position_mode,
                special_token_ids=special_token_ids,
            )

            log_row = {
                "global_step": global_step,
                "base_train_loss": base_loss_value,
                "full_train_loss": full_loss_value,
                "val_loss": val_loss,
            }
            train_history.append(log_row)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_embedding_state = (
                    model.get_input_embeddings().weight.detach().cpu().clone()
                )

    return global_step, best_val_loss, best_embedding_state, val_loss


def run_training(config: TrainConfig) -> Dict[str, Any]:
    set_seed(config.seed)

    if not config.run_name:
        config.run_name = build_run_name(config)

    examples_path = Path(config.repo_root) / config.examples_path
    examples = load_jsonl(examples_path)

    split_info = build_leave_one_topic_out_split(
        examples=examples,
        base_persona_id=config.base_persona_id,
        style_id=config.style_id,
        held_out_topic_id=config.held_out_topic_id,
    )

    train_examples = split_info["train_examples"]
    val_examples = split_info["val_examples"]

    if len(train_examples) == 0:
        raise ValueError("Training split is empty.")
    if len(val_examples) == 0:
        raise ValueError("Validation split is empty.")

    model, tokenizer, special_tokens, special_token_ids = prepare_model_and_tokenizer(config)

    freeze_all_parameters(model)
    enable_input_embeddings(model)
    initialize_special_token_embeddings(model, special_token_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    grad_hook_handle = register_embedding_gradient_mask(model, special_token_ids)

    train_dataset = NextUserTurnDataset(
        examples=train_examples,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        token_placement=config.token_placement,
        max_length=config.max_length,
        default_chat_template=config.default_chat_template,
        use_examples_percentage=config.use_examples_percentage,
    )
    val_dataset = NextUserTurnDataset(
        examples=val_examples,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        token_placement=config.token_placement,
        max_length=config.max_length,
        default_chat_template=config.default_chat_template,
        use_examples_percentage=1.0,
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after tensorization.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty after tensorization.")

    collate_fn = lambda batch: collate_batch(batch, tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    baseline_val_loss = run_validation_loss(
        model=model,
        val_loader=val_loader,
        device=device,
        position_mode=config.position_mode,
        special_token_ids=special_token_ids,
    )

    run_dir = build_run_dir(config)

    if config.num_special_tokens == 0:
        train_history: List[Dict[str, Any]] = []

        save_special_token_artifacts(
            run_dir=run_dir,
            config=config,
            special_tokens=special_tokens,
            special_token_ids=special_token_ids,
            model=model,
            tokenizer=tokenizer,
            train_history=train_history,
            best_val_loss=baseline_val_loss,
            final_val_loss=baseline_val_loss,
            val_losses=[],
            extra_metadata={
                "split_info": split_info,
                "n_train_examples": len(train_dataset),
                "n_val_examples": len(val_dataset),
                "dropped_train_examples": train_dataset.dropped_examples,
                "dropped_val_examples": val_dataset.dropped_examples,
                "is_baseline": True,
            },
        )

        if grad_hook_handle is not None:
            grad_hook_handle.remove()

        return {
            "config": asdict(config),
            "split_info": split_info,
            "n_train_examples": len(train_dataset),
            "n_val_examples": len(val_dataset),
            "dropped_train_examples": train_dataset.dropped_examples,
            "dropped_val_examples": val_dataset.dropped_examples,
            "baseline_val_loss": baseline_val_loss,
            "best_val_loss": baseline_val_loss,
            "final_val_loss": baseline_val_loss,
            "train_history": train_history,
            "special_tokens": special_tokens,
            "special_token_ids": special_token_ids,
            "is_baseline": True,
            "val_losses": [],
        }

    optimizer = build_optimizer(model=model, config=config)
    scheduler, total_train_steps = build_scheduler(
        optimizer=optimizer,
        config=config,
        num_train_batches=len(train_loader),
    )

    train_history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_embedding_state = None
    global_step = 0
    val_losses: List[float] = []

    for _epoch in range(config.num_epochs):
        global_step, best_val_loss, best_embedding_state, val_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            special_token_ids=special_token_ids,
            val_loader=val_loader,
            train_history=train_history,
            global_step=global_step,
            best_val_loss=best_val_loss,
            best_embedding_state=best_embedding_state,
        )
        val_losses.append(val_loss)

        if config.save_per_epoch:
            # TODO: Function for this...
            if len(special_token_ids) > 0:
                special_rows = get_special_token_embedding_rows(model, special_token_ids)
                if special_rows is not None:
                    torch.save(
                        {
                            "special_tokens": special_tokens,
                            "special_token_ids": special_token_ids,
                            "embeddings": special_rows.detach().cpu(),
                        },
                        run_dir / f"ep{_epoch+1}_special_token_embeddings.pt",
                    )


    if best_embedding_state is not None:
        model.get_input_embeddings().weight.data.copy_(best_embedding_state.to(device))

    final_val_loss = run_validation_loss(
        model=model,
        val_loader=val_loader,
        device=device,
        position_mode=config.position_mode,
        special_token_ids=special_token_ids,
    )

    save_special_token_artifacts(
        run_dir=run_dir,
        config=config,
        special_tokens=special_tokens,
        special_token_ids=special_token_ids,
        model=model,
        tokenizer=tokenizer,
        train_history=train_history,
        best_val_loss=best_val_loss,
        val_losses=val_losses,
        final_val_loss=final_val_loss,
        extra_metadata={
            "split_info": split_info,
            "n_train_examples": len(train_dataset),
            "n_val_examples": len(val_dataset),
            "dropped_train_examples": train_dataset.dropped_examples,
            "dropped_val_examples": val_dataset.dropped_examples,
            "baseline_val_loss": baseline_val_loss,
            "total_train_steps": total_train_steps,
            "is_baseline": False,
        },
    )

    if grad_hook_handle is not None:
        grad_hook_handle.remove()

    return {
        "config": asdict(config),
        "split_info": split_info,
        "n_train_examples": len(train_dataset),
        "n_val_examples": len(val_dataset),
        "dropped_train_examples": train_dataset.dropped_examples,
        "dropped_val_examples": val_dataset.dropped_examples,
        "baseline_val_loss": baseline_val_loss,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "train_history": train_history,
        "special_tokens": special_tokens,
        "special_token_ids": special_token_ids,
        "is_baseline": False,
        "val_losses": val_losses,
    }


# === CLI ===

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--examples_path", type=str, default="data/examples.jsonl")
    parser.add_argument("--runs_root", type=str, default="data/runs")

    parser.add_argument("--base_persona_id", type=str, required=True)
    parser.add_argument("--style_id", type=str, required=True)
    parser.add_argument("--held_out_topic_id", type=str, required=True)

    parser.add_argument("--num_special_tokens", type=int, default=1)
    parser.add_argument("--special_token_base", type=str, default="<st>")
    parser.add_argument(
        "--token_placement",
        type=str,
        choices=["before_context", "after_context"],
        default="after_context",
    )
    parser.add_argument(
        "--position_mode",
        type=str,
        choices=["default", "shared_position"],
        default="default",
    )

    parser.add_argument("--default_chat_template", action="store_true")
    parser.add_argument("--use_examples_percentage", type=float, default=1.0)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--eval_every_steps", type=int, default=20)
    parser.add_argument("--save_per_epoch", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.use_fp16 and args.use_bf16:
        raise ValueError("Use at most one of --use_fp16 and --use_bf16.")

    config = TrainConfig(
        repo_root=args.repo_root,
        examples_path=args.examples_path,
        runs_root=args.runs_root,
        base_persona_id=args.base_persona_id,
        style_id=args.style_id,
        held_out_topic_id=args.held_out_topic_id,
        num_special_tokens=args.num_special_tokens,
        special_token_base=args.special_token_base,
        token_placement=args.token_placement,
        position_mode=args.position_mode,
        default_chat_template=args.default_chat_template,
        use_examples_percentage=args.use_examples_percentage,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        eval_every_steps=args.eval_every_steps,
        save_per_epoch=args.save_per_epoch,
        seed=args.seed,
        use_fp16=args.use_fp16,
        use_bf16=args.use_bf16,
    )

    config.run_name = build_run_name(config)

    print("=" * 80)
    print("=== RUNNING SPECIAL TOKEN TRAINING ===")
    print(json.dumps(asdict(config), indent=2, ensure_ascii=False))

    results = run_training(config)

    run_dir = build_run_dir(config)
    save_json(results, run_dir / "result.json")

    summary_row = {
        "run_name": config.run_name,
        "base_persona_id": config.base_persona_id,
        "style_id": config.style_id,
        "held_out_topic_id": config.held_out_topic_id,
        "num_special_tokens": config.num_special_tokens,
        "token_placement": config.token_placement,
        "position_mode": config.position_mode,
        "default_chat_template": config.default_chat_template,
        "use_examples_percentage": config.use_examples_percentage,
        "model_name": config.model_name,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        "n_train_examples": results["n_train_examples"],
        "n_val_examples": results["n_val_examples"],
        "baseline_val_loss": results["baseline_val_loss"],
        "best_val_loss": results["best_val_loss"],
        "final_val_loss": results["final_val_loss"],
        "is_baseline": results["is_baseline"],
        "val_losses": results["val_losses"],
    }
    append_jsonl(summary_row, Path(config.repo_root) / config.runs_root / "runs_summary.jsonl")

    print("=== TRAINING COMPLETED ===")
    print(f"Run directory: {run_dir}")
    print(json.dumps(summary_row, indent=2, ensure_ascii=False))
    print("=" * 80)

if __name__ == "__main__":
    main()