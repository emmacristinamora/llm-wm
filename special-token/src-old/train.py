# === IMPORTS ===

import math
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from src.evaluate import evaluate_run


# === CONFIG ===

@dataclass
class TrainConfig:
    model_name: str

    num_special_tokens: int = 1
    special_token_base: str = "<st>"
    token_placement: str = "after_context"      # before_context | after_context
    position_mode: str = "default"              # default | shared_position

    max_length: int = 1024
    batch_size: int = 1
    num_epochs: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    grad_accum_steps: int = 1
    max_grad_norm: float = 0.5

    eval_every_steps: int = 50

    generation_max_new_tokens: int = 80
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    max_generation_examples: int = 100
    cosine_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    seed: int = 42
    use_fp16: bool = False
    use_bf16: bool = False


# === REPRODUCIBILITY ===

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# === PROMPT HELPERS ===

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
    return "\n".join(format_message(m["role"], m["content"]) for m in messages).strip()


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
    full_text = f"{prompt_text} {example['target_message'].strip()}"
    return full_text


# === SAFE TOKENIZATION ===

def build_training_example_tensors(
    tokenizer,
    example: Dict[str, Any],
    special_tokens: List[str],
    token_placement: str,
    max_length: int,
) -> Optional[Dict[str, Any]]:
    """
    Build one supervised example while preserving the target under truncation.

    Strategy:
    - tokenize prompt and target separately (without special tokens)
    - if too long, truncate prompt from the LEFT
    - then concatenate:
        [BOS?] + prompt_ids + target_ids
    - labels mask prompt tokens and supervise target tokens only
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

    bos_ids = []
    if tokenizer.bos_token_id is not None:
        bos_ids = [tokenizer.bos_token_id]

    available_for_prompt = max_length - len(bos_ids) - len(target_ids)

    # If target alone does not fit, skip this example.
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
            "user_turn_number": example.get("user_turn_number"),
            "target_message": example.get("target_message"),
            "prompt_text": prompt_text,
        },
    }


# === DATASET ===

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

        processed_examples = []
        dropped_no_target = 0

        for example in examples:
            item = build_training_example_tensors(
                tokenizer=tokenizer,
                example=example,
                special_tokens=special_tokens,
                token_placement=token_placement,
                max_length=max_length,
            )
            if item is None:
                dropped_no_target += 1
            else:
                processed_examples.append(item)

        self.items = processed_examples
        self.dropped_no_target = dropped_no_target

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
        torch_dtype = torch.bfloat16
    elif config.use_fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
    )

    special_tokens = make_special_tokens(config.special_token_base, config.num_special_tokens)

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


def register_embedding_gradient_mask(model, special_token_ids: List[int]) -> None:
    if len(special_token_ids) == 0:
        return

    embedding_weight = model.get_input_embeddings().weight
    special_token_ids_tensor = torch.tensor(special_token_ids, dtype=torch.long)

    def grad_hook(grad: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(grad)
        mask[special_token_ids_tensor.to(grad.device)] = 1.0
        return grad * mask

    embedding_weight.register_hook(grad_hook)


def build_position_ids_with_shared_special_tokens(
    input_ids: torch.Tensor,
    special_token_ids: List[int],
) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1).clone()
    special_token_id_set = set(special_token_ids)

    for b in range(batch_size):
        for t in range(seq_len):
            if input_ids[b, t].item() in special_token_id_set:
                position_ids[b, t] = 0

    return position_ids


def build_prompt_fn(
    special_tokens: List[str],
    token_placement: str,
):
    def _build_prompt(example: Dict[str, Any]) -> str:
        return build_prompt_text(
            example=example,
            special_tokens=special_tokens,
            token_placement=token_placement,
        )
    return _build_prompt


# === TRAINING ===

def run_training(
    config: TrainConfig,
    train_examples: List[Dict[str, Any]],
    val_examples: List[Dict[str, Any]],
    test_examples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    set_seed(config.seed)

    if test_examples is None:
        test_examples = []

    model, tokenizer, special_tokens, special_token_ids = prepare_model_and_tokenizer(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = NextUserTurnDataset(
        examples=train_examples,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        token_placement=config.token_placement,
        max_length=config.max_length,
    )
    val_dataset = NextUserTurnDataset(
        examples=val_examples,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        token_placement=config.token_placement,
        max_length=config.max_length,
    )
    test_dataset = NextUserTurnDataset(
        examples=test_examples,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        token_placement=config.token_placement,
        max_length=config.max_length,
    ) if len(test_examples) > 0 else None

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after filtering invalid/truncated examples.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty after filtering invalid/truncated examples.")
    if test_dataset is not None and len(test_dataset) == 0:
        test_dataset = None

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    ) if test_dataset is not None else None

    prompt_fn = build_prompt_fn(
        special_tokens=special_tokens,
        token_placement=config.token_placement,
    )

    # === BASELINE CASE ===

    if config.num_special_tokens == 0:
        baseline_val = evaluate_run(
            model=model,
            tokenizer=tokenizer,
            eval_examples=val_examples,
            eval_dataloader=val_loader,
            build_prompt_fn=prompt_fn,
            device=device,
            position_mode="default",
            special_token_ids=[],
            build_position_ids_fn=None,
            generation_max_new_tokens=config.generation_max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            max_generation_examples=config.max_generation_examples,
            cosine_model_name=config.cosine_model_name,
            max_input_length=config.max_length,
        )

        results = {
            "config": asdict(config),
            "special_tokens": [],
            "special_token_ids": [],
            "n_train_examples": len(train_dataset),
            "n_val_examples": len(val_dataset),
            "n_test_examples": len(test_dataset) if test_dataset is not None else 0,
            "dropped_train_examples": train_dataset.dropped_no_target,
            "dropped_val_examples": val_dataset.dropped_no_target,
            "dropped_test_examples": test_dataset.dropped_no_target if test_dataset is not None else 0,
            "train_history": [],
            "best_val_loss": baseline_val["eval_loss"],
            "final_val_loss": baseline_val["eval_loss"],
            "val_generations": baseline_val["generations"],
            "val_cosine_metrics": baseline_val["cosine_metrics"],
            "is_baseline": True,
        }

        if test_loader is not None:
            baseline_test = evaluate_run(
                model=model,
                tokenizer=tokenizer,
                eval_examples=test_examples,
                eval_dataloader=test_loader,
                build_prompt_fn=prompt_fn,
                device=device,
                position_mode="default",
                special_token_ids=[],
                build_position_ids_fn=None,
                generation_max_new_tokens=config.generation_max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
                max_generation_examples=config.max_generation_examples,
                cosine_model_name=config.cosine_model_name,
                max_input_length=config.max_length,
            )
            results["final_test_loss"] = baseline_test["eval_loss"]
            results["test_generations"] = baseline_test["generations"]
            results["test_cosine_metrics"] = baseline_test["cosine_metrics"]

        return results

    # === SPECIAL TOKEN TRAINING CASE ===

    freeze_all_parameters(model)
    enable_input_embeddings(model)
    register_embedding_gradient_mask(model, special_token_ids)

    optimizer = torch.optim.AdamW(
        [model.get_input_embeddings().weight],
        lr=config.learning_rate,
        weight_decay=0.0,
    )

    num_update_steps_per_epoch = max(1, math.ceil(len(train_loader) / config.grad_accum_steps))
    total_train_steps = max(1, config.num_epochs * num_update_steps_per_epoch)
    warmup_steps = int(config.warmup_ratio * total_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    train_history = []
    best_val_loss = float("inf")
    best_embedding_state = None
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            if config.position_mode == "shared_position":
                forward_kwargs["position_ids"] = build_position_ids_with_shared_special_tokens(
                    input_ids=input_ids,
                    special_token_ids=special_token_ids,
                )

            outputs = model(**forward_kwargs)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(
                    f"NaN/Inf training loss encountered at epoch={epoch}, step={step}, global_step={global_step}."
                )

            loss = loss / config.grad_accum_steps

            if config.weight_decay > 0 and len(special_token_ids) > 0:
                st_ids = torch.tensor(special_token_ids, device=device)
                st_emb = model.get_input_embeddings().weight[st_ids]
                loss = loss + 0.5 * config.weight_decay * (st_emb ** 2).sum() / config.grad_accum_steps

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
                    val_metrics = evaluate_run(
                        model=model,
                        tokenizer=tokenizer,
                        eval_examples=val_examples,
                        eval_dataloader=val_loader,
                        build_prompt_fn=prompt_fn,
                        device=device,
                        position_mode=config.position_mode,
                        special_token_ids=special_token_ids,
                        build_position_ids_fn=build_position_ids_with_shared_special_tokens,
                        generation_max_new_tokens=config.generation_max_new_tokens,
                        do_sample=config.do_sample,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        max_generation_examples=min(20, config.max_generation_examples),
                        cosine_model_name=config.cosine_model_name,
                        max_input_length=config.max_length,
                    )

                    log_row = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": loss.item() * config.grad_accum_steps,
                        "val_loss": val_metrics["eval_loss"],
                    }
                    train_history.append(log_row)

                    if val_metrics["eval_loss"] < best_val_loss:
                        best_val_loss = val_metrics["eval_loss"]
                        best_embedding_state = model.get_input_embeddings().weight.detach().cpu().clone()

    if best_embedding_state is not None:
        model.get_input_embeddings().weight.data.copy_(best_embedding_state.to(device))

    final_val = evaluate_run(
        model=model,
        tokenizer=tokenizer,
        eval_examples=val_examples,
        eval_dataloader=val_loader,
        build_prompt_fn=prompt_fn,
        device=device,
        position_mode=config.position_mode,
        special_token_ids=special_token_ids,
        build_position_ids_fn=build_position_ids_with_shared_special_tokens,
        generation_max_new_tokens=config.generation_max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
        max_generation_examples=config.max_generation_examples,
        cosine_model_name=config.cosine_model_name,
        max_input_length=config.max_length,
    )

    results = {
        "config": asdict(config),
        "special_tokens": special_tokens,
        "special_token_ids": special_token_ids,
        "n_train_examples": len(train_dataset),
        "n_val_examples": len(val_dataset),
        "n_test_examples": len(test_dataset) if test_dataset is not None else 0,
        "dropped_train_examples": train_dataset.dropped_no_target,
        "dropped_val_examples": val_dataset.dropped_no_target,
        "dropped_test_examples": test_dataset.dropped_no_target if test_dataset is not None else 0,
        "train_history": train_history,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val["eval_loss"],
        "val_generations": final_val["generations"],
        "val_cosine_metrics": final_val["cosine_metrics"],
        "is_baseline": False,
    }

    if test_loader is not None:
        final_test = evaluate_run(
            model=model,
            tokenizer=tokenizer,
            eval_examples=test_examples,
            eval_dataloader=test_loader,
            build_prompt_fn=prompt_fn,
            device=device,
            position_mode=config.position_mode,
            special_token_ids=special_token_ids,
            build_position_ids_fn=build_position_ids_with_shared_special_tokens,
            generation_max_new_tokens=config.generation_max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            max_generation_examples=config.max_generation_examples,
            cosine_model_name=config.cosine_model_name,
            max_input_length=config.max_length,
        )
        results["final_test_loss"] = final_test["eval_loss"]
        results["test_generations"] = final_test["generations"]
        results["test_cosine_metrics"] = final_test["cosine_metrics"]

    return results