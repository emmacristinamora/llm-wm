import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Data structures
# =========================

@dataclass
class Example:
    conversation_id: str
    persona_id: str
    synthetic_user_id: str
    base_persona_id: str
    style_id: str
    target_turn_index: int
    context_messages: List[Dict[str, str]]
    target_message: str


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    output_dir: str = "outputs/special_token"
    token_count: int = 1
    train_steps: int = 100
    lr: float = 5e-5
    weight_decay: float = 0.0
    train_mode: str = "single_conversation"   # single_conversation | multi_conversation
    num_train_conversations: int = 1          # used in multi_conversation mode
    special_position_mode: str = "default"    # default | shared_position
    max_eval_gen_tokens: int = 128
    seed: int = 0


# =========================
# Reproducibility
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# IO
# =========================

def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_examples_from_jsonl(
    path: Path,
    synthetic_user_id: Optional[str] = None,
) -> List[Example]:
    rows = load_jsonl(path)

    examples: List[Example] = []
    for row in rows:
        if synthetic_user_id is not None and row.get("synthetic_user_id") != synthetic_user_id:
            continue

        examples.append(
            Example(
                conversation_id=row["conversation_id"],
                persona_id=row["persona_id"],
                synthetic_user_id=row["synthetic_user_id"],
                base_persona_id=row["base_persona_id"],
                style_id=row["style_id"],
                target_turn_index=row["target_turn_index"],
                context_messages=row["context_messages"],
                target_message=row["target_message"],
            )
        )

    return examples


# =========================
# Grouping helpers
# =========================

def group_examples_by_conversation(examples: List[Example]) -> Dict[str, List[Example]]:
    grouped: Dict[str, List[Example]] = {}
    for ex in examples:
        grouped.setdefault(ex.conversation_id, []).append(ex)

    for conv_id in grouped:
        grouped[conv_id] = sorted(grouped[conv_id], key=lambda x: x.target_turn_index)

    return grouped


def select_train_examples(
    train_examples_all: List[Example],
    train_mode: str,
    num_train_conversations: int,
) -> List[Example]:
    grouped = group_examples_by_conversation(train_examples_all)
    conversation_ids = sorted(grouped.keys())

    if len(conversation_ids) == 0:
        raise ValueError("No training conversations found.")

    if train_mode == "single_conversation":
        selected_conv_ids = conversation_ids[:1]
    elif train_mode == "multi_conversation":
        k = min(num_train_conversations, len(conversation_ids))
        selected_conv_ids = conversation_ids[:k]
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    selected_examples: List[Example] = []
    for conv_id in selected_conv_ids:
        selected_examples.extend(grouped[conv_id])

    return selected_examples


# =========================
# Model / tokenizer setup
# =========================

def build_special_tokens(token_count: int) -> List[str]:
    return [f"<st{i}>" for i in range(token_count)]


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    return model, tokenizer


def resize_vocab_for_special_tokens(model, tokenizer, special_tokens: List[str]) -> List[int]:
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    return token_ids


def freeze_all_but_input_embeddings(model) -> None:
    for param in model.parameters():
        param.requires_grad = False

    input_embeddings = model.get_input_embeddings()
    input_embeddings.weight.requires_grad = True


def register_embedding_grad_mask(model, trainable_token_ids: List[int]):
    embedding_layer = model.get_input_embeddings()
    weight = embedding_layer.weight

    mask = torch.zeros_like(weight)
    for token_id in trainable_token_ids:
        mask[token_id] = 1.0

    hook = weight.register_hook(lambda grad: grad * mask)
    return hook


# =========================
# Input construction
# =========================

def build_training_texts(
    tokenizer,
    example: Example,
    special_tokens: List[str],
) -> Tuple[str, str, str]:
    """
    Returns:
      prompt_text
      target_text
      full_text

    prompt_text already includes the inserted special tokens.
    """
    prefix_text = tokenizer.apply_chat_template(
        example.context_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    special_text = "".join(special_tokens)

    target_text = f"<|im_start|>user\n{example.target_message}<|im_end|>\n"

    full_text = prefix_text + special_text + target_text
    prompt_text = prefix_text + "".join(special_tokens) + "<|im_start|>user\n"

    return prompt_text, target_text, full_text

# set max length for training to something reasonable (currently 2048 tokens) to avoid OOM issues
def build_inputs_for_example(
    model,
    tokenizer,
    example: Example,
    special_tokens: List[str],
    special_position_mode: str = "default",
    max_length: int = 2048,
) -> Dict[str, torch.Tensor]:
    prompt_text, _, full_text = build_training_texts(tokenizer, example, special_tokens)

    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
    full_prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
    prompt_len_full = full_prompt_ids.shape[0]

    if full_ids.shape[1] > max_length:
        # Truncate from the left, keeping the target (right side) intact
        input_ids = full_ids[:, -max_length:].to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)
        # Adjust prompt_len: how much of the prompt survived truncation
        tokens_cut = full_ids.shape[1] - max_length
        prompt_len = max(prompt_len_full - tokens_cut, 0)
    else:
        input_ids = full_ids.to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)
        prompt_len = prompt_len_full

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    if special_position_mode == "shared_position":
        batch["position_ids"] = build_shared_position_ids_for_special_tokens(
            tokenizer=tokenizer,
            input_ids=input_ids,
            special_tokens=special_tokens,
        ).to(model.device)

    return batch


def build_shared_position_ids_for_special_tokens(
    tokenizer,
    input_ids: torch.Tensor,
    special_tokens: List[str],
) -> torch.Tensor:
    """
    Proxy setting for special-token position handling.

    This does not remove positional information from the model.
    It forces all inserted special tokens to share the same position id.
    """
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    special_token_ids = set(tokenizer.convert_tokens_to_ids(special_tokens))

    shared_pos = None
    for j in range(input_ids.shape[1]):
        tok_id = int(input_ids[0, j].item())
        if tok_id in special_token_ids:
            if shared_pos is None:
                shared_pos = int(position_ids[0, j].item())
            position_ids[0, j] = shared_pos

    return position_ids


# =========================
# Training / evaluation
# =========================

def train_step(model, optimizer, batch: Dict[str, torch.Tensor]) -> float:
    model.train()
    outputs = model(**batch)
    loss = outputs.loss

    # debug
    #print("loss:", loss.item())
    #print("logits has NaN:", outputs.logits.isnan().any().item())
    #print("logits has Inf:", outputs.logits.isinf().any().item())
    #print("input_ids shape:", batch["input_ids"].shape)
    #print("labels min:", batch["labels"].min().item(), "max:", batch["labels"].max().item())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([model.get_input_embeddings().weight], max_norm=1.0)
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def compute_loss(model, batch: Dict[str, torch.Tensor]) -> float:
    model.eval()
    outputs = model(**batch)
    return float(outputs.loss.item())


@torch.no_grad()
def generate_user_turn(
    model,
    tokenizer,
    example: Example,
    special_tokens: List[str],
    special_position_mode: str,
    max_new_tokens: int,
) -> str:
    prefix_text = tokenizer.apply_chat_template(
        example.context_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = prefix_text + "".join(special_tokens)

    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if special_position_mode == "shared_position":
        gen_kwargs["position_ids"] = build_shared_position_ids_for_special_tokens(
            tokenizer=tokenizer,
            input_ids=input_ids,
            special_tokens=special_tokens,
        ).to(model.device)

    out = model.generate(**gen_kwargs)
    gen_ids = out[0][input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    return text.strip()


def exact_match(a: str, b: str) -> int:
    return int(a.strip() == b.strip())


# =========================
# Main experiment logic
# =========================

def run_training(
    train_examples_all: List[Example],
    eval_examples: List[Example],
    config: TrainConfig,
) -> Dict:
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = select_train_examples(
        train_examples_all=train_examples_all,
        train_mode=config.train_mode,
        num_train_conversations=config.num_train_conversations,
    )

    if len(train_examples) == 0:
        raise ValueError("No training examples selected.")
    if len(eval_examples) == 0:
        raise ValueError("No evaluation examples found.")

    train_conv_ids = sorted({ex.conversation_id for ex in train_examples})
    eval_conv_ids = sorted({ex.conversation_id for ex in eval_examples})

    model, tokenizer = load_model_and_tokenizer(config.model_name)
    special_tokens = build_special_tokens(config.token_count)
    trainable_token_ids = resize_vocab_for_special_tokens(model, tokenizer, special_tokens)

    # debug: Initialize the special token embeddings to something sensible, we are getting NaNs with random init
    with torch.no_grad():
        embeddings = model.get_input_embeddings()
        mean_embedding = embeddings.weight[:-len(special_tokens)].float().mean(dim=0, keepdim=True).to(embeddings.weight.dtype)
        # debug
        #print("mean_embedding has NaN:", mean_embedding.isnan().any().item())
        #print("mean_embedding has Inf:", mean_embedding.isinf().any().item())
        for token_id in trainable_token_ids:
            embeddings.weight[token_id].copy_(mean_embedding.squeeze())
    

    freeze_all_but_input_embeddings(model)
    hook = register_embedding_grad_mask(model, trainable_token_ids)

    train_batches = [
        build_inputs_for_example(
            model=model,
            tokenizer=tokenizer,
            example=ex,
            special_tokens=special_tokens,
            special_position_mode=config.special_position_mode,
        )
        for ex in train_examples
    ]

    eval_batches = [
        build_inputs_for_example(
            model=model,
            tokenizer=tokenizer,
            example=ex,
            special_tokens=special_tokens,
            special_position_mode=config.special_position_mode,
        )
        for ex in eval_examples
    ]

    optimizer = torch.optim.AdamW(
        [model.get_input_embeddings().weight],
        lr=config.lr,
        weight_decay=config.weight_decay, eps=1e-8
    )

    history = {
        "train_loss": [],
        "eval_loss": [],
    }

    eval_every = 10

    for step in range(config.train_steps):
        batch = train_batches[step % len(train_batches)]
        train_loss = train_step(model, optimizer, batch)
        history["train_loss"].append(train_loss)

        if step % eval_every == 0 or step == config.train_steps - 1:
            torch.cuda.empty_cache()
            eval_loss = sum(compute_loss(model, b) for b in eval_batches) / len(eval_batches)
            history["eval_loss"].append({"step": step, "loss": eval_loss})

    torch.cuda.empty_cache()

    reproduce_records = []
    for ex in eval_examples:
        predicted = generate_user_turn(
            model=model,
            tokenizer=tokenizer,
            example=ex,
            special_tokens=special_tokens,
            special_position_mode=config.special_position_mode,
            max_new_tokens=config.max_eval_gen_tokens,
        )
        gold = ex.target_message
        reproduce_records.append(
            {
                "conversation_id": ex.conversation_id,
                "target_turn_index": ex.target_turn_index,
                "gold_user_turn": gold,
                "generated_text": predicted,
                "exact_match": exact_match(predicted, gold),
            }
        )
        del predicted
        torch.cuda.empty_cache()

    hook.remove()

    result = {
        "config": asdict(config),
        "special_tokens": special_tokens,
        "num_train_examples": len(train_examples),
        "num_eval_examples": len(eval_examples),
        "num_train_conversations_used": len(train_conv_ids),
        "num_eval_conversations": len(eval_conv_ids),
        "train_conversation_ids": train_conv_ids,
        "eval_conversation_ids": eval_conv_ids,
        "final_train_loss": history["train_loss"][-1],
        "final_eval_loss": history["eval_loss"][-1]["loss"],
        "history": history,
        "reproduce_records": reproduce_records,
    }

    return result


# =========================
# Sweeps
# =========================

def plot_curve(
    xs: List[float],
    ys: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_token_count_sweep(
    train_examples_all: List[Example],
    eval_examples: List[Example],
    base_config: TrainConfig,
    token_counts: List[int],
) -> List[Dict]:
    results = []
    for token_count in token_counts:
        cfg = TrainConfig(**asdict(base_config))
        cfg.token_count = token_count
        cfg.output_dir = str(Path(base_config.output_dir) / f"token_count_{token_count}")
        result = run_training(train_examples_all, eval_examples, cfg)
        results.append(result)
    return results


def run_weight_decay_sweep(
    train_examples_all: List[Example],
    eval_examples: List[Example],
    base_config: TrainConfig,
    weight_decays: List[float],
) -> List[Dict]:
    results = []
    for wd in weight_decays:
        cfg = TrainConfig(**asdict(base_config))
        cfg.weight_decay = wd
        cfg.output_dir = str(Path(base_config.output_dir) / f"wd_{wd}")
        result = run_training(train_examples_all, eval_examples, cfg)
        results.append(result)
    return results


def run_num_conversations_sweep(
    train_examples_all: List[Example],
    eval_examples: List[Example],
    base_config: TrainConfig,
    conversation_counts: List[int],
) -> List[Dict]:
    results = []
    for k in conversation_counts:
        cfg = TrainConfig(**asdict(base_config))
        cfg.train_mode = "multi_conversation"
        cfg.num_train_conversations = k
        cfg.output_dir = str(Path(base_config.output_dir) / f"num_train_conversations_{k}")
        result = run_training(train_examples_all, eval_examples, cfg)
        results.append(result)
    return results


# =========================
# CLI
# =========================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_examples_path", type=str, required=True)
    p.add_argument("--test_examples_path", type=str, required=True)
    p.add_argument("--synthetic_user_id", type=str, required=True)

    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--output_dir", type=str, default="outputs/special_token")

    p.add_argument("--token_count", type=int, default=1)
    p.add_argument("--train_steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument(
        "--train_mode",
        type=str,
        default="single_conversation",
        choices=["single_conversation", "multi_conversation"],
    )
    p.add_argument("--num_train_conversations", type=int, default=1)

    p.add_argument(
        "--special_position_mode",
        type=str,
        default="default",
        choices=["default", "shared_position"],
    )

    p.add_argument("--max_eval_gen_tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--run_token_count_sweep", action="store_true")
    p.add_argument("--token_count_values", type=int, nargs="*", default=[1, 3, 5])

    p.add_argument("--run_weight_decay_sweep", action="store_true")
    p.add_argument("--weight_decay_values", type=float, nargs="*", default=[0.0, 1e-4, 1e-3, 1e-2])

    p.add_argument("--run_num_conversations_sweep", action="store_true")
    p.add_argument("--num_conversation_values", type=int, nargs="*", default=[1, 2, 4, 6, 8])

    return p.parse_args()


def main():
    args = parse_args()

    train_examples_all = load_examples_from_jsonl(
        path=Path(args.train_examples_path),
        synthetic_user_id=args.synthetic_user_id,
    )
    eval_examples = load_examples_from_jsonl(
        path=Path(args.test_examples_path),
        synthetic_user_id=args.synthetic_user_id,
    )

    if len(train_examples_all) == 0:
        raise ValueError("No training examples found for the requested synthetic_user_id.")
    if len(eval_examples) == 0:
        raise ValueError("No test examples found for the requested synthetic_user_id.")

    base_config = TrainConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        token_count=args.token_count,
        train_steps=args.train_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_mode=args.train_mode,
        num_train_conversations=args.num_train_conversations,
        special_position_mode=args.special_position_mode,
        max_eval_gen_tokens=args.max_eval_gen_tokens,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_token_count_sweep:
        sweep_results = run_token_count_sweep(
            train_examples_all=train_examples_all,
            eval_examples=eval_examples,
            base_config=base_config,
            token_counts=args.token_count_values,
        )

        xs = [r["config"]["token_count"] for r in sweep_results]
        ys = [r["final_eval_loss"] for r in sweep_results]

        with (output_dir / "token_count_sweep.json").open("w", encoding="utf-8") as f:
            json.dump(sweep_results, f, indent=2)

        plot_curve(
            xs=xs,
            ys=ys,
            xlabel="Number of special tokens",
            ylabel="Final eval loss",
            title="Special token count sweep",
            out_path=output_dir / "token_count_sweep.png",
        )

    elif args.run_weight_decay_sweep:
        sweep_results = run_weight_decay_sweep(
            train_examples_all=train_examples_all,
            eval_examples=eval_examples,
            base_config=base_config,
            weight_decays=args.weight_decay_values,
        )

        xs = [r["config"]["weight_decay"] for r in sweep_results]
        ys = [r["final_eval_loss"] for r in sweep_results]

        with (output_dir / "weight_decay_sweep.json").open("w", encoding="utf-8") as f:
            json.dump(sweep_results, f, indent=2)

        plot_curve(
            xs=xs,
            ys=ys,
            xlabel="Weight decay",
            ylabel="Final eval loss",
            title="Weight decay sweep",
            out_path=output_dir / "weight_decay_sweep.png",
        )

    elif args.run_num_conversations_sweep:
        sweep_results = run_num_conversations_sweep(
            train_examples_all=train_examples_all,
            eval_examples=eval_examples,
            base_config=base_config,
            conversation_counts=args.num_conversation_values,
        )

        xs = [r["config"]["num_train_conversations"] for r in sweep_results]
        ys = [r["final_eval_loss"] for r in sweep_results]

        with (output_dir / "num_conversations_sweep.json").open("w", encoding="utf-8") as f:
            json.dump(sweep_results, f, indent=2)

        plot_curve(
            xs=xs,
            ys=ys,
            xlabel="Number of training conversations",
            ylabel="Final eval loss",
            title="Training conversation count sweep",
            out_path=output_dir / "num_conversations_sweep.png",
        )

    else:
        result = run_training(
            train_examples_all=train_examples_all,
            eval_examples=eval_examples,
            config=base_config,
        )

        with (output_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        with (output_dir / "train_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(base_config), f, indent=2)

        plot_curve(
            xs=list(range(len(result["history"]["train_loss"]))),
            ys=result["history"]["train_loss"],
            xlabel="Step",
            ylabel="Train loss",
            title="Training loss",
            out_path=output_dir / "train_loss.png",
        )

        eval_steps = [pt["step"] for pt in result["history"]["eval_loss"]]
        eval_losses = [pt["loss"] for pt in result["history"]["eval_loss"]]

        plot_curve(
            xs=eval_steps,
            ys=eval_losses,
            xlabel="Step",
            ylabel="Eval loss",
            title="Evaluation loss",
            out_path=output_dir / "eval_loss.png",
        )


if __name__ == "__main__":
    main()