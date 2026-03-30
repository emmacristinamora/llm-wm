# special-token/evaluate_special_token.py

# TO FIX - DO NOT RUN ON ENTIRE EVAL SET YET
## fix truncation for the conditioned cases (rn the truncationtakes out the system prompts)
## figure out whether we should compute P(gold|system prompt+context+special_tokens) or P(gold|system prompt+context) for the user/assistant (currently doing the latter but that means the special tokens not are part of the scoring prompt, which may not be ideal?)
## other metrics? should we still keep the cosine similarity? is P(generated|)

# === IMPORTS ===

import argparse
import json
import math
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# === CONFIG DATACLASS ===

@dataclass
class EvalConfig:
    # paths
    repo_root: str = "."
    examples_path: str = "data/examples.jsonl"
    transcripts_path: str = "data/transcripts.jsonl"
    runs_root: str = "data/runs"
    evals_root: str = "data/evals"
    run_name: str = ""

    # optional restriction of control universe
    allowed_personas: Optional[List[str]] = None
    allowed_styles: Optional[List[str]] = None

    # generation config
    generation_max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0

    # sentence similarity model
    sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # compute
    max_length: int = 1024
    max_examples_per_bucket: Optional[int] = None

    # reproducibility / precision
    seed: int = 42
    use_fp16: bool = False
    use_bf16: bool = False

    # saving
    save_per_example: bool = False


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


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(obj), f, indent=2, ensure_ascii=False)


def append_jsonl(row: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sanitize_for_json(row), ensure_ascii=False) + "\n")


def build_run_dir(config: EvalConfig) -> Path:
    if not config.run_name:
        raise ValueError("config.run_name is empty.")

    return Path(config.repo_root) / config.runs_root / config.run_name


def build_eval_dir(config: EvalConfig) -> Path:
    if not config.run_name:
        raise ValueError("config.run_name is empty.")

    return Path(config.repo_root) / config.evals_root / config.run_name


def load_training_artifacts(
    run_dir: Path,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    run_summary_path = run_dir / "run_summary.json"
    embedding_path = run_dir / "special_token_embeddings.pt"

    if not run_summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {run_summary_path}")

    run_summary = load_json(run_summary_path)
    train_config = run_summary["config"]
    num_special_tokens = int(train_config.get("num_special_tokens", 0))

    if num_special_tokens == 0:
        return run_summary, None

    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing learned embedding file: {embedding_path}")

    embedding_artifact = torch.load(embedding_path, map_location="cpu")
    return run_summary, embedding_artifact


# === EXAMPLE / TRANSCRIPT HELPERS ===

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


def build_transcript_lookup(transcripts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}

    for row in transcripts:
        transcript_id = row.get("transcript_id")
        conversation_id = row.get("conversation_id")

        if transcript_id is not None:
            lookup[str(transcript_id)] = row

        if conversation_id is not None:
            lookup[str(conversation_id)] = row

    return lookup

def get_system_prompts_for_example(
    example: Dict[str, Any],
    transcript_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    example_transcript_id = str(example["transcript_id"])

    if example_transcript_id not in transcript_lookup:
        raise KeyError(
            f"example transcript_id={example_transcript_id} not found in transcripts lookup"
        )

    row = transcript_lookup[example_transcript_id]

    system_llm1 = str(row.get("system_llm1", "")).strip()
    system_llm2 = str(row.get("system_llm2", "")).strip()

    if not system_llm1:
        raise ValueError(f"Missing system_llm1 for example transcript_id={example_transcript_id}")
    if not system_llm2:
        raise ValueError(f"Missing system_llm2 for example transcript_id={example_transcript_id}")

    return system_llm1, system_llm2


# === PROMPT / FORMATTING HELPERS ===

def format_message(role: str, content: str, default_chat_template: bool) -> str:
    role = role.lower().strip()
    content = content.strip()

    if default_chat_template:
        return f"<|im_start|>{role}\n{content}<|im_end|>"

    if role == "system":
        prefix = "System"
    elif role == "user":
        prefix = "User"
    elif role == "assistant":
        prefix = "Assistant"
    else:
        prefix = role.capitalize()

    return f"{prefix}: {content}"


def render_messages(messages: List[Dict[str, str]], default_chat_template: bool) -> str:
    rendered: List[str] = []

    for message in messages:
        role = str(message["role"]).strip()
        content = str(message["content"]).strip()
        rendered.append(
            format_message(
                role=role,
                content=content,
                default_chat_template=default_chat_template,
            )
        )

    return "\n".join(rendered).strip()


def make_special_tokens(base: str, n: int) -> List[str]:
    if n <= 0:
        return []

    if n == 1:
        return [base]

    return [f"{base}{i}" for i in range(n)]


def strip_system_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        m for m in messages
        if str(m.get("role", "")).strip().lower() != "system"
    ]


def build_prompt_text_train_aligned(
    example: Dict[str, Any],
    special_tokens: List[str],
    token_placement: str,
    default_chat_template: bool,
) -> str:
    """
    Mirrors training.

    Important:
    - uses example["context_messages"] exactly as stored
    - does not inject transcript system prompts
    - inserts learned special tokens before/after context
    - ends with a fresh user header because target_message is the next user turn
    """
    context_text = render_messages(
        example["context_messages"],
        default_chat_template=default_chat_template,
    )
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

    if default_chat_template:
        return f"{conditioned_context}<|im_start|>user\n"

    return f"{conditioned_context}User:"


def build_prompt_text_conditioned(
    example: Dict[str, Any],
    conditioning_system_prompt: str,
    default_chat_template: bool,
) -> str:
    """
    Prompt used for alternative conditioned scoring.

    Important:
    - injects explicit transcript-level system prompt
    - removes system messages from example context so we do not duplicate them
    - DOES NOT include learned special tokens

    This is intentional: these metrics are meant to answer
    "how plausible is this text under the user-conditioned or assistant-conditioned
    prompting regime?"
    rather than
    "how plausible is it under system prompt + learned tokens together?"
    """
    non_system_context = strip_system_messages(example["context_messages"])

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": conditioning_system_prompt},
        *non_system_context,
    ]

    context_text = render_messages(
        messages,
        default_chat_template=default_chat_template,
    )

    if default_chat_template:
        return f"{context_text}\n<|im_start|>user\n"

    return f"{context_text}\nUser:"


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
    allowed_personas: Optional[List[str]] = None,
    allowed_styles: Optional[List[str]] = None,
    max_examples_per_bucket: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    topic_examples = filter_examples(
        examples=examples,
        topic_id=held_out_topic_id,
    )

    if allowed_personas is not None:
        allowed_persona_set = {str(x) for x in allowed_personas}
        topic_examples = [
            ex for ex in topic_examples
            if str(ex["base_persona_id"]) in allowed_persona_set
        ]

    if allowed_styles is not None:
        allowed_style_set = {str(x) for x in allowed_styles}
        topic_examples = [
            ex for ex in topic_examples
            if str(ex["style_id"]) in allowed_style_set
        ]

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
    embedding_artifact: Optional[Dict[str, Any]],
) -> Optional[torch.Tensor]:
    if embedding_artifact is None:
        return None

    if "embeddings" not in embedding_artifact:
        raise ValueError("Embedding artifact does not contain 'embeddings'.")

    learned_rows = embedding_artifact["embeddings"]

    if not isinstance(learned_rows, torch.Tensor):
        raise ValueError("Loaded learned embeddings are not a torch.Tensor.")

    return learned_rows


def inject_special_token_embeddings(
    model,
    special_token_ids: List[int],
    learned_rows: Optional[torch.Tensor],
) -> None:
    if len(special_token_ids) == 0:
        return

    if learned_rows is None:
        raise ValueError("Expected learned_rows for non-baseline evaluation, but got None.")

    if learned_rows.shape[0] != len(special_token_ids):
        raise ValueError(
            f"Mismatch between learned rows ({learned_rows.shape[0]}) "
            f"and special token ids ({len(special_token_ids)})"
        )

    embedding_weight = model.get_input_embeddings().weight

    with torch.no_grad():
        for i, token_id in enumerate(special_token_ids):
            embedding_weight[token_id].copy_(
                learned_rows[i].to(
                    embedding_weight.device,
                    dtype=embedding_weight.dtype,
                )
            )


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


# === TOKENIZATION / SCORING HELPERS ===

def build_scoring_tensors(
    tokenizer,
    prompt_text: str,
    target_text: str,
    max_length: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Same masking logic as training:
    - prompt tokens are masked out in labels
    - only target tokens contribute to loss
    - prompt is left-truncated if needed so target is preserved
    """
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
        "input_ids": input_ids.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0),
        "labels": labels.unsqueeze(0),
    }


def build_generation_inputs(
    tokenizer,
    prompt_text: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Generation uses the same BOS + left-truncation convention as scoring,
    except there is no target segment.
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    bos_ids: List[int] = []
    if tokenizer.bos_token_id is not None:
        bos_ids = [tokenizer.bos_token_id]

    available_for_prompt = max_length - len(bos_ids)
    if available_for_prompt < 0:
        raise ValueError(f"max_length={max_length} is too small even for BOS.")

    if len(prompt_ids) > available_for_prompt:
        prompt_ids = prompt_ids[-available_for_prompt:] if available_for_prompt > 0 else []

    input_ids = torch.tensor([bos_ids + prompt_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


@torch.no_grad()
def compute_text_loss_under_prompt(
    model,
    tokenizer,
    prompt_text: str,
    target_text: str,
    device: torch.device,
    max_length: int,
    position_mode: str,
    special_token_ids: List[int],
) -> float:
    tensors = build_scoring_tensors(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        target_text=target_text,
        max_length=max_length,
    )

    if tensors is None:
        return float("nan")

    input_ids = tensors["input_ids"].to(device)
    attention_mask = tensors["attention_mask"].to(device)
    labels = tensors["labels"].to(device)

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


# === GENERATION HELPERS ===

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


def compute_repetition_score(text: str) -> float:
    """
    Simple repetition proxy:
    repeated bigram ratio = 1 - unique_bigrams / total_bigrams

    0.0 means no repeated bigrams.
    Higher means more repetition.
    """
    tokens = re.findall(r"\S+", text.lower())

    if len(tokens) < 2:
        return 0.0

    bigrams = list(zip(tokens[:-1], tokens[1:]))

    if len(bigrams) == 0:
        return 0.0

    unique_bigram_count = len(set(bigrams))
    total_bigram_count = len(bigrams)

    return 1.0 - (unique_bigram_count / total_bigram_count)


@torch.no_grad()
def generate_from_prompt(
    model,
    tokenizer,
    prompt_text: str,
    position_mode: str,
    special_token_ids: List[int],
    generation_max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_length: int,
    device: torch.device,
) -> str:
    tensors = build_generation_inputs(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_length=max_length,
    )

    input_ids = tensors["input_ids"].to(device)
    attention_mask = tensors["attention_mask"].to(device)

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


# === PER-EXAMPLE EVALUATION ===

def evaluate_one_example(
    example: Dict[str, Any],
    model,
    tokenizer,
    sentence_model,
    transcript_lookup: Dict[str, Dict[str, Any]],
    special_tokens: List[str],
    token_placement: str,
    position_mode: str,
    special_token_ids: List[int],
    default_chat_template: bool,
    eval_config: EvalConfig,
    device: torch.device,
) -> Dict[str, Any]:
    system_llm1, system_llm2 = get_system_prompts_for_example(
        example=example,
        transcript_lookup=transcript_lookup,
    )

    # Train-aligned prompt: this is the one that mirrors how the special-token run was trained.
    prompt_train_aligned = build_prompt_text_train_aligned(
        example=example,
        special_tokens=special_tokens,
        token_placement=token_placement,
        default_chat_template=default_chat_template,
    )

    # Conditioned prompts: these intentionally DO NOT include learned special tokens.
    prompt_user_conditioned = build_prompt_text_conditioned(
        example=example,
        conditioning_system_prompt=system_llm1,
        default_chat_template=default_chat_template,
    )

    prompt_assistant_conditioned = build_prompt_text_conditioned(
        example=example,
        conditioning_system_prompt=system_llm2,
        default_chat_template=default_chat_template,
    )

    gold_text = str(example["target_message"]).strip()

    # --- gold-target probabilistic metrics ---
    teacher_forced_loss_train_aligned = compute_text_loss_under_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_train_aligned,
        target_text=gold_text,
        device=device,
        max_length=eval_config.max_length,
        position_mode=position_mode,
        special_token_ids=special_token_ids,
    )

    # No special tokens are present in the conditioned prompts.
    teacher_forced_loss_user_conditioned = compute_text_loss_under_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_user_conditioned,
        target_text=gold_text,
        device=device,
        max_length=eval_config.max_length,
        position_mode=position_mode,
        special_token_ids=[],
    )

    teacher_forced_loss_assistant_conditioned = compute_text_loss_under_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_assistant_conditioned,
        target_text=gold_text,
        device=device,
        max_length=eval_config.max_length,
        position_mode=position_mode,
        special_token_ids=[],
    )

    # --- generation from train-aligned prompt only ---
    generated_text = generate_from_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_train_aligned,
        position_mode=position_mode,
        special_token_ids=special_token_ids,
        generation_max_new_tokens=eval_config.generation_max_new_tokens,
        do_sample=eval_config.do_sample,
        temperature=eval_config.temperature,
        top_p=eval_config.top_p,
        max_length=eval_config.max_length,
        device=device,
    )

    generation_cosine_similarity = compute_sentence_cosine_similarity(
        sentence_model=sentence_model,
        text_a=generated_text,
        text_b=gold_text,
    )

    exact_match = int(generated_text.strip() == gold_text.strip())
    repetition_score = compute_repetition_score(generated_text)

    # --- generated-text probabilistic metrics under alternative scoring regimes ---
    generated_text_loss_user_conditioned = compute_text_loss_under_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_user_conditioned,
        target_text=generated_text,
        device=device,
        max_length=eval_config.max_length,
        position_mode=position_mode,
        special_token_ids=[],
    )

    generated_text_loss_assistant_conditioned = compute_text_loss_under_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_assistant_conditioned,
        target_text=generated_text,
        device=device,
        max_length=eval_config.max_length,
        position_mode=position_mode,
        special_token_ids=[],
    )

    return {
        "example_id": example.get("example_id"),
        "transcript_id": example.get("transcript_id"),
        "topic_id": example.get("topic_id"),
        "base_persona_id": example.get("base_persona_id"),
        "style_id": example.get("style_id"),
        "user_turn_number": example.get("user_turn_number"),
        "target_message_index": example.get("target_message_index"),
        "gold_text": gold_text,
        "generated_text": generated_text,
        "teacher_forced_loss_train_aligned": teacher_forced_loss_train_aligned,
        "teacher_forced_loss_user_conditioned": teacher_forced_loss_user_conditioned,
        "teacher_forced_loss_assistant_conditioned": teacher_forced_loss_assistant_conditioned,
        "generation_cosine_similarity": generation_cosine_similarity,
        "exact_match": exact_match,
        "repetition_score": repetition_score,
        "generated_text_loss_user_conditioned": generated_text_loss_user_conditioned,
        "generated_text_loss_assistant_conditioned": generated_text_loss_assistant_conditioned,
    }


def mean_of_metric(rows: List[Dict[str, Any]], key: str) -> float:
    values: List[float] = []

    for row in rows:
        if key not in row:
            continue

        value = row[key]
        if value is None:
            continue

        value = float(value)
        if math.isnan(value) or math.isinf(value):
            continue

        values.append(value)

    if len(values) == 0:
        return float("nan")

    return sum(values) / len(values)


def evaluate_one_bucket(
    bucket_name: str,
    bucket_examples: List[Dict[str, Any]],
    model,
    tokenizer,
    sentence_model,
    transcript_lookup: Dict[str, Dict[str, Any]],
    special_tokens: List[str],
    token_placement: str,
    position_mode: str,
    special_token_ids: List[int],
    default_chat_template: bool,
    eval_config: EvalConfig,
    device: torch.device,
) -> Dict[str, Any]:
    per_example_rows: List[Dict[str, Any]] = []
    dropped_examples = 0

    for example in bucket_examples:
        try:
            row = evaluate_one_example(
                example=example,
                model=model,
                tokenizer=tokenizer,
                sentence_model=sentence_model,
                transcript_lookup=transcript_lookup,
                special_tokens=special_tokens,
                token_placement=token_placement,
                position_mode=position_mode,
                special_token_ids=special_token_ids,
                default_chat_template=default_chat_template,
                eval_config=eval_config,
                device=device,
            )
            per_example_rows.append(row)
        except Exception as exc:
            dropped_examples += 1
            example_id = example.get("example_id")
            print(
                f"[warn] bucket={bucket_name} example_id={example_id} dropped during evaluation: {exc}"
            )

    return {
        "bucket_name": bucket_name,
        "n_examples_raw": len(bucket_examples),
        "n_examples_used": len(per_example_rows),
        "n_examples_dropped": dropped_examples,
        "mean_teacher_forced_loss_train_aligned": mean_of_metric(per_example_rows, "teacher_forced_loss_train_aligned"),
        "mean_teacher_forced_loss_user_conditioned": mean_of_metric(per_example_rows, "teacher_forced_loss_user_conditioned"),
        "mean_teacher_forced_loss_assistant_conditioned": mean_of_metric(per_example_rows, "teacher_forced_loss_assistant_conditioned"),
        "mean_generation_cosine_similarity": mean_of_metric(per_example_rows, "generation_cosine_similarity"),
        "exact_match_rate": mean_of_metric(per_example_rows, "exact_match"),
        "mean_repetition_score": mean_of_metric(per_example_rows, "repetition_score"),
        "mean_generated_text_loss_user_conditioned": mean_of_metric(per_example_rows, "generated_text_loss_user_conditioned"),
        "mean_generated_text_loss_assistant_conditioned": mean_of_metric(per_example_rows, "generated_text_loss_assistant_conditioned"),
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

    metric_names = [
        "mean_teacher_forced_loss_train_aligned",
        "mean_teacher_forced_loss_user_conditioned",
        "mean_teacher_forced_loss_assistant_conditioned",
        "mean_generation_cosine_similarity",
        "exact_match_rate",
        "mean_repetition_score",
        "mean_generated_text_loss_user_conditioned",
        "mean_generated_text_loss_assistant_conditioned",
    ]

    for control_name in control_bucket_names:
        control = bucket_results[control_name]

        for metric_name in metric_names:
            deltas[f"{metric_name}__delta_matched_minus_{control_name}"] = (
                matched[metric_name] - control[metric_name]
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
    transcripts_path = Path(config.repo_root) / config.transcripts_path

    examples = load_jsonl(examples_path)
    transcripts = load_jsonl(transcripts_path)
    transcript_lookup = build_transcript_lookup(transcripts)

    target_base_persona_id = str(train_config["base_persona_id"])
    target_style_id = str(train_config["style_id"])
    held_out_topic_id = str(train_config["held_out_topic_id"])
    token_placement = str(train_config["token_placement"])
    position_mode = str(train_config["position_mode"])
    default_chat_template = bool(train_config.get("default_chat_template", False))

    model, tokenizer, special_tokens, special_token_ids = prepare_model_and_tokenizer_from_train_config(
        train_config=train_config,
        eval_config=config,
    )

    learned_rows = load_learned_special_token_embeddings(
        embedding_artifact=embedding_artifact
    )

    if int(train_config["num_special_tokens"]) > 0:
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
        allowed_personas=config.allowed_personas,
        allowed_styles=config.allowed_styles,
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
            sentence_model=sentence_model,
            transcript_lookup=transcript_lookup,
            special_tokens=special_tokens,
            token_placement=token_placement,
            position_mode=position_mode,
            special_token_ids=special_token_ids,
            default_chat_template=default_chat_template,
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
            enriched_row["default_chat_template"] = default_chat_template
            all_per_example_rows.append(enriched_row)

        print(
            f"[eval] bucket={bucket_name} "
            f"| n_used={bucket_result['n_examples_used']} "
            f"| train_aligned_loss={bucket_result['mean_teacher_forced_loss_train_aligned']:.6f} "
            f"| user_loss={bucket_result['mean_teacher_forced_loss_user_conditioned']:.6f} "
            f"| assistant_loss={bucket_result['mean_teacher_forced_loss_assistant_conditioned']:.6f} "
            f"| cosine={bucket_result['mean_generation_cosine_similarity']:.6f} "
            f"| exact_match={bucket_result['exact_match_rate']:.6f} "
            f"| repetition={bucket_result['mean_repetition_score']:.6f}"
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
        "is_baseline": int(train_config["num_special_tokens"]) == 0,
        "bucket_summaries": {
            bucket_name: {
                "bucket_name": bucket_result["bucket_name"],
                "n_examples_raw": bucket_result["n_examples_raw"],
                "n_examples_used": bucket_result["n_examples_used"],
                "n_examples_dropped": bucket_result["n_examples_dropped"],
                "mean_teacher_forced_loss_train_aligned": bucket_result["mean_teacher_forced_loss_train_aligned"],
                "mean_teacher_forced_loss_user_conditioned": bucket_result["mean_teacher_forced_loss_user_conditioned"],
                "mean_teacher_forced_loss_assistant_conditioned": bucket_result["mean_teacher_forced_loss_assistant_conditioned"],
                "mean_generation_cosine_similarity": bucket_result["mean_generation_cosine_similarity"],
                "exact_match_rate": bucket_result["exact_match_rate"],
                "mean_repetition_score": bucket_result["mean_repetition_score"],
                "mean_generated_text_loss_user_conditioned": bucket_result["mean_generated_text_loss_user_conditioned"],
                "mean_generated_text_loss_assistant_conditioned": bucket_result["mean_generated_text_loss_assistant_conditioned"],
            }
            for bucket_name, bucket_result in bucket_results.items()
        },
        "matched_vs_control_deltas": deltas,
    }

    save_json(summary, eval_dir / "eval_summary.json")

    if config.save_per_example:
        per_example_path = eval_dir / "per_example_results.jsonl"

        if per_example_path.exists():
            per_example_path.unlink()

        for row in all_per_example_rows:
            append_jsonl(row, per_example_path)

    return summary


# === CLI ===

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--examples_path", type=str, default="data/examples.jsonl")
    parser.add_argument("--transcripts_path", type=str, default="data/transcripts.jsonl")
    parser.add_argument("--runs_root", type=str, default="data/runs")
    parser.add_argument("--evals_root", type=str, default="data/evals")
    parser.add_argument("--run_name", type=str, required=True)

    parser.add_argument("--allowed_personas", type=str, default="")
    parser.add_argument("--allowed_styles", type=str, default="")

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
    parser.add_argument("--max_examples_per_bucket", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--save_per_example", action="store_true")

    return parser.parse_args()


# === CLI MAIN ===

def main() -> None:
    args = parse_args()

    if args.use_fp16 and args.use_bf16:
        raise ValueError("Use at most one of --use_fp16 and --use_bf16.")

    allowed_personas = (
        [x.strip() for x in args.allowed_personas.split(",") if x.strip()]
        if args.allowed_personas else None
    )
    allowed_styles = (
        [x.strip() for x in args.allowed_styles.split(",") if x.strip()]
        if args.allowed_styles else None
    )

    config = EvalConfig(
        repo_root=args.repo_root,
        examples_path=args.examples_path,
        transcripts_path=args.transcripts_path,
        runs_root=args.runs_root,
        evals_root=args.evals_root,
        run_name=args.run_name,
        allowed_personas=allowed_personas,
        allowed_styles=allowed_styles,
        generation_max_new_tokens=args.generation_max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        sentence_model_name=args.sentence_model_name,
        max_length=args.max_length,
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