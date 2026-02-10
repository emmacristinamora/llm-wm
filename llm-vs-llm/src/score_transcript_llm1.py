# src/score_transcript_llm1.py
#
# Minimal scorer for transcripts where investigator_mode == "none".
# Produces per-USER-turn (LLM1) NLL scores and saves to data/scores/*.parquet.
#
# Assumptions (robust to minor variations):
# - Transcript is a JSONL where each row has:
#   - persona_id
#   - profile: { base_persona_id, style_id, investigator_mode }
#   - system_llm2 (assistant system prompt)  <-- used as the scorer system prompt for consistency
#   - messages: list[{role, content}]
#
# Output (parquet):
# - conversation_idx (row index)
# - persona_id, base_persona_id, style_id
# - user_turn_idx
# - avg_nll, num_tokens, ppl

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import math

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# === CONFIG ===

BASE_DIR = Path(__file__).parent.parent
TRANSCRIPT_PATH = BASE_DIR / "src" / "data" / "conversations" / "transcripts.jsonl"
OUT_DIR = BASE_DIR / "src" / "data" / "scores"


SCORER_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # keep consistent with your setup
DEVICE_MAP = "auto"
MAX_CONTEXT_TOKENS = 8192


# === IO ===

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# === MODEL ===

def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model(model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=pick_dtype(),
        device_map=DEVICE_MAP,
        trust_remote_code=True,
    )
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


# === CHAT TEMPLATE HELPERS ===

def apply_chat_template(tokenizer, messages, add_generation_prompt: bool):
    """
    Robust wrapper across tokenizers that implement apply_chat_template with either:
    - apply_chat_template(conversation=..., ...)
    - apply_chat_template(messages=..., ...)
    """
    if hasattr(tokenizer, "apply_chat_template"):
        for kwargs in (
            {"messages": messages, "add_generation_prompt": add_generation_prompt, "return_tensors": "pt"},
            {"conversation": messages, "add_generation_prompt": add_generation_prompt, "return_tensors": "pt"},
        ):
            try:
                return tokenizer.apply_chat_template(**kwargs, enable_thinking=False)
            except TypeError:
                try:
                    return tokenizer.apply_chat_template(**kwargs)
                except TypeError:
                    pass

    # fallback: plain text format
    text = ""
    for m in messages:
        text += f"{m['role'].upper()}: {m['content']}\n"
    if add_generation_prompt:
        text += "ASSISTANT: "
    return tokenizer(text, return_tensors="pt").input_ids


def build_inputs_for_user_turn(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    history: List[Dict[str, str]],
    user_text: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create (input_ids, labels) such that loss is computed ONLY on user_text tokens.

    Note: We include the user message as a normal chat message with role='user'
    and score only the tokens that belong to that user message.
    """
    prompt_msgs: List[Dict[str, str]] = []
    if system_prompt:
        prompt_msgs.append({"role": "system", "content": system_prompt})
    prompt_msgs.extend(history)

    # build ids up to "user role marker" but with empty content, so that labels are correctly aligned to only score user_text tokens 
    # we want to score only the actual user content tokens, not the role marker or any system/assistant tokens before it (USER:/)
    empty_user_msgs = list(prompt_msgs) + [{"role": "user", "content": ""}]
    empty_ids = apply_chat_template(tokenizer, empty_user_msgs, add_generation_prompt=False)

    # now build full ids with actual user content
    full_msgs = list(prompt_msgs) + [{"role": "user", "content": user_text}]
    full_ids = apply_chat_template(tokenizer, full_msgs, add_generation_prompt=False)

    empty_len = empty_ids.shape[-1]
    labels = torch.full_like(full_ids, fill_value=-100)
    if full_ids.shape[-1] > empty_len:
        labels[:, empty_len:] = full_ids[:, empty_len:]

    return full_ids, labels


def iter_user_turns(messages: List[Dict[str, str]]) -> List[Tuple[int, List[Dict[str, str]], str]]:
    """
    Returns list of (user_turn_idx, history_before_turn, user_text).
    """
    turns: List[Tuple[int, List[Dict[str, str]], str]] = []
    history: List[Dict[str, str]] = []
    u_idx = 0

    for m in messages:
        role = m.get("role")
        content = m.get("content", "")

        if role == "user":
            turns.append((u_idx, list(history), content))
            u_idx += 1
            history.append({"role": "user", "content": content})
        elif role == "assistant":
            history.append({"role": "assistant", "content": content})
        else:
            continue

    return turns


@torch.no_grad()
def score_one_user_turn(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    system_prompt: str,
    history: List[Dict[str, str]],
    user_text: str,
) -> Dict[str, Any]:
    input_ids, labels = build_inputs_for_user_turn(tokenizer, system_prompt, history, user_text)

    # truncate from left if too long
    if input_ids.shape[-1] > MAX_CONTEXT_TOKENS:
        overflow = input_ids.shape[-1] - MAX_CONTEXT_TOKENS
        input_ids = input_ids[:, overflow:]
        labels = labels[:, overflow:]

    attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.to(model.device)
    labels = labels.to(model.device)
    attention_mask = attention_mask.to(model.device)

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    avg_nll = float(out.loss.detach().cpu().item())
    num_tokens = int((labels != -100).sum().detach().cpu().item())

    ppl = math.exp(avg_nll) if avg_nll < 50 else float("inf")

    return {
        "avg_nll": avg_nll,
        "num_tokens": num_tokens,
        "ppl": ppl,
    }


def score_transcript_inv_none_llm1(
    transcript_path: Path,
    out_dir: Path,
    model_name: str = SCORER_MODEL,
) -> Path:
    """
    Minimal pipeline:
    - load transcript jsonl
    - keep only investigator_mode == "none"
    - score each USER turn
    - save parquet
    """
    ensure_dir(out_dir)

    rows = read_jsonl(transcript_path)
    tok, model = load_model(model_name)

    records: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        profile = row.get("profile", {}) or {}
        inv_mode = profile.get("investigator_mode") or row.get("investigator_mode")
        if inv_mode != "none":
            continue

        persona_id = row.get("persona_id")
        base_persona_id = profile.get("base_persona_id")
        style_id = profile.get("style_id")

        system_llm2 = row.get("system_llm2", "") or ""

        messages = row.get("messages") or []
        turns = iter_user_turns(messages)

        for turn_idx, history_before, user_text in turns:
            s = score_one_user_turn(tok, model, system_llm2, history_before, user_text)
            records.append({
                "conversation_idx": i,
                "persona_id": persona_id,
                "base_persona_id": base_persona_id,
                "style_id": style_id,
                "user_turn_idx": turn_idx,
                **s,
            })

    df = pd.DataFrame.from_records(records)

    out_path = out_dir / f"{transcript_path.stem}__inv_none__llm1_per_turn.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    out_path = score_transcript_inv_none_llm1(TRANSCRIPT_PATH, OUT_DIR, SCORER_MODEL)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()