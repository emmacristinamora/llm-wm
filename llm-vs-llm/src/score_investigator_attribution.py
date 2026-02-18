# src/score_investigator_attribution.py
#
# Computes investigator attribution + likelihood ratio metrics for investigator-mode transcripts.
#
# Inputs:
# - transcripts JSONL (your generated conversations; assistant messages have INVESTIGATION tag stripped)
# - investigator guesses JSONL (one row per assistant turn where <INVESTIGATION .../> was extracted)
#
# This script:
# 1) matches transcript rows with guess rows by conversation_id
# 2) drops unmatched rows (either missing transcript or missing guess)
# 3) filters by investigator_mode and base_persona_id
# 4) keeps the first N *guided* experiments (by ordinal rank, not raw experiment_index)
# 5) for each (conversation_id, turn_idx) guess event, computes:
#    (A) token-level attribution on the CURRENT GUESS:
#        - delta logprob(guess) when masking one user token at a time
#    (B) likelihood ratio (TRUE - GUESS):
#        - LR_total and LR_mean_per_token
#    (C) token influence on LR:
#        - delta LR when masking one user token at a time
#
# Output:
# - parquet with one row per guess event:
#   - metadata + guess/true/confidence
#   - logprobs + LR
#   - attribution payloads stored as JSON strings (safe for parquet)
#
# Notes:
# - "True" label is taken from transcript["profile"]["style_id"] (guided style task).
# - This works best for investigator_mode == "guided".
# - For unguided, "guess" is free-form and does not match style_id space; LR metrics are still computed
#   only if the "guess" accidentally equals a style_id. Otherwise LR fields are NaN.
# - Results are saved incrementally every --checkpoint_every events, so cluster timeouts
#   don't lose all progress. The final parquet overwrites the checkpoint.
#
# Performance:
# - Masking is expensive. Start with small --num_styles and/or --max_user_tokens_to_mask.
#
# Usage:
#   python -u src/score_investigator_attribution.py \
#     --transcripts_path src/data/conversations/transcripts.jsonl \
#     --guesses_path src/data/conversations/investigator_guesses_partial.jsonl \
#     --out_dir src/data/scores_partial \
#     --model Qwen/Qwen3-4B-Instruct-2507 \
#     --num_styles 5 \
#     --guided_only \
#     --max_user_tokens_to_mask 200
#

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import argparse
import json
import math

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# === CONFIG (defaults; override via CLI) ===

BASE_DIR = Path(__file__).parent.parent


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


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


# === MODEL ===

def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model(model_name: str, device_map: str = "auto") -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=pick_dtype(),
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


# === CHAT TEMPLATE HELPERS ===

def apply_chat_template_text(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    """
    Returns the serialized chat prompt as TEXT.
    Uses tokenizer.apply_chat_template if available; falls back to a simple format.

    Important: We want a base prompt that ends with an ASSISTANT generation marker,
    so we can append the assistant response text + investigation prefix ourselves.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        for kwargs in (
            {"messages": messages, "tokenize": False, "add_generation_prompt": add_generation_prompt},
            {"conversation": messages, "tokenize": False, "add_generation_prompt": add_generation_prompt},
        ):
            try:
                # Some tokenizers accept enable_thinking; if not, this will be ignored by TypeError path.
                return tokenizer.apply_chat_template(**kwargs, enable_thinking=False)
            except TypeError:
                try:
                    return tokenizer.apply_chat_template(**kwargs)
                except TypeError:
                    pass

    # fallback: plain format
    s = ""
    for m in messages:
        s += f"{m['role'].upper()}: {m['content']}\n"
    if add_generation_prompt:
        s += "ASSISTANT: "
    return s


# === UTILS: transcript alignment ===

def assistant_msg_index_from_turn_idx(turn_idx: int) -> int:
    """
    Generation script produces:
      idx 0: user0
      idx 1: assistant1
      idx 2: user1
      idx 3: assistant2
      ...
    So assistant turn t is at idx = 2*t - 1.
    """
    return 2 * int(turn_idx) - 1


def history_before_assistant_turn(messages: List[Dict[str, str]], assistant_turn_idx: int) -> Tuple[List[Dict[str, str]], str]:
    """
    Returns:
      - chat history messages (user/assistant) strictly BEFORE the assistant message at assistant_turn_idx
      - the assistant message content at that index (clean assistant text in your transcripts)
    """
    a_idx = assistant_msg_index_from_turn_idx(assistant_turn_idx)
    if a_idx < 0 or a_idx >= len(messages):
        raise IndexError(f"assistant_turn_idx={assistant_turn_idx} maps to msg_index={a_idx}, len(messages)={len(messages)}")

    assistant_text = messages[a_idx].get("content", "")
    hist = []
    for j in range(a_idx):
        role = messages[j].get("role")
        if role in ("user", "assistant"):
            hist.append({"role": role, "content": messages[j].get("content", "")})
    return hist, assistant_text


def iter_user_messages_only(history: List[Dict[str, str]]) -> List[Tuple[int, str]]:
    """
    Returns list of (user_msg_ord, user_text) in the given history.
    user_msg_ord counts user messages from 0..K-1 within this history slice.
    """
    out: List[Tuple[int, str]] = []
    k = 0
    for m in history:
        if m["role"] == "user":
            out.append((k, m.get("content", "")))
            k += 1
    return out


# === SCORING: logprob of a target label ===

@torch.no_grad()
def logprob_of_target_continuation(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    base_prompt_text: str,
    target_text: str,
    suffix_text: str,
    max_context_tokens: int = 8192,
) -> Dict[str, Any]:
    """
    Compute log P(target_text | base_prompt_text) by scoring the target token slice
    inside: base + target + suffix.

    We include suffix_text (constant) to reduce "end-of-text" artefacts.
    We compute:
      - total_logprob for target tokens
      - mean_logprob per target token
      - per_token_logprobs (list[float])
      - num_target_tokens
    """
    full_text = base_prompt_text + target_text + suffix_text

    base_ids = tokenizer(base_prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids

    base_len = int(base_ids.shape[-1])
    full_len = int(full_ids.shape[-1])

    # Target ids in isolation (context-sensitive tokenization can differ at boundaries,
    # but in practice with base ending exactly at the quote, this is stable enough.)
    target_ids = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids
    target_len = int(target_ids.shape[-1])

    # Safety: ensure we have room
    if base_len + target_len > full_len:
        # fallback: derive target_len as the remaining before suffix (approx)
        target_len = max(0, full_len - base_len)

    # Truncate from LEFT if too long
    if full_len > max_context_tokens:
        overflow = full_len - max_context_tokens
        full_ids = full_ids[:, overflow:]
        # Adjust base_len relative to truncated sequence
        base_len = max(0, base_len - overflow)
        full_len = int(full_ids.shape[-1])

    full_ids = full_ids.to(model.device)
    attn = torch.ones_like(full_ids, device=model.device)

    out = model(input_ids=full_ids, attention_mask=attn)
    logits = out.logits  # [1, T, V]

    # We want logprobs for token positions corresponding to the target tokens:
    # token at position t is predicted by logits at t-1.
    start = base_len
    end = min(full_len, base_len + target_len)
    if start >= end:
        return {
            "total_logprob": float("nan"),
            "mean_logprob": float("nan"),
            "num_target_tokens": 0,
            "per_token_logprobs": [],
        }

    # Compute log softmax over vocab for the relevant positions
    # Positions to score: start..end-1 (token indices), use logits at (start-1)..(end-2)
    # But if start==0, can't score first token (no context). That won't happen here.
    tok_ids = full_ids[0, start:end]  # [L]
    prev_logits = logits[0, start - 1:end - 1, :]  # [L, V]

    log_probs = torch.log_softmax(prev_logits, dim=-1)
    picked = log_probs.gather(1, tok_ids.unsqueeze(1)).squeeze(1)  # [L]

    per_tok = picked.detach().cpu().tolist()
    total_lp = float(sum(per_tok))
    mean_lp = float(total_lp / max(1, len(per_tok)))

    return {
        "total_logprob": total_lp,
        "mean_logprob": mean_lp,
        "num_target_tokens": int(len(per_tok)),
        "per_token_logprobs": per_tok,
    }


def score_guess_and_true(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    system_prompt: str,
    history: List[Dict[str, str]],
    assistant_text: str,
    guess_label: str,
    true_label: Optional[str],
    max_context_tokens: int,
) -> Dict[str, Any]:
    """
    Creates a base prompt that ends where the investigator would start printing the label,
    then scores guess_label and true_label as continuations.
    """
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history)

    # We want the assistant to "continue" here:
    base_prompt = apply_chat_template_text(tokenizer, msgs, add_generation_prompt=True)

    # The exact continuation format from your prompt:
    # assistant_text + "\n<INVESTIGATION guess=" + LABEL + '" confidence="50" />'
    # Use constant confidence to keep suffix identical across comparisons.
    prefix = assistant_text + "\n<INVESTIGATION guess=\""
    suffix = "\" confidence=\"50\" />"

    # Score guess
    guess_score = logprob_of_target_continuation(
        tokenizer, model,
        base_prompt_text=base_prompt + prefix,
        target_text=str(guess_label),
        suffix_text=suffix,
        max_context_tokens=max_context_tokens,
    )

    # Score true (if available)
    if true_label is None:
        true_score = None
    else:
        true_score = logprob_of_target_continuation(
            tokenizer, model,
            base_prompt_text=base_prompt + prefix,
            target_text=str(true_label),
            suffix_text=suffix,
            max_context_tokens=max_context_tokens,
        )

    out: Dict[str, Any] = {
        "guess_total_logprob": guess_score["total_logprob"],
        "guess_mean_logprob": guess_score["mean_logprob"],
        "guess_num_target_tokens": guess_score["num_target_tokens"],
        "guess_per_token_logprobs_json": dumps_json(guess_score["per_token_logprobs"]),
    }

    if true_score is None:
        out.update({
            "true_total_logprob": float("nan"),
            "true_mean_logprob": float("nan"),
            "true_num_target_tokens": 0,
            "true_per_token_logprobs_json": dumps_json([]),
            "lr_total_true_minus_guess": float("nan"),
            "lr_mean_true_minus_guess": float("nan"),
        })
    else:
        out.update({
            "true_total_logprob": true_score["total_logprob"],
            "true_mean_logprob": true_score["mean_logprob"],
            "true_num_target_tokens": true_score["num_target_tokens"],
            "true_per_token_logprobs_json": dumps_json(true_score["per_token_logprobs"]),
            "lr_total_true_minus_guess": float(true_score["total_logprob"] - guess_score["total_logprob"]),
            "lr_mean_true_minus_guess": float(true_score["mean_logprob"] - guess_score["mean_logprob"]),
        })

    return out


# === MASKING: token-level attribution ===

def mask_one_token_in_text(
    tokenizer: AutoTokenizer,
    text: str,
    token_idx: int,
    mask_text: str = "[REDACTED]",
) -> str:
    """
    Token-level masking via tokenization->decode. We replace exactly one token id with
    the tokenization of mask_text.

    This is not "perfect causal masking" but works well for short user utterances and
    gives meaningful token influence curves.
    """
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if token_idx < 0 or token_idx >= len(ids):
        return text

    mask_ids = tokenizer(mask_text, add_special_tokens=False).input_ids
    new_ids = ids[:token_idx] + mask_ids + ids[token_idx + 1:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def compute_user_token_attributions(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    system_prompt: str,
    history: List[Dict[str, str]],
    assistant_text: str,
    guess_label: str,
    true_label: Optional[str],
    max_context_tokens: int,
    max_user_tokens_to_mask: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Returns dict with:
      - base scores (guess logprob; LR if true provided)
      - token attribution lists as JSON:
          * delta_guess_total_logprob per masked token
          * delta_lr_mean per masked token (if true provided)
    """
    # Base scores
    base_scores = score_guess_and_true(
        tokenizer, model,
        system_prompt=system_prompt,
        history=history,
        assistant_text=assistant_text,
        guess_label=guess_label,
        true_label=true_label,
        max_context_tokens=max_context_tokens,
    )

    base_guess_lp = base_scores["guess_total_logprob"]
    base_lr_mean = base_scores["lr_mean_true_minus_guess"]

    # Enumerate user tokens across all user messages in history
    user_msgs = iter_user_messages_only(history)

    guess_attr_rows: List[Dict[str, Any]] = []
    lr_attr_rows: List[Dict[str, Any]] = []

    total_masked = 0

    for user_msg_ord, user_text in user_msgs:
        user_ids = tokenizer(user_text, add_special_tokens=False).input_ids
        user_toks = tokenizer.convert_ids_to_tokens(user_ids)

        for j in range(len(user_ids)):
            if max_user_tokens_to_mask is not None and total_masked >= max_user_tokens_to_mask:
                break

            # Build masked history
            masked_history: List[Dict[str, str]] = []
            cur_user_count = 0
            for m in history:
                if m["role"] != "user":
                    masked_history.append(m)
                    continue

                if cur_user_count == user_msg_ord:
                    masked_text = mask_one_token_in_text(tokenizer, m["content"], j)
                    masked_history.append({"role": "user", "content": masked_text})
                else:
                    masked_history.append(m)

                cur_user_count += 1

            masked_scores = score_guess_and_true(
                tokenizer, model,
                system_prompt=system_prompt,
                history=masked_history,
                assistant_text=assistant_text,
                guess_label=guess_label,
                true_label=true_label,
                max_context_tokens=max_context_tokens,
            )

            # (A) Token attribution for current guess
            masked_guess_lp = masked_scores["guess_total_logprob"]
            guess_attr_rows.append({
                "user_msg_ord": user_msg_ord,
                "token_idx": j,
                "token": user_toks[j] if j < len(user_toks) else None,
                "delta_guess_total_logprob": float(masked_guess_lp - base_guess_lp),
            })

            # (C) Token influence on LR (mean LR is recommended)
            if true_label is not None and not (math.isnan(base_lr_mean) or math.isnan(masked_scores["lr_mean_true_minus_guess"])):
                lr_attr_rows.append({
                    "user_msg_ord": user_msg_ord,
                    "token_idx": j,
                    "token": user_toks[j] if j < len(user_toks) else None,
                    "delta_lr_mean": float(masked_scores["lr_mean_true_minus_guess"] - base_lr_mean),
                })

            total_masked += 1

        if max_user_tokens_to_mask is not None and total_masked >= max_user_tokens_to_mask:
            break

    return {
        **base_scores,
        "guess_token_attr_json": dumps_json(guess_attr_rows),
        "lr_token_attr_json": dumps_json(lr_attr_rows),
        "num_masked_user_tokens": int(total_masked),
    }


# === CHECKPOINT HELPERS ===

def save_checkpoint(records: List[Dict[str, Any]], out_path: Path) -> None:
    """Save current records to parquet. Overwrites previous checkpoint."""
    if not records:
        return
    df = pd.DataFrame.from_records(records)
    df.to_parquet(out_path, index=False)
    print(f"[checkpoint] saved {len(records)} rows -> {out_path}", flush=True)


# === MATCHING: content-aware join ===

def resolve_experiment_indices_for_guided(
    transcripts: List[Dict[str, Any]],
    num_styles: Optional[int],
    base_persona_id: Optional[str],
) -> set:
    """
    Discover which experiment_index values correspond to guided transcripts,
    optionally filtered by base_persona_id. Then keep the first num_styles
    of them (sorted) to cover that many distinct styles.

    Returns a set of experiment_index values to keep.
    """
    guided_exp_indices: set = set()
    for t in transcripts:
        mode = t.get("investigator_mode", "")
        if mode != "guided":
            continue
        profile = t.get("profile", {}) or {}
        if base_persona_id is not None:
            bp = profile.get("base_persona_id") or ""
            if bp != base_persona_id:
                continue
        exp_idx = t.get("experiment_index")
        if exp_idx is not None:
            guided_exp_indices.add(int(exp_idx))

    # Sort and take the first num_styles
    sorted_indices = sorted(guided_exp_indices)
    if num_styles is not None:
        sorted_indices = sorted_indices[:num_styles]

    kept = set(sorted_indices)
    print(f"[info] guided experiment_indices found: {sorted(guided_exp_indices)}", flush=True)
    print(f"[info] keeping {len(kept)} guided experiment_indices: {sorted(kept)}", flush=True)
    return kept


# === PIPELINE ===

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--transcripts_path", type=str, required=True)
    p.add_argument("--guesses_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--max_context_tokens", type=int, default=8192)

    # --- changed: replace --num_experiments with --num_styles ---
    p.add_argument("--num_styles", type=int, default=None,
                   help="Keep the first N guided styles (by experiment_index rank). "
                        "Default: all guided styles found.")
    p.add_argument("--guided_only", action="store_true",
                   help="If set, keep only investigator_mode == guided.")
    p.add_argument("--base_persona_id", type=str, default=None,
                   help="If set, restrict to transcripts with this base_persona_id. "
                        "Example: bp_tech_starter")
    p.add_argument("--max_events", type=int, default=None,
                   help="Optional cap on number of guess events processed.")

    p.add_argument("--max_user_tokens_to_mask", type=int, default=None,
                   help="Optional cap on number of user tokens masked per event (for speed).")

    # --- new: incremental checkpoint ---
    p.add_argument("--checkpoint_every", type=int, default=10,
                   help="Save intermediate results every N processed events.")

    # --- back-compat: accept but ignore --num_experiments ---
    p.add_argument("--num_experiments", type=int, default=None,
                   help="DEPRECATED. Use --num_styles instead. If provided and --num_styles "
                        "is not, this value is forwarded to --num_styles for back-compat.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Back-compat: --num_experiments -> --num_styles
    if args.num_styles is None and args.num_experiments is not None:
        print(f"[warn] --num_experiments is deprecated; interpreting as --num_styles={args.num_experiments}", flush=True)
        args.num_styles = args.num_experiments

    transcripts_path = Path(args.transcripts_path)
    guesses_path = Path(args.guesses_path)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    transcripts = read_jsonl(transcripts_path)
    guesses = read_jsonl(guesses_path)

    # --- Resolve which experiment_indices to keep ---
    # Instead of a blanket "experiment_index < N", discover the actual guided
    # experiment_indices from the transcripts and keep the first --num_styles.
    if args.guided_only:
        keep_exp_indices = resolve_experiment_indices_for_guided(
            transcripts, args.num_styles, args.base_persona_id,
        )
    else:
        # If not guided_only, fall back to keeping everything (or first N unique indices)
        all_indices = sorted({int(t.get("experiment_index", -1)) for t in transcripts if t.get("experiment_index") is not None})
        if args.num_styles is not None:
            all_indices = all_indices[:args.num_styles]
        keep_exp_indices = set(all_indices)
        print(f"[info] keeping experiment_indices: {sorted(keep_exp_indices)}", flush=True)

    # Index transcripts by conversation_id
    t_by_cid: Dict[str, Dict[str, Any]] = {}
    for i, row in enumerate(transcripts):
        cid = row.get("conversation_id")
        if cid:
            t_by_cid[str(cid)] = {"row": row, "conversation_idx": i}

    # Filter guess rows: match transcript exists + experiment_index in kept set + optional guided_only
    matched: List[Dict[str, Any]] = []
    dropped_no_transcript = 0
    dropped_experiment_limit = 0
    dropped_guided = 0
    dropped_base_persona = 0

    for g in guesses:
        cid = str(g.get("conversation_id", ""))
        if cid not in t_by_cid:
            dropped_no_transcript += 1
            continue

        exp_idx = g.get("experiment_index")
        if exp_idx is None or int(exp_idx) not in keep_exp_indices:
            dropped_experiment_limit += 1
            continue

        if args.guided_only and g.get("investigator_mode") != "guided":
            dropped_guided += 1
            continue

        # Optional base_persona_id filter (check against transcript profile)
        if args.base_persona_id is not None:
            t_row = t_by_cid[cid]["row"]
            profile = t_row.get("profile", {}) or {}
            bp = profile.get("base_persona_id") or ""
            if bp != args.base_persona_id:
                dropped_base_persona += 1
                continue

        matched.append(g)

    print(f"[info] transcripts={len(transcripts)} guesses={len(guesses)}", flush=True)
    print(f"[info] matched_guess_events={len(matched)}", flush=True)
    print(f"[info] dropped_no_transcript={dropped_no_transcript}", flush=True)
    print(f"[info] dropped_experiment_filter={dropped_experiment_limit}", flush=True)
    if args.guided_only:
        print(f"[info] dropped_not_guided={dropped_guided}", flush=True)
    if args.base_persona_id is not None:
        print(f"[info] dropped_base_persona_mismatch={dropped_base_persona}", flush=True)

    if args.max_events is not None:
        matched = matched[: int(args.max_events)]
        print(f"[info] capped matched events to max_events={args.max_events}", flush=True)

    # Summarize what we're about to score
    styles_in_matched = set()
    for g in matched:
        cid = str(g["conversation_id"])
        t_row = t_by_cid[cid]["row"]
        profile = t_row.get("profile", {}) or {}
        sid = profile.get("style_id")
        if sid:
            styles_in_matched.add(sid)
    print(f"[info] styles to score: {sorted(styles_in_matched)}", flush=True)

    # Output path
    n_styles_tag = args.num_styles if args.num_styles is not None else "all"
    bp_tag = args.base_persona_id or "all_personas"
    out_path = out_dir / f"{transcripts_path.stem}__investigator_attr__{bp_tag}__n{n_styles_tag}.parquet"
    print(f"[info] output path: {out_path}", flush=True)

    if len(matched) == 0:
        print("[warn] no matched events to score. Writing empty parquet.", flush=True)
        pd.DataFrame().to_parquet(out_path, index=False)
        return

    # Load model once
    tok, model = load_model(args.model, device_map=args.device_map)

    records: List[Dict[str, Any]] = []

    for k, g in enumerate(matched):
        cid = str(g["conversation_id"])
        t = t_by_cid[cid]["row"]
        conversation_idx = int(t_by_cid[cid]["conversation_idx"])

        profile = t.get("profile", {}) or {}
        inv_mode_true = profile.get("investigator_mode") or t.get("investigator_mode")
        style_true = profile.get("style_id")  # guided target
        base_persona_true = profile.get("base_persona_id")
        persona_id = t.get("persona_id", g.get("persona_id"))

        turn_idx = int(g.get("turn_idx"))
        guess_label = str(g.get("guess", ""))
        confidence = int(g.get("confidence", -1))

        system_prompt = t.get("system_llm2", "") or ""
        messages = t.get("messages") or []

        try:
            history, assistant_text = history_before_assistant_turn(messages, turn_idx)
        except Exception as e:
            # cannot align this guess event to transcript structure
            print(f"[warn] skip cid={cid} turn_idx={turn_idx} due to alignment error: {e}", flush=True)
            continue

        # Decide whether LR can be computed:
        # - For guided: true is style_id (e.g., st_formal)
        # - For unguided: guess is free-form; LR only makes sense if guess_label equals one of your style_ids
        true_label_for_lr: Optional[str] = str(style_true) if style_true is not None else None

        # Compute scores + attributions
        scores = compute_user_token_attributions(
            tok, model,
            system_prompt=system_prompt,
            history=history,
            assistant_text=assistant_text,
            guess_label=guess_label,
            true_label=true_label_for_lr,
            max_context_tokens=int(args.max_context_tokens),
            max_user_tokens_to_mask=args.max_user_tokens_to_mask,
        )

        records.append({
            "conversation_id": cid,
            "conversation_idx": conversation_idx,

            "persona_id": persona_id,
            "base_persona_id": base_persona_true,
            "style_id": style_true,
            "investigator_mode_transcript": inv_mode_true,

            "experiment_index": int(g.get("experiment_index")),
            "replicate_index": int(g.get("replicate_index")),
            "turn_idx": turn_idx,

            "guess": guess_label,
            "confidence": confidence,

            # Store assistant text (optional; can be large)
            "assistant_text": assistant_text,

            # Scores:
            **scores,
        })

        processed = k + 1
        if processed % args.checkpoint_every == 0:
            print(f"[progress] processed {processed}/{len(matched)} guess events", flush=True)
            save_checkpoint(records, out_path)

    # Final save
    save_checkpoint(records, out_path)
    print(f"[done] wrote {out_path} rows={len(records)}", flush=True)


if __name__ == "__main__":
    main()