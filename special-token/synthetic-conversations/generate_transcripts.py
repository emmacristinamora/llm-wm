# synthetic-conversations/generate_transcripts.py
#
# Generate transcripts from data/experiments.jsonl by:
# 1) Loading system_llm1 (user persona/style writing rules)
# 2) Loading system_llm2 (assistant rules)
# 3) Starting from init_user_message
# 4) Alternating user/assistant turns for num_turns user turns total
#
# Output row schema (data/transcripts.jsonl):
# { persona_id, messages, profile, + provenance }
#
# Notes:
# - enable_thinking=False prevents <think> blocks from leaking into turn text
# - attention_mask is always set to avoid eos-as-pad warnings with Qwen
# - --dry_run mode stubs out model loading for pipeline/logic testing


# === 1. IMPORTS & CONFIG ===

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import argparse
import json
import random
import re
import time
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# model config
USER_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ASSISTANT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


# === 2. UTILS ===

# IO
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


# text cleanup
THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
SPECIAL_TOKENS_RE = re.compile(r"(<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|<\|eot_id\|>)")

def strip_reasoning(text: str) -> str:
    if not text:
        return text
    text = THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>")[0]
    text = SPECIAL_TOKENS_RE.sub("", text)
    return text.strip()


# path helpers
def resolve_path(p: str) -> Path:
    # if absolute keep it; else resolve relative to this script's directory
    path = Path(p)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


# args
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--experiments_path", type=str, default="data/experiments.jsonl")
    p.add_argument("--output_path", type=str, default="data/transcripts.jsonl")

    p.add_argument("--user_model", type=str, default=USER_MODEL_NAME)
    p.add_argument("--assistant_model", type=str, default=ASSISTANT_MODEL_NAME)

    p.add_argument("--num_experiments", type=int, default=None,
                   help="How many experiment rows to use (default: all).")
    p.add_argument("--conversations_per_experiment", type=int, default=1,
                   help="How many conversations (N) to generate per experiment row.")

    p.add_argument("--num_turns", type=int, default=6,
                   help="Number of user turns (incl. init_user_message).")

    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    # generation knobs
    p.add_argument("--assistant_max_new_tokens", type=int, default=900)
    p.add_argument("--assistant_temp", type=float, default=0.7)
    p.add_argument("--user_max_new_tokens", type=int, default=400)
    p.add_argument("--user_temp", type=float, default=0.8)

    p.add_argument("--verbose", action="store_true")
    p.add_argument("--print_every", type=int, default=1)

    p.add_argument("--dry_run", action="store_true",
                   help="Skip model loading; use stub replies to test pipeline logic.")

    return p.parse_args()


# === 3. MODEL ===

def load_model(name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # pick a reasonable dtype (bfloat16 if supported, else float16)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def generate_reply(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    # try enable_thinking=False (Qwen3+) to prevent <think> blocks leaking in
    try:
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    if isinstance(enc, dict):
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        else:
            attention_mask = torch.ones_like(input_ids)
    else:
        # apply_chat_template returned a plain tensor — build attention mask manually
        # (Qwen uses eos as pad, so transformers cannot infer the mask reliably)
        input_ids = enc.to(model.device)
        attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][input_ids.shape[1]:]

    # decode without skipping specials first (skip=True can yield empty on Qwen)
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
    if not decoded:
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return strip_reasoning(decoded)


# === 4. VALIDATION ===

def validate_experiment_row(exp: Dict[str, Any]) -> None:
    required = ["persona_id", "profile", "system_llm1", "system_llm2", "init_user_message"]
    missing = [k for k in required if k not in exp]
    if missing:
        raise ValueError(f"experiment row missing keys: {missing}")
    if not isinstance(exp["profile"], dict):
        raise ValueError("profile must be a dict")
    for k in ["system_llm1", "system_llm2", "init_user_message"]:
        if not isinstance(exp.get(k), str) or not exp[k].strip():
            raise ValueError(f"{k} must be a non-empty string")


# === 5. MAIN GENERATION ===

# user turn constraint injected into every user-model prompt
TASK_PROMPT_USER = """\
You are the USER in this conversation (not the assistant).
Write ONLY the user's next message responding to the assistant.

Constraints:
- Write 3-6 sentences (aim ~80-140 words). If a shorter reply is truly natural, keep it >= 2 sentences.
- Stay in American English.
- You may ask a question, accept/reject, request changes, add a constraint, or clarify.
- Include at least ONE of the following in each message:
  (a) a constraint or evaluation criterion, OR
  (b) a trade-off / “what would you assume?” question, OR
  (c) a concrete example or edge case, OR
  (d) a quick recap (“So you're saying…”) before a follow-up.
- Do NOT write an answer/solution or continue the assistant's output.
- Do NOT write "Assistant:" or anything except the user message.
""".strip()

def generate_conversation(
    user_model,
    user_tok,
    assistant_model,
    assistant_tok,
    exp_row: Dict[str, Any],
    num_turns: int,
    user_max_new_tokens: int,
    user_temp: float,
    assistant_max_new_tokens: int,
    assistant_temp: float,
    verbose: bool = False,
    tag: str = "",
    dry_run: bool = False,
) -> Dict[str, Any]:
    validate_experiment_row(exp_row)

    persona_id = exp_row["persona_id"]
    profile = exp_row["profile"]
    system_llm1 = exp_row["system_llm1"].strip()
    system_llm2 = exp_row["system_llm2"].strip()
    seed_prompt = exp_row["init_user_message"].strip()

    def _gen_user(msgs):
        if dry_run:
            return "[DRY_RUN user reply]"
        return generate_reply(user_model, user_tok, msgs,
                              max_new_tokens=user_max_new_tokens, temperature=user_temp)

    def _gen_assistant(msgs):
        if dry_run:
            return "[DRY_RUN assistant reply]"
        return generate_reply(assistant_model, assistant_tok, msgs,
                              max_new_tokens=assistant_max_new_tokens, temperature=assistant_temp)

    messages: List[Dict[str, str]] = [{"role": "user", "content": seed_prompt}]

    # first assistant reply
    log(f"[{tag}] assistant_turn=1 generating...", verbose)
    t0 = time.time()
    assistant_reply = _gen_assistant([{"role": "system", "content": system_llm2}] + messages)
    log(f"[{tag}] assistant_turn=1 done ({time.time()-t0:.1f}s)", verbose)
    messages.append({"role": "assistant", "content": assistant_reply})

    assistant_turn_idx = 1

    # alternate user/assistant for remaining turns
    for _ in range(num_turns - 1):
        # user turn (sees only system_llm1 + last assistant message)
        user_msgs = [
            {"role": "system", "content": system_llm1},
            {"role": "user", "content": (
                TASK_PROMPT_USER
                + "\n\nAssistant just said:\n"
                + messages[-1]["content"]
                + "\n\nNow write the user's next message:"
            )},
        ]
        log(f"[{tag}] user_turn generating...", verbose)
        t0 = time.time()
        user_reply = _gen_user(user_msgs)
        log(f"[{tag}] user_turn done ({time.time()-t0:.1f}s)", verbose)
        messages.append({"role": "user", "content": user_reply})

        # assistant turn (sees full transcript)
        assistant_turn_idx += 1
        log(f"[{tag}] assistant_turn={assistant_turn_idx} generating...", verbose)
        t0 = time.time()
        assistant_reply = _gen_assistant([{"role": "system", "content": system_llm2}] + messages)
        log(f"[{tag}] assistant_turn={assistant_turn_idx} done ({time.time()-t0:.1f}s)", verbose)
        messages.append({"role": "assistant", "content": assistant_reply})

    return {
        "persona_id": persona_id,
        "messages": messages,
        "profile": profile,
        # provenance
        "conversation_id": str(uuid.uuid4()),
        "system_llm1": system_llm1,
        "system_llm2": system_llm2,
        "seed_prompt": seed_prompt,
    }


# === 6. MAIN ===

def main() -> None:
    args = parse_args()

    experiments_path = resolve_path(args.experiments_path)
    output_path = resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # load experiments
    rows = read_jsonl(experiments_path)
    if args.shuffle:
        random.shuffle(rows)
    if args.num_experiments is not None:
        rows = rows[: args.num_experiments]
    if not rows:
        raise ValueError(f"no experiment rows loaded from {experiments_path}")

    total_exps = len(rows)
    total_convs = total_exps * args.conversations_per_experiment

    print(f"[load] {total_exps} experiment rows from {experiments_path}", flush=True)
    print(f"[config] conversations_per_experiment={args.conversations_per_experiment} "
          f"total_convs={total_convs} num_turns={args.num_turns}", flush=True)
    print(f"[out] {output_path}", flush=True)

    if args.dry_run:
        print("[dry_run] skipping model loading — stub replies will be used", flush=True)
        user_model = user_tok = assistant_model = assistant_tok = None
    else:
        print(f"[model] loading user model: {args.user_model}", flush=True)
        user_model, user_tok = load_model(args.user_model)
        print(f"[model] loading assistant model: {args.assistant_model}", flush=True)
        assistant_model, assistant_tok = load_model(args.assistant_model)

    num_written = 0

    with output_path.open("w", encoding="utf-8") as f_out:
        for exp_idx, exp in enumerate(rows):
            for rep in range(args.conversations_per_experiment):
                # deterministic per-(experiment, replicate) seed
                seed_here = args.seed + exp_idx * 100000 + rep
                random.seed(seed_here)
                torch.manual_seed(seed_here)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_here)

                global_idx = exp_idx * args.conversations_per_experiment + rep + 1

                if global_idx % args.print_every == 0:
                    log(f"[START] {global_idx}/{total_convs} "
                        f"exp_idx={exp_idx} rep={rep} "
                        f"persona_id={exp.get('persona_id', 'UNKNOWN')}",
                        args.verbose)

                try:
                    conv = generate_conversation(
                        user_model, user_tok,
                        assistant_model, assistant_tok,
                        exp_row=exp,
                        num_turns=args.num_turns,
                        user_max_new_tokens=args.user_max_new_tokens,
                        user_temp=args.user_temp,
                        assistant_max_new_tokens=args.assistant_max_new_tokens,
                        assistant_temp=args.assistant_temp,
                        verbose=args.verbose,
                        tag=f"exp{exp_idx}_rep{rep}",
                        dry_run=args.dry_run,
                    )

                    conv["experiment_index"] = exp_idx
                    conv["replicate_index"] = rep
                    conv["user_model_name"] = args.user_model
                    conv["assistant_model_name"] = args.assistant_model
                    conv["experiments_path"] = str(experiments_path)

                    f_out.write(json.dumps(conv, ensure_ascii=False) + "\n")
                    f_out.flush()
                    num_written += 1

                    if global_idx % args.print_every == 0:
                        log(f"[DONE ] {global_idx}/{total_convs} "
                            f"wrote conversation_id={conv['conversation_id']}",
                            args.verbose)

                except Exception as e:
                    print(f"[skip] exp_idx={exp_idx} rep={rep} error: {e}", flush=True)

    print(f"[done] wrote {num_written} conversations -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
