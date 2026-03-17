# src/generate_transcripts.py

# === 1. IMPORTS ===

import json
from pyexpat.errors import messages
import time
import random
import re
import uuid
import argparse
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# config
USER_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ASSISTANT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


# === 2. HELPERS ===

def get_repo_root() -> Path:
    """
    Assumes this script is located in repo/src/.
    repo_root = parent of src/
    """
    here = Path(__file__).resolve()
    # .../repo/src/this_script.py -> .../repo/src -> .../repo
    return here.parent.parent

def resolve_path(p: str) -> Path:
    """
    If path is absolute, keep it.
    Else resolve relative to repo root.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    return get_repo_root() / path

def parse_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser()

    p.add_argument(
        "--experiments_path",
        type=str,
        default="experiments.jsonl",
        help="Path to experiments.jsonl (default: repo_root/experiments.jsonl)",
    )

    p.add_argument("--user_model", type=str, default=USER_MODEL_NAME)
    p.add_argument("--assistant_model", type=str, default=ASSISTANT_MODEL_NAME)

    p.add_argument(
        "--num_experiments",
        type=int,
        default=None,
        help="How many experiment rows to use (default: all).",
    )
    p.add_argument(
        "--conversations_per_experiment",
        type=int,
        default=50,
        help="How many conversations to generate PER experiment row.",
    )

    p.add_argument("--num_turns", type=int, default=6,
                   help="Number of user turns (incl. init_user_message)")

    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--output_path",
        type=str,
        default=f"src/data/conversations/with_cat_persona_{timestamp}.jsonl",
        help="Where to write conversations JSONL (relative to repo root unless absolute).",
    )

    p.add_argument(
        "--inv_output_path",
        type=str,
        default=f"src/data/conversations/investigation_meta_{timestamp}.jsonl",
        help="Where to write extracted <INVESTIGATION .../> lines (relative to repo root unless absolute).",
    )

    # generation knobs
    p.add_argument("--assistant_max_new_tokens", type=int, default=1000)
    p.add_argument("--assistant_temp", type=float, default=0.7)
    p.add_argument("--user_max_new_tokens", type=int, default=150)
    p.add_argument("--user_temp", type=float, default=0.8)
    p.add_argument("--verbose", action="store_true", help="Print progress during generation.")
    p.add_argument("--print_every", type=int, default=1, help="Print every N conversations.")
    return p.parse_args()


# === 3. UTILS ===

THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
SPECIAL_TOKENS_RE = re.compile(
    r"(<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|<\|eot_id\|>)"
)

def strip_reasoning(text: str) -> str:
    if not text:
        return text

    # remove Qwen reasoning blocks
    text = THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>")[0]

    # remove literal special tokens that sometimes leak into decoded text
    text = SPECIAL_TOKENS_RE.sub("", text)

    return text.strip()

def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def log(msg: str, enabled: bool = True):
    if enabled:
        print(msg, flush=True)

def load_model(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tok


def generate_reply(model, tokenizer, messages: List[Dict[str, str]],
                   max_new_tokens: int, temperature: float) -> str:
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(enc, dict):
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
    else:
        input_ids = enc.to(model.device)
        attention_mask = None

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,  # prevents instant-empty generations
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][input_ids.shape[1]:]

    # decode WITHOUT skipping special tokens first (skip_special_tokens=True can yield empty)
    decoded_raw = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

    # fallback decode
    if not decoded_raw:
        decoded_raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return strip_reasoning(decoded_raw)


# === 4. ASSISTANT PARSING ===

INV_TAIL_RE = re.compile(
    r'<INVESTIGATION\s+guess="([^"]+)"\s+confidence="(\d{1,3})"\s*/>\s*$'
)

def parse_assistant(text: str) -> Tuple[str, Optional[Dict]]:
    """
    Extracts a trailing <INVESTIGATION .../> tag even if it is:
      - on its own line, OR
      - appended at the end of the last sentence.
    """
    if not text:
        return text, None

    t = text.strip()

    m = INV_TAIL_RE.search(t)
    if not m:
        return t, None

    guess = m.group(1).strip()
    conf = max(0, min(100, int(m.group(2))))

    # remove the tag (and any whitespace before it)
    clean_text = t[: m.start()].rstrip()

    parsed = {"guess": guess, "confidence": conf, "raw_line": m.group(0).strip()}
    return clean_text, parsed


# === 5. MAIN GENERATION ===

def validate_experiment_row(exp: Dict):
    required = ["persona_id", "profile", "system_llm1", "system_llm2", "init_user_message"]
    missing = [k for k in required if k not in exp]
    if missing:
        raise ValueError(f"Experiment row missing keys: {missing}")

    if not isinstance(exp["profile"], dict):
        raise ValueError("profile must be a dict")

    if not isinstance(exp["system_llm1"], str) or not exp["system_llm1"].strip():
        raise ValueError("system_llm1 must be a non-empty string")
    if not isinstance(exp["system_llm2"], str) or not exp["system_llm2"].strip():
        raise ValueError("system_llm2 must be a non-empty string")
    if not isinstance(exp["init_user_message"], str) or not exp["init_user_message"].strip():
        raise ValueError("init_user_message must be a non-empty string")

def infer_investigator_mode(persona_id: str, profile: Dict) -> Optional[str]:
    # persona_id ends with __inv_none / __inv_guided / __inv_unguided
    try:
        return persona_id.split("__")[-1].replace("inv_", "")
    except Exception:
        return profile.get("investigator_mode")

def summarize_history_for_user(messages: list[dict]) -> str:
    user_questions = []
    assistant_points = []

    for m in messages:
        if m["role"] == "user":
            user_questions.append(m["content"].strip())
        elif m["role"] == "assistant":
            assistant_points.append(m["content"].strip())

    # keep only recent items to avoid huge prompts
    recent_user = user_questions[-4:]
    recent_assistant = assistant_points[-3:]

    out = "Previously asked by the user:\n"
    for i, x in enumerate(recent_user, 1):
        out += f"{i}. {x}\n"

    out += "\nRecently answered by the assistant:\n"
    for i, x in enumerate(recent_assistant, 1):
        short_x = x.replace("\n", " ")
        if len(short_x) > 220:
            short_x = short_x[:220] + "..."
        out += f"{i}. {short_x}\n"

    return out.strip()

def generate_conversation_with_persona(
    user_model, user_tok,
    assistant_model, assistant_tok,
    exp_row: Dict,
    num_turns: int,
    user_max_new_tokens: int,
    user_temp: float,
    assistant_max_new_tokens: int,
    assistant_temp: float,
    verbose: bool = False,
    tag: str = "",
) -> Tuple[Dict, List[Dict]]:
    """
    Returns:
      - conversation dict (with clean assistant messages)
      - inv_meta list (one entry per assistant turn where <INVESTIGATION.../> was extracted)
    """
    # validate & unpack experiment row
    validate_experiment_row(exp_row)

    persona_id = exp_row["persona_id"]
    profile = exp_row["profile"]
    system_llm1 = exp_row["system_llm1"].strip()    # user system prompt (persona/style)
    system_llm2 = exp_row["system_llm2"].strip()    # assistant system prompt
    seed_prompt = exp_row["init_user_message"].strip()

    inv_mode = infer_investigator_mode(persona_id, profile)

    # initialize transcript with seed prompt aka first user message
    messages = [{"role": "user", "content": seed_prompt}]
    inv_meta: List[Dict] = []

    # first assistant reply
    assistant_messages = [{"role": "system", "content": system_llm2}] + messages
    assistant_reply_raw = generate_reply(
        assistant_model, assistant_tok, assistant_messages,
        max_new_tokens=assistant_max_new_tokens,
        temperature=assistant_temp
    )

    # parse if investigator mode is on (guided or unguided)
    assistant_reply_clean, inv = parse_assistant(assistant_reply_raw) if inv_mode in ("guided", "unguided") else (assistant_reply_raw.strip(), None)
    messages.append({"role": "assistant", "content": assistant_reply_clean})

    if inv is not None:
        inv_meta.append({
            "turn_idx": 1,  # assistant turn count starting at 1
            "guess": inv["guess"],
            "confidence": inv["confidence"],
            "raw_line": inv["raw_line"],
        })
    
    # additional constraint so the user doesn't become assistant
    TASK_PROMPT_USER = """\
You are the USER in this conversation (not the assistant).
Write ONLY the user's next message.

Requirements:
- Write 1 or 2 sentences, max 50 words.
- Respond naturally to the FULL conversation so far, not just the last assistant message.
- Do NOT repeat or restate a question that was already asked earlier.
- Use the conversation summary below to avoid repetition.
- Do NOT ask for the same example, definition, or explanation again in different words.
- Your next message should do one of these:
  1) ask a narrower follow-up,
  2) introduce a concrete constraint or scenario,
  3) ask for a comparison,
  4) ask for a practical recommendation,
  5) shift to a closely related sub-question.
- If the last assistant answer already addressed the most recent question, move the conversation forward.
- Do NOT answer your own question.
- Do NOT write "Assistant:" or anything except the user message.
""".strip()

    assistant_turn_idx = 1

    # alternate user/assistant until we have num_turns user messages total
    for _ in range(num_turns - 1):
        history_summary = summarize_history_for_user(messages)

        user_messages = [
            {"role": "system", "content": system_llm1},
            {"role": "user", "content": (
                TASK_PROMPT_USER
                + "\n\nConversation progress so far:\n"
                + history_summary
                + "\n\nAssistant just said:\n"
                + messages[-1]["content"]
                + "\n\nWrite the user's next message. It must not repeat an already-asked question."
            )},
        ]
        """
        # user reacts based on persona instructions + last assistant message only
        user_messages = [
            {"role": "system", "content": system_llm1},
            {"role": "user", "content": (
                TASK_PROMPT_USER
                + "\n\nFull conversation so far:\n"
                + transcript_so_far
                + "\n\nNow write the user's next message:"
            )},
        ]
        """

        if verbose:
            print(f"[{tag}] user_turn generating...", flush=True)
        t0 = time.time()

        user_reply = generate_reply(
            user_model, user_tok, user_messages,
            max_new_tokens=user_max_new_tokens,
            temperature=user_temp
        )

        if verbose:
            print(f"[{tag}] user_turn done ({time.time()-t0:.1f}s)", flush=True)

        messages.append({"role": "user", "content": user_reply})

        # assistant sees system_llm2 + full transcript
        assistant_messages = [{"role": "system", "content": system_llm2}] + messages

        if verbose:
            print(f"[{tag}] assistant_turn={assistant_turn_idx+1} generating...", flush=True)
        t0 = time.time()

        assistant_reply_raw = generate_reply(
            assistant_model, assistant_tok, assistant_messages,
            max_new_tokens=assistant_max_new_tokens,
            temperature=assistant_temp
        )

        if verbose:
            print(f"[{tag}] assistant_turn={assistant_turn_idx+1} done ({time.time()-t0:.1f}s)", flush=True)

        assistant_turn_idx += 1
        assistant_reply_clean, inv = parse_assistant(assistant_reply_raw) if inv_mode in ("guided", "unguided") else (assistant_reply_raw.strip(), None)
        messages.append({"role": "assistant", "content": assistant_reply_clean})

        if inv is not None:
            inv_meta.append({
                "turn_idx": assistant_turn_idx,
                "guess": inv["guess"],
                "confidence": inv["confidence"],
                "raw_line": inv["raw_line"],
            })

    conv = {
        "conversation_id": str(uuid.uuid4()),
        "condition": "with_persona",
        "persona_id": persona_id,
        "investigator_mode": inv_mode,
        "profile": profile,
        "seed_prompt": seed_prompt,
        "system_llm1": system_llm1,
        "system_llm2": system_llm2,
        "persona_text": system_llm1,  # scoring expects this name
        "messages": messages,
    }

    return conv, inv_meta


def main():
    # parse args and set up paths
    args = parse_args()

    experiments_path = resolve_path(args.experiments_path)
    output_path = resolve_path(args.output_path)
    inv_output_path = resolve_path(args.inv_output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inv_output_path.parent.mkdir(parents=True, exist_ok=True)

    # load experiments
    rows = read_jsonl(experiments_path)
    if args.shuffle:
        random.shuffle(rows)

    if args.num_experiments is not None:
        rows = rows[: args.num_experiments]

    if len(rows) == 0:
        raise ValueError("No experiment rows loaded")

    # load models once
    user_model, user_tok = load_model(args.user_model)
    assistant_model, assistant_tok = load_model(args.assistant_model)

    total_exps = len(rows)
    total_convs = total_exps * args.conversations_per_experiment
    log(f"[INIT] experiments={total_exps} convs_per_exp={args.conversations_per_experiment} total_convs={total_convs}", args.verbose)
    log(f"[INIT] user_model={args.user_model} assistant_model={args.assistant_model}", args.verbose)
    log(f"[INIT] turns={args.num_turns} user_max_new={args.user_max_new_tokens} assistant_max_new={args.assistant_max_new_tokens}", args.verbose)

    num_written = 0
    num_inv_written = 0

    # open files once (stream write so we don't keep everything in RAM)
    with output_path.open("w", encoding="utf-8") as f_conv, inv_output_path.open("w", encoding="utf-8") as f_inv:
        for exp_idx, exp in enumerate(rows):
            for rep in range(args.conversations_per_experiment):
                # per-(experiment,row) deterministic seed
                seed_here = args.seed + exp_idx * 100000 + rep
                random.seed(seed_here)
                torch.manual_seed(seed_here)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_here)
                try:
                    global_idx = exp_idx * args.conversations_per_experiment + rep + 1
                    if args.verbose and (global_idx % args.print_every == 0):
                        persona_id = exp.get("persona_id", "UNKNOWN")
                        inv_mode = infer_investigator_mode(persona_id, exp.get("profile", {}))
                        log(f"[START] {global_idx}/{total_convs} exp_idx={exp_idx} rep={rep} persona_id={persona_id} inv_mode={inv_mode}", True)
                
                    # generate one conversation for this experiment row
                    conv, inv_meta = generate_conversation_with_persona(
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
                    )

                    # add provenance
                    conv["experiments_path"] = str(experiments_path)
                    conv["user_model_name"] = args.user_model
                    conv["assistant_model_name"] = args.assistant_model
                    conv["experiment_index"] = exp_idx
                    conv["replicate_index"] = rep

                    f_conv.write(json.dumps(conv, ensure_ascii=False) + "\n")
                    num_written += 1

                    # write inv meta rows (if any)
                    if inv_meta:
                        for rec in inv_meta:
                            rec_out = {
                                "conversation_id": conv["conversation_id"],
                                "persona_id": conv["persona_id"],
                                "investigator_mode": conv["investigator_mode"],
                                "experiment_index": exp_idx,
                                "replicate_index": rep,
                                **rec,
                            }
                            f_inv.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                            num_inv_written += 1

                    if args.verbose and (global_idx % args.print_every == 0):
                        log(f"[DONE ] {global_idx}/{total_convs} wrote conversation_id={conv['conversation_id']}", True)

                except Exception as e:
                    print(f"Skipping (exp_idx={exp_idx}, rep={rep}) due to error: {e}")

    print(f"Wrote {num_written} conversations -> {output_path}")
    print(f"Wrote {num_inv_written} investigation records -> {inv_output_path}")


if __name__ == "__main__":
    main()