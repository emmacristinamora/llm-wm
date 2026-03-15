# synthetic-conversations/generate_experiments.py
#
# Generate experiments.jsonl for synthetic conversations by:
# 1) Reading base personas, styles, and topics from config/personas.yaml
# 2) Reading prompt templates from config/prompts.yaml
# 3) Using a local HF model (e.g., Qwen) to generate:
#    - system_llm1 (user-writing system prompt) ONCE per (base_persona, style)
#    - init_user_message N times per (base_persona, style, topic)
# 4) Writing rows to data/experiments.jsonl:
#    { persona_id, system_llm1, system_llm2, init_user_message, profile }
#
# Notes:
# - disables Qwen thinking in apply_chat_template (enable_thinking=False) when supported
# - adds an attention_mask to avoid eos-as-pad warnings/odd behavior
# - supports resume (skip persona_id already in output) or overwrite


# === 1. IMPORTS & CONFIG ===

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse
import json
import random
import re

import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# model config
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 256
MAX_RETRIES = 4
SEED = 42
DEVICE_MAP = "auto"


# === 2. UTILS ===

# IO
def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def render_prompt(template: str, **kwargs: str) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", v)
    return out

def load_existing_persona_ids(path: Path) -> set:
    # used to resume: if persona_id already exists in experiments.jsonl => skip it
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get("persona_id")
                if pid:
                    ids.add(str(pid))
            except Exception:
                continue
    return ids

def make_persona_id(base_key: str, style_key: str, topic_key: str, init_idx: int) -> str:
    return f"{base_key}__{style_key}__{topic_key}__init{init_idx}"


# JSON extraction
def safe_extract_json_object(text: str) -> Dict[str, Any]:
    # tries strict json parse; if the model wrapped it in text, extract the first {...} blob
    t = (text or "").strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*?\}", t, flags=re.DOTALL)
    if not m:
        raise ValueError(f"could not find json object in model output:\n{t[:500]}")
    return json.loads(m.group(0))


# leakage controls
def collect_banned_strings(
    personas_cfg: Dict[str, Any],
    base_keys: List[str],
    style_keys: List[str],
    topic_keys: List[str],
) -> List[str]:
    # collects strings that should never appear in system_llm1 / init_user_message
    bans = set()

    leakage = personas_cfg.get("leakage_controls", {}) or {}
    for s in leakage.get("banned_exact_strings", []) or []:
        s = str(s).strip()
        if s:
            bans.add(s)

    # ban ids/keys so model doesn't output them
    for k in base_keys:
        bans.add(str(k))
    for k in style_keys:
        bans.add(str(k))
    for k in topic_keys:
        bans.add(str(k))

    # extra common meta leakage tokens
    for s in ["base_persona_id", "style_id", "persona_id", "profile",
              "roleplay", "system prompt", "assistant:", "user:",
              "experiment", "schema", "metadata"]:
        bans.add(s)

    return sorted(bans)

def contains_banned(text: str, banned: List[str]) -> Optional[str]:
    if not text:
        return None
    low = text.lower()
    for b in banned:
        if b and b.lower() in low:
            return b
    return None


# config parsing
def to_key_text_map(x: Any) -> Dict[str, str]:
    # accepts dict {Name: "text"} or {Name: {text: "..."}}
    # or list [{"id": "...", "text": "..."}]
    if x is None:
        return {}
    if isinstance(x, dict):
        out: Dict[str, str] = {}
        for k, v in x.items():
            if isinstance(v, dict):
                txt = v.get("text", None)
                if txt is None:
                    txt = json.dumps(v, ensure_ascii=False)
                out[str(k)] = str(txt)
            else:
                out[str(k)] = str(v)
        return out
    if isinstance(x, list):
        out = {}
        for item in x:
            if isinstance(item, dict) and "id" in item and "text" in item:
                out[str(item["id"])] = str(item["text"])
        return out
    raise ValueError("unsupported personas.yaml section shape (expected dict or list)")


# === 3. MODEL ===

def load_qwen(model_name: str, device_map: str = "auto") -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    # loads tokenizer + model for local inference
    # trust_remote_code is commonly needed for qwen chat templates
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # pick a reasonable dtype (bfloat16 if supported, else float16)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    # ensure pad token exists for generation
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    return tok, model


def qwen_complete_json(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Dict[str, Any]:
    # feeds the prompt through the qwen chat template if available
    # returns a parsed json object
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        # enable_thinking=False prevents "<think>..." from appearing in the output
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        # attention mask is required because qwen often uses eos as pad
        # and transformers cannot reliably infer the mask in that case
        attention_mask = torch.ones_like(input_ids)
    else:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask

    # move tensors to the model device (works with device_map="auto")
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # decode only the newly generated portion
    gen_ids = output_ids[0, input_ids.shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return safe_extract_json_object(text)


def generate_with_retries_qwen(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    key: str,
    banned: List[str],
    tag: str,
    max_retries: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Optional[str]:
    """
    Generates a JSON object with qwen and extracts the value for `key`.
    Retries up to max_retries if:
        - json parsing fails
        - target key is missing or empty
        - any banned strings are present in the value
    Returns the extracted value if successful, else None.
    """
    for attempt in range(1, max_retries + 1):
        try:
            obj = qwen_complete_json(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            print(f"[parse] {tag} key={key} attempt {attempt}/{max_retries}: {e}")
            continue

        val = (obj.get(key) or "").strip()
        if not val:
            print(f"[empty] {tag} key={key} (attempt {attempt}/{max_retries})")
            continue

        bad = contains_banned(val, banned)
        if bad is None:
            print(f"[ok] {tag} key={key} (attempt {attempt})")
            return val

        print(f"[leak] {tag} key={key} leaked '{bad}' (attempt {attempt}/{max_retries})")

    print(f"[fail] {tag} key={key} failed after {max_retries} attempts")
    return None


def set_seed(seed: int) -> None:
    # ensures repeatability for sampling
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# === 4. MAIN ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--config_dir", type=str, default="config")
    p.add_argument("--personas_yaml", type=str, default="personas.yaml")
    p.add_argument("--prompts_yaml", type=str, default="prompts.yaml")

    p.add_argument("--out_path", type=str, default="data/experiments.jsonl")
    p.add_argument("--overwrite", action="store_true", help="overwrite output file (ignore resume)")

    p.add_argument("--only_base", type=str, default=None, help="generate only for this base_persona key")
    p.add_argument("--only_style", type=str, default=None, help="generate only for this style key")
    p.add_argument("--only_topic", type=str, default=None, help="generate only for this topic key")

    p.add_argument("--num_init_prompts", type=int, default=1, help="init_user_message samples per (profile, topic)")
    p.add_argument("--max_retries", type=int, default=MAX_RETRIES)

    p.add_argument("--model", type=str, default=MODEL_NAME)
    p.add_argument("--device_map", type=str, default=DEVICE_MAP)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--top_p", type=float, default=TOP_P)
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)

    p.add_argument("--seed", type=int, default=SEED)

    p.add_argument("--dry_run", action="store_true",
                   help="Skip model loading; use stub outputs to test pipeline/IO logic.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    config_dir = (base_dir / args.config_dir).resolve()
    out_path = (base_dir / args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and out_path.exists():
        out_path.unlink()

    # load configs
    personas_cfg = read_yaml(config_dir / args.personas_yaml)
    prompts_cfg = read_yaml(config_dir / args.prompts_yaml)

    base_map = to_key_text_map(personas_cfg.get("base_persona"))
    style_map = to_key_text_map(personas_cfg.get("style"))
    topic_map = to_key_text_map(personas_cfg.get("topic"))

    if not base_map:
        raise ValueError(f"no base_persona entries found in {config_dir / args.personas_yaml}")
    if not style_map:
        raise ValueError(f"no style entries found in {config_dir / args.personas_yaml}")
    if not topic_map:
        raise ValueError(f"no topic entries found in {config_dir / args.personas_yaml}")

    if args.only_base is not None and args.only_base not in base_map:
        raise ValueError(f"--only_base={args.only_base!r} not in base_persona keys: {sorted(base_map)}")
    if args.only_style is not None and args.only_style not in style_map:
        raise ValueError(f"--only_style={args.only_style!r} not in style keys: {sorted(style_map)}")
    if args.only_topic is not None and args.only_topic not in topic_map:
        raise ValueError(f"--only_topic={args.only_topic!r} not in topic keys: {sorted(topic_map)}")

    base_items = [(k, base_map[k]) for k in ([args.only_base] if args.only_base else base_map)]
    style_items = [(k, style_map[k]) for k in ([args.only_style] if args.only_style else style_map)]
    topic_items = [(k, topic_map[k]) for k in ([args.only_topic] if args.only_topic else topic_map)]

    tmpl_sys = prompts_cfg["generation_prompt_system_llm1"]["prompt"]
    tmpl_init = prompts_cfg["generation_prompt_init_user_prompt"]["prompt"]
    system_llm2 = prompts_cfg["system_llm2"]["prompt"]

    banned = collect_banned_strings(personas_cfg, list(base_map), list(style_map), list(topic_map))

    existing_ids = load_existing_persona_ids(out_path)

    total_profiles = len(base_items) * len(style_items)
    total_rows = total_profiles * len(topic_items) * args.num_init_prompts

    print(f"[load] personas={config_dir / args.personas_yaml}")
    print(f"[load] prompts={config_dir / args.prompts_yaml}")
    print(f"[out] {out_path} (existing rows: {len(existing_ids)})")
    print(f"[config] base={len(base_items)} style={len(style_items)} "
          f"topics={len(topic_items)} num_init_prompts={args.num_init_prompts} "
          f"=> {total_rows} total rows")
    if args.dry_run:
        print("[dry_run] skipping model loading — stub outputs will be used", flush=True)
        tok = model = None
    else:
        print(f"[model] loading {args.model} device_map={args.device_map}")
        tok, model = load_qwen(args.model, device_map=args.device_map)

    def _gen(prompt: str, key: str, tag: str) -> Optional[str]:
        if args.dry_run:
            return f"[DRY_RUN {key} for {tag}]"
        return generate_with_retries_qwen(
            tokenizer=tok,
            model=model,
            prompt=prompt,
            key=key,
            banned=banned,
            tag=tag,
            max_retries=args.max_retries,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

    # cache system_llm1 once per (base_key, style_key)
    sys_cache: Dict[str, str] = {}

    written = 0
    prof_i = 0

    with out_path.open("a", encoding="utf-8") as f_out:
        for base_key, base_text in base_items:
            for style_key, style_text in style_items:
                prof_i += 1
                cache_key = f"{base_key}__{style_key}"
                print(f"\n[{prof_i}/{total_profiles}] profile={cache_key}")

                base_json = json_dumps_compact({"text": base_text})
                style_json = json_dumps_compact({"text": style_text})

                # generate system_llm1 once per (base, style)
                if cache_key not in sys_cache:
                    print(f"[cache miss] generating system_llm1 for {cache_key}")
                    prompt_sys = render_prompt(
                        tmpl_sys,
                        BASE_PERSONA_JSON=base_json,
                        STYLE_JSON=style_json,
                    )
                    sys_llm1 = _gen(prompt_sys, "system_llm1", f"{cache_key}|system_llm1")
                    if sys_llm1 is None:
                        print(f"[skip] system_llm1 failed for {cache_key}")
                        continue
                    sys_cache[cache_key] = sys_llm1
                else:
                    sys_llm1 = sys_cache[cache_key]
                    print(f"[cache hit] reusing system_llm1 for {cache_key}")

                # loop topics x N init prompts
                for topic_key, topic_text in topic_items:
                    topic_json = json_dumps_compact({"text": topic_text})

                    for init_idx in range(args.num_init_prompts):
                        persona_id = make_persona_id(base_key, style_key, topic_key, init_idx)

                        if persona_id in existing_ids:
                            print(f"[skip] {persona_id} already exists")
                            continue

                        prompt_init = render_prompt(
                            tmpl_init,
                            BASE_PERSONA_JSON=base_json,
                            STYLE_JSON=style_json,
                            TOPIC_JSON=topic_json,
                        )
                        init_user = _gen(prompt_init, "init_user_message", f"{persona_id}|init_user_message")
                        if init_user is None:
                            print(f"[skip] init_user_message failed for {persona_id}")
                            continue

                        row = {
                            "persona_id": persona_id,
                            "system_llm1": sys_llm1,
                            "system_llm2": system_llm2,
                            "init_user_message": init_user,
                            "profile": {
                                "base_persona_id": base_key,
                                "style_id": style_key,
                                "topic_id": topic_key,
                                "init_idx": init_idx,
                                "base_persona": {"text": base_text},
                                "style": {"text": style_text},
                                "topic": {"text": topic_text},
                            },
                        }

                        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f_out.flush()

                        existing_ids.add(persona_id)
                        written += 1
                        print(f"[write] wrote row #{written}: {persona_id}")

    print(f"\n[done] wrote {written} new rows -> {out_path}")


if __name__ == "__main__":
    main()
