# src/generate_experiments.py
#
# This script generates experiments.jsonl by:
# 1) Reading hidden persona + style specs from yaml
# 2) Using QWEN locally (via transformers) to generate:
#    - system_llm1 (persona-conditioned system prompt)
#    - init_user_message (persona-conditioned first user turn)
#    with leakage checks + retries
# 3) Combining those with llm2 system prompts (investigator modes) and writing jsonl
#
# Notes:
# - it disables qwen "thinking" in the chat template to keep outputs parseable json
# - it supplies attention_mask to avoid pad_token==eos_token warnings/odd behavior


# === 1. IMPORTS & CONFIG ===

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re
import random
import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# config
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
ATTRIBUTES_YAML = CONFIG_DIR / "hidden_persona_attributes.yaml"
PROMPTS_YAML = CONFIG_DIR / "prompts.yaml"
OUT_FILE = BASE_DIR / "experiments.jsonl"

# BASE_PERSONA_ID = "bp_tech_starter"
# if None, use ALL base personas found in hidden_persona_attributes.yaml
BASE_PERSONA_IDS = None  
INVESTIGATOR_MODES = ["none", "guided", "unguided"]
LIMIT_STYLES = None  # 2 for debugging

# model config
MODEL_NAME = "Qwen/Qwen3-8B"
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 256
MAX_RETRIES = 4
SEED = 42
DEVICE_MAP = "auto"  # or "cpu" for cpu-only


# === 2. UTILS ===

# IO utils
def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_extract_json_object(text: str) -> Dict[str, Any]:
    # tries strict json parse; if the model wrapped it in text, extract the first {...} blob
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    # non greedy regex
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"could not find json object in model output:\n{text[:500]}")
    return json.loads(m.group(0))

def json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def render_prompt(template: str, **kwargs: str) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", v)
    return out


def load_existing_jsonl_ids(path: Path) -> set:
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
                    ids.add(pid)
            except Exception:
                continue
    return ids

def make_persona_id(base_persona_id: str, style_id: str, investigator_mode: str) -> str:
    return f"{base_persona_id}__{style_id}__inv_{investigator_mode}"


# LLM2 system prompt builder
def build_llm2_system_prompt(
    prompts_cfg: Dict[str, Any],
    investigator_mode: str,
    style_ids: List[str],
    style_names: List[str],
    ) -> str:
    """
    Builds the system prompt for LLM2 based on investigator mode.
    """
    if investigator_mode == "none":
        return prompts_cfg["system_llm2_base"]["prompt"]

    if investigator_mode == "guided":
        tmpl = prompts_cfg["system_llm2_investigator_guided"]["prompt"]
        return render_prompt(
            tmpl,
            STYLE_ID_LIST=", ".join(style_ids),
            STYLE_NAME_LIST=", ".join(style_names),
        )

    if investigator_mode == "unguided":
        return prompts_cfg["system_llm2_investigator_unguided"]["prompt"]

    raise ValueError(f"unknown investigator_mode: {investigator_mode}")


# leakage controls
def collect_dynamic_bans(attrs_cfg: Dict[str, Any]) -> List[str]:
    # collects strings that should never appear in system_llm1 / init_user_message
    bans = set()

    leakage = attrs_cfg.get("leakage_controls", {}) or {}
    for s in leakage.get("banned_exact_strings", []) or []:
        bans.add(str(s))

    base_personas = attrs_cfg["profiles"]["base_persona_id"]
    styles = attrs_cfg["profiles"]["style_id"]

    # ban ids like bp_* and st_*
    bans.update(base_personas.keys())
    bans.update(styles.keys())

    return sorted({b.strip() for b in bans if b and str(b).strip()})

def contains_banned(text: str, banned: List[str]) -> Optional[str]:
    if not text:
        return None
    lower = text.lower()
    for b in banned:
        if b.lower() in lower:
            return b
    return None


# === 3. MODEL ===

def set_seed(seed: int) -> None:
    # ensures repeatability for sampling
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        # enable_thinking=false prevents "<think>..." from appearing in the output
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
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
    persona_id: str,
    label: str,
    max_retries: int = MAX_RETRIES,
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
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        except Exception as e:
            print(f"[parse] {persona_id} {label} parse failed (attempt {attempt}/{max_retries}): {e}")
            continue

        candidate = (obj.get(key) or "").strip()
        if not candidate:
            print(f"[empty] {persona_id} {label} empty value (attempt {attempt}/{max_retries}) -> retry")
            continue

        bad = contains_banned(candidate, banned)
        if bad is None:
            print(f"[ok] {persona_id} {label} (attempt {attempt})")
            return candidate

        print(f"[leak] {persona_id} {label} leaked '{bad}' (attempt {attempt}/{max_retries}) -> retry")

    print(f"[fail] {persona_id} {label} failed after {max_retries} attempts -> skip")
    return None


# === 4. MAIN EXECUTION ===

def main():
    # load configs
    print(f"[load] attributes from {ATTRIBUTES_YAML}")
    attrs_cfg = read_yaml(ATTRIBUTES_YAML)

    print(f"[load] prompts from {PROMPTS_YAML}")
    prompts_cfg = read_yaml(PROMPTS_YAML)

    base_personas = attrs_cfg["profiles"]["base_persona_id"]
    styles = attrs_cfg["profiles"]["style_id"]

    # choose which base personas to use
    if BASE_PERSONA_IDS is None:
        base_persona_items = list(base_personas.items())   # ALL
    else:
        missing = [bid for bid in BASE_PERSONA_IDS if bid not in base_personas]
        if missing:
            raise ValueError(f"BASE_PERSONA_IDS contains unknown ids: {missing}")
        base_persona_items = [(bid, base_personas[bid]) for bid in BASE_PERSONA_IDS]

    style_items = list(styles.items())
    effective_limit = LIMIT_STYLES if LIMIT_STYLES is not None else LIMIT_STYLES
    if effective_limit is not None:
        style_items = style_items[:effective_limit]

    style_ids = [sid for sid, _ in style_items]
    style_names = [sobj.get("name", sid) for sid, sobj in style_items]

    print(f"[config] base_personas={len(base_persona_items)} styles={len(style_items)} investigator_modes={INVESTIGATOR_MODES}")

    # leakage controls 
    banned_strings = collect_dynamic_bans(attrs_cfg)
    print(f"[leakage] banned strings total: {len(banned_strings)}")

    # resume support 
    existing_ids = load_existing_jsonl_ids(OUT_FILE)
    print(f"[out] writing to {OUT_FILE} (existing rows: {len(existing_ids)})")

    # prompt templates 
    tmpl_sys_llm1 = prompts_cfg["generation_prompt_system_llm1"]["prompt"]
    tmpl_init_user = prompts_cfg["generation_prompt_init_user_prompt"]["prompt"]

    # load qwen locally 
    print(f"[model] loading {MODEL_NAME} (device_map={DEVICE_MAP})")
    tokenizer, model = load_qwen(MODEL_NAME, device_map=DEVICE_MAP)

    # cache llm1 outputs once per (base_persona_id, style_id) 
    llm1_cache: Dict[str, Dict[str, str]] = {}

    written = 0
    total = len(base_persona_items) * len(style_items) * len(INVESTIGATOR_MODES)
    idx = 0

    with OUT_FILE.open("a", encoding="utf-8") as f_out:
        for base_persona_id, base_persona in base_persona_items:
            for style_id, style_obj in style_items:
                cache_key = f"{base_persona_id}__{style_id}"

                base_persona_json = json_dumps_compact(base_persona)
                style_json = json_dumps_compact(style_obj)

                # generate llm1 prompts once per style (cache miss)
                if cache_key not in llm1_cache:
                    print(f"\n[cache miss] generating llm1 prompts for {cache_key}")

                    prompt_sys = render_prompt(
                        tmpl_sys_llm1,
                        BASE_PERSONA_JSON=base_persona_json,
                        STYLE_JSON=style_json,
                    )
                    system_llm1 = generate_with_retries_qwen(
                        tokenizer=tokenizer,
                        model=model,
                        prompt=prompt_sys,
                        key="system_llm1",
                        banned=banned_strings,
                        persona_id=cache_key,
                        label="system_llm1",
                        max_retries=MAX_RETRIES,
                    )
                    if system_llm1 is None:
                        print(f"[skip] {cache_key} system_llm1 failed leakage checks")
                        continue

                    prompt_user = render_prompt(
                        tmpl_init_user,
                        BASE_PERSONA_JSON=base_persona_json,
                        STYLE_JSON=style_json,
                    )
                    init_user_message = generate_with_retries_qwen(
                        tokenizer=tokenizer,
                        model=model,
                        prompt=prompt_user,
                        key="init_user_message",
                        banned=banned_strings,
                        persona_id=cache_key,
                        label="init_user_message",
                        max_retries=MAX_RETRIES,
                    )
                    if init_user_message is None:
                        print(f"[skip] {cache_key} init_user_message failed leakage checks")
                        continue

                    llm1_cache[cache_key] = {
                        "system_llm1": system_llm1,
                        "init_user_message": init_user_message,
                    }
                else:
                    print(f"[cache hit] reusing llm1 prompts for {cache_key}")

                system_llm1 = llm1_cache[cache_key]["system_llm1"]
                init_user_message = llm1_cache[cache_key]["init_user_message"]

                # now only vary llm2 investigator mode
                for inv_mode in INVESTIGATOR_MODES:
                    idx += 1
                    persona_id = make_persona_id(base_persona_id, style_id, inv_mode)

                    if persona_id in existing_ids:
                        print(f"[{idx}/{total}] [skip] {persona_id} already exists")
                        continue

                    print(f"[{idx}/{total}] [gen] persona_id={persona_id}")

                    system_llm2 = build_llm2_system_prompt(
                        prompts_cfg=prompts_cfg,
                        investigator_mode=inv_mode,
                        style_ids=style_ids,
                        style_names=style_names,
                    )

                    row = {
                        "persona_id": persona_id,
                        "profile": {
                            "base_persona_id": base_persona_id,
                            "style_id": style_id,
                            "investigator_mode": inv_mode,
                            "base_persona": base_persona,
                            "style": style_obj,
                        },
                        "system_llm1": system_llm1,
                        "system_llm2": system_llm2,
                        "init_user_message": init_user_message,
                    }

                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f_out.flush()

                    existing_ids.add(persona_id)
                    written += 1
                    print(f"[write] wrote row #{written}: {persona_id}")

    print(f"\n[done] wrote {written} new rows to {OUT_FILE}")


if __name__ == "__main__":
    main()