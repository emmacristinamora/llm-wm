# src/generate_experiments.py

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import json
import os
import re
import random

import yaml

# qwen local inference (huggingface)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# === 1) config ===

BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
ATTRIBUTES_YAML = CONFIG_DIR / "hidden_persona_attributes.yaml"
PROMPTS_YAML = CONFIG_DIR / "prompts.yaml"
OUT_FILE = BASE_DIR / "experiments.jsonl"

BASE_PERSONA_ID = "bp_tech_starter"
INVESTIGATOR_MODES = ["none", "guided", "unguided"]
LIMIT_STYLES = None  # set to 2 for debugging

# qwen model config
# if your colleague used "Qwen/Qwen3-8B", keep it identical for prompt generation.
MODEL_NAME = "Qwen/Qwen3-8B"

# generation params
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 256
MAX_RETRIES = 4
SEED = 42


# === 2) utils ===

def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_extract_json_object(text: str) -> Dict[str, Any]:
    # tries strict json parse; if model wrapped it in text, extract the first {...} blob
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
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


def build_llm2_system_prompt(
    prompts_cfg: Dict[str, Any],
    investigator_mode: str,
    style_ids: List[str],
    style_names: List[str],
) -> str:
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

    # ban name fields too (these can leak)
    for obj in base_personas.values():
        if isinstance(obj, dict) and obj.get("name"):
            bans.add(str(obj["name"]))
    for obj in styles.values():
        if isinstance(obj, dict) and obj.get("name"):
            bans.add(str(obj["name"]))

    return sorted({b.strip() for b in bans if b and str(b).strip()})


def contains_banned(text: str, banned: List[str]) -> Optional[str]:
    if not text:
        return None
    lower = text.lower()
    for b in banned:
        if b.lower() in lower:
            return b
    return None


# === 3) qwen local inference ===

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_qwen(model_name: str, device_map: str = "auto") -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    # loads tokenizer + model for local inference
    # trust_remote_code is often required for qwen chat templates
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
    # feeds the prompt through the qwen chat template if available,
    # and returns a parsed json object.
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # move tensors to the model device (works with device_map="auto")
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
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
    for attempt in range(1, max_retries + 1):
        obj = qwen_complete_json(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        candidate = (obj.get(key) or "").strip()

        bad = contains_banned(candidate, banned)
        if bad is None:
            print(f"[ok] {persona_id} {label} (attempt {attempt})")
            return candidate

        print(f"[leak] {persona_id} {label} leaked '{bad}' (attempt {attempt}/{max_retries}) -> retry")

    print(f"[fail] {persona_id} {label} failed after {max_retries} attempts -> skip")
    return None


# === 4) main generation ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--out_file", type=str, default=str(OUT_FILE))
    parser.add_argument("--limit_styles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device_map", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)

    out_file = Path(args.out_file)

    # --- load configs ---
    print(f"[load] attributes from {ATTRIBUTES_YAML}")
    attrs_cfg = read_yaml(ATTRIBUTES_YAML)

    print(f"[load] prompts from {PROMPTS_YAML}")
    prompts_cfg = read_yaml(PROMPTS_YAML)

    base_personas = attrs_cfg["profiles"]["base_persona_id"]
    styles = attrs_cfg["profiles"]["style_id"]

    if BASE_PERSONA_ID not in base_personas:
        raise ValueError(f"BASE_PERSONA_ID={BASE_PERSONA_ID} not found in yaml")

    base_persona = base_personas[BASE_PERSONA_ID]

    style_items = list(styles.items())
    effective_limit = args.limit_styles if args.limit_styles is not None else LIMIT_STYLES
    if effective_limit is not None:
        style_items = style_items[:effective_limit]

    style_ids = [sid for sid, _ in style_items]
    style_names = [sobj.get("name", sid) for sid, sobj in style_items]

    print(f"[config] BASE_PERSONA_ID={BASE_PERSONA_ID}")
    print(f"[config] styles={len(style_items)} investigator_modes={INVESTIGATOR_MODES}")

    # --- leakage controls ---
    banned_strings = collect_dynamic_bans(attrs_cfg)
    print(f"[leakage] banned strings total: {len(banned_strings)}")

    # --- resume support ---
    existing_ids = load_existing_jsonl_ids(out_file)
    print(f"[out] writing to {out_file} (existing rows: {len(existing_ids)})")

    # --- prompt templates ---
    tmpl_sys_llm1 = prompts_cfg["generation_prompt_system_llm1"]["prompt"]
    tmpl_init_user = prompts_cfg["generation_prompt_init_user_prompt"]["prompt"]

    # --- load qwen locally ---
    print(f"[model] loading {args.model_name} (device_map={args.device_map})")
    tokenizer, model = load_qwen(args.model_name, device_map=args.device_map)

    # --- cache llm1 outputs once per (base_persona_id, style_id) ---
    llm1_cache: Dict[str, Dict[str, str]] = {}

    written = 0
    total = len(style_items) * len(INVESTIGATOR_MODES)
    idx = 0

    with out_file.open("a", encoding="utf-8") as f_out:
        for style_id, style_obj in style_items:
            cache_key = f"{BASE_PERSONA_ID}__{style_id}"

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
                persona_id = make_persona_id(BASE_PERSONA_ID, style_id, inv_mode)

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
                        "base_persona_id": BASE_PERSONA_ID,
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

    print(f"\n[done] wrote {written} new rows to {out_file}")


if __name__ == "__main__":
    main()