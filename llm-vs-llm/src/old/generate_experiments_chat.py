# src/generate_experiments.py


# === 1. IMPORTS & CONFIG ===

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
import json
import os
import re
from dataclasses import dataclass
import yaml
from openai import OpenAI

# config
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
ATTRIBUTES_YAML = CONFIG_DIR / "hidden_persona_attributes.yaml"
PROMPTS_YAML = CONFIG_DIR / "prompts.yaml"
OUT_FILE = BASE_DIR / "experiments.jsonl"

BASE_PERSONA_ID = "bp_tech_starter"          # start cheap: one base persona
INVESTIGATOR_MODES = ["none", "guided", "unguided"]

OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7
MAX_TOKENS = 256
LIMIT_STYLES = None  # 2 for debugging


# === 2. UTILS ===

def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Could not find JSON object in model output:\n{text[:500]}")
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
                if "persona_id" in obj:
                    ids.add(obj["persona_id"])
            except Exception:
                continue
    return ids

def make_persona_id(base_persona_id: str, style_id: str, investigator_mode: str) -> str:
    return f"{base_persona_id}__{style_id}__inv_{investigator_mode}"

def build_llm2_system_prompt(
        prompts_cfg: Dict[str, Any], 
        investigator_mode: str,
        style_ids: List[str], 
        style_names: List[str]
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
    raise ValueError(f"Unknown investigator_mode: {investigator_mode}")

def openai_complete_json(
        client: OpenAI, 
        prompt: str
        ) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    text = resp.choices[0].message.content or ""
    return safe_extract_json_object(text)

def collect_dynamic_bans(attrs_cfg: Dict[str, Any], allow_topic_anchors: bool = True) -> List[str]:
    bans = set()

    leakage = attrs_cfg.get("leakage_controls", {}) or {}
    for s in leakage.get("banned_exact_strings", []) or []:
        bans.add(s)

    base_personas = attrs_cfg["profiles"]["base_persona_id"]
    styles = attrs_cfg["profiles"]["style_id"]

    bans.update(base_personas.keys())
    bans.update(styles.keys())

    for obj in base_personas.values():
        if isinstance(obj, dict) and obj.get("name"):
            bans.add(obj["name"])

    return sorted({b.strip() for b in bans if b and b.strip()})

def contains_banned(text: str, banned: List[str]) -> Optional[str]:
    if not text:
        return None
    lower = text.lower()
    for b in banned:
        if b.lower() in lower:
            return b
    return None


def generate_with_retries(
    client: OpenAI,
    prompt: str,
    key: str,
    banned: List[str],
    persona_id: str,
    label: str,
    max_retries: int = 4,
) -> Optional[str]:
    for attempt in range(1, max_retries + 1):
        obj = openai_complete_json(client, prompt)
        candidate = (obj.get(key) or "").strip()

        bad = contains_banned(candidate, banned)
        if bad is None:
            print(f"[OK] {persona_id} {label} (attempt {attempt})")
            return candidate

        print(f"[LEAK] {persona_id} {label} leaked '{bad}' (attempt {attempt}/{max_retries}) -> retry")

    print(f"[FAIL] {persona_id} {label} failed after {max_retries} attempts -> skip")
    return None


# === 3. EXPERIMENT GENERATION ===

def main():
    # --- load configs ---
    print(f"[Load] attributes from {ATTRIBUTES_YAML}")
    attrs_cfg = read_yaml(ATTRIBUTES_YAML)

    print(f"[Load] prompts from {PROMPTS_YAML}")
    prompts_cfg = read_yaml(PROMPTS_YAML)

    base_personas = attrs_cfg["profiles"]["base_persona_id"]
    styles = attrs_cfg["profiles"]["style_id"]

    if BASE_PERSONA_ID not in base_personas:
        raise ValueError(f"BASE_PERSONA_ID={BASE_PERSONA_ID} not found in YAML")

    base_persona = base_personas[BASE_PERSONA_ID]

    style_items = list(styles.items())
    if LIMIT_STYLES is not None:
        style_items = style_items[:LIMIT_STYLES]

    style_ids = [sid for sid, _ in style_items]
    style_names = [sobj.get("name", sid) for sid, sobj in style_items]

    print(f"[Config] BASE_PERSONA_ID={BASE_PERSONA_ID}")
    print(f"[Config] styles={len(style_items)} investigator_modes={INVESTIGATOR_MODES}")

    # --- leakage controls ---
    banned_strings = collect_dynamic_bans(attrs_cfg, allow_topic_anchors=True)
    print(f"[Leakage] banned strings total: {len(banned_strings)}")

    # --- output file / resume ---
    existing_ids = load_existing_jsonl_ids(OUT_FILE)
    print(f"[Out] writing to {OUT_FILE} (existing rows: {len(existing_ids)})")

    # --- prompts ---
    tmpl_sys_llm1 = prompts_cfg["generation_prompt_system_llm1"]["prompt"]
    tmpl_init_user = prompts_cfg["generation_prompt_init_user_prompt"]["prompt"]

    # --- client ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running.")
    print("[Env] OPENAI_API_KEY found")

    client = OpenAI(api_key=api_key)

    # --- cache: generate LLM1 prompts once per (base_persona_id, style_id) ---
    llm1_cache: Dict[str, Dict[str, str]] = {}

    written = 0
    total = len(style_items) * len(INVESTIGATOR_MODES)
    idx = 0

    with OUT_FILE.open("a", encoding="utf-8") as f_out:
        for style_id, style_obj in style_items:
            cache_key = f"{BASE_PERSONA_ID}__{style_id}"
            base_persona_json = json_dumps_compact(base_persona)
            style_json = json_dumps_compact(style_obj)

            # Generate system_llm1 + init_user_message ONCE per style
            if cache_key not in llm1_cache:
                print(f"\n[Cache Miss] generating LLM1 prompts for {cache_key}")

                # --- generate system_llm1 (leakage-safe) ---
                prompt_sys = render_prompt(
                    tmpl_sys_llm1,
                    BASE_PERSONA_JSON=base_persona_json,
                    STYLE_JSON=style_json,
                )
                system_llm1 = generate_with_retries(
                    client=client,
                    prompt=prompt_sys,
                    key="system_llm1",
                    banned=banned_strings,
                    persona_id=cache_key,
                    label="system_llm1",
                    max_retries=4,
                )
                if system_llm1 is None:
                    print(f"[Skip] {cache_key} system_llm1 failed leakage checks")
                    continue

                # --- generate init_user_message (leakage-safe) ---
                prompt_user = render_prompt(
                    tmpl_init_user,
                    BASE_PERSONA_JSON=base_persona_json,
                    STYLE_JSON=style_json,
                )
                init_user_message = generate_with_retries(
                    client=client,
                    prompt=prompt_user,
                    key="init_user_message",
                    banned=banned_strings,
                    persona_id=cache_key,
                    label="init_user_message",
                    max_retries=4,
                )
                if init_user_message is None:
                    print(f"[Skip] {cache_key} init_user_message failed leakage checks")
                    continue

                llm1_cache[cache_key] = {
                    "system_llm1": system_llm1,
                    "init_user_message": init_user_message,
                }
            else:
                print(f"[Cache Hit] reusing LLM1 prompts for {cache_key}")

            # Reuse cached LLM1 prompts across investigator modes
            system_llm1 = llm1_cache[cache_key]["system_llm1"]
            init_user_message = llm1_cache[cache_key]["init_user_message"]

            # Now only vary LLM2 investigator mode
            for inv_mode in INVESTIGATOR_MODES:
                idx += 1
                persona_id = make_persona_id(BASE_PERSONA_ID, style_id, inv_mode)

                if persona_id in existing_ids:
                    print(f"[{idx}/{total}] [Skip] {persona_id} already exists")
                    continue

                print(f"[{idx}/{total}] [Gen] persona_id={persona_id}")

                # system prompt for LLM2 (varies by investigator mode)
                system_llm2 = build_llm2_system_prompt(
                    prompts_cfg=prompts_cfg,
                    investigator_mode=inv_mode,
                    style_ids=style_ids,
                    style_names=style_names,
                )

                # --- save row ---
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
                print(f"[Write] wrote row #{written}: {persona_id}")

    print(f"\n[Done] Wrote {written} new rows to {OUT_FILE}")

if __name__ == "__main__":
    main()