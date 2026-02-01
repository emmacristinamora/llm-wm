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
    

# === 3. EXPERIMENT GENERATION ===

def main():

    # load configs
    attrs_cfg = read_yaml(ATTRIBUTES_YAML)
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

    existing_ids = load_existing_jsonl_ids(OUT_FILE)

    tmpl_sys_llm1 = prompts_cfg["generation_prompt_system_llm1"]["prompt"]
    tmpl_init_user = prompts_cfg["generation_prompt_init_user_prompt"]["prompt"]

    client = OpenAI()  # OPENAI_API_KEY from env

    written = 0
    with OUT_FILE.open("a", encoding="utf-8") as f_out:
        for style_id, style_obj in style_items:
            for inv_mode in INVESTIGATOR_MODES:
                persona_id = make_persona_id(BASE_PERSONA_ID, style_id, inv_mode)
                if persona_id in existing_ids:
                    continue

                system_llm2 = build_llm2_system_prompt(
                    prompts_cfg, 
                    inv_mode, 
                    style_ids, 
                    style_names
                    )

                base_persona_json = json_dumps_compact(base_persona)
                style_json = json_dumps_compact(style_obj)

                prompt_sys = render_prompt(
                    tmpl_sys_llm1,
                    BASE_PERSONA_JSON=base_persona_json,
                    STYLE_JSON=style_json,
                    )
                sys_obj = openai_complete_json(client, prompt_sys)
                system_llm1 = sys_obj["system_llm1"]

                prompt_user = render_prompt(
                    tmpl_init_user,
                    BASE_PERSONA_JSON=base_persona_json,
                    STYLE_JSON=style_json,
                    )
                user_obj = openai_complete_json(client, prompt_user)
                init_user_message = user_obj["init_user_message"]

                # save in JSON as a row
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

    print(f"Wrote {written} rows to {OUT_FILE}")

if __name__ == "__main__":
    main()