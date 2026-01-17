# src/style_scheduler.py

# === 1. IMPORTS ===

from __future__ import annotations
import random
from typing import List, Optional, Any, Dict


# === 2. SCHEDULED RANDOMNESS ===

def _rng_for_schedule(base_seed: int, seed_offset: int=0) -> random.Random:
    """
    Create a random number generator for scheduled randomness based on a base seed and an optional offset.
    Args:
        base_seed (int): The base seed for the RNG.
        seed_offset (int): An optional offset to modify the seed for different schedules.
    Returns:
        random.Random: A random number generator instance.
    """
    combined_seed = base_seed + seed_offset
    rng = random.Random(combined_seed)
    return rng


# === 3. REGIME FOR N ===

def choose_style_id(schedule_cfg: Dict[str, Any], round_idx: int, base_seed: int) -> str:
    """
    Decide which style id is active at a given round index.
    Args:
        schedule_cfg (Dict[str, Any]): Configuration dictionary for the style schedule.
            Supported schedule types from YAML:
                - fixed:
                    {type: fixed, style_id: "formal"}
                - every_n:
                    {type: every_n, n: 4, style_ids: [...], mode: "random", random_seed_offset: 1000}
                - random_change:
                    {type: random_change, p_change: 0.2, style_ids: [...], random_seed_offset: 2000}
        round_idx (int): The current round index.
        base_seed (int): The base seed for randomness.
    Returns:
        str: The chosen style id (formal, terse, enthusiastic etc).
    """
    schedule_type = schedule_cfg["type"]

    # FIXED STYLE
    if schedule_type == "fixed":
        return schedule_cfg["style_id"]
    
    # STYLE CHANGES EVERY N ROUNDS (the new style is chosen randomly)
    if schedule_type == "every_n":
        # load config values
        n = int(schedule_cfg["n"])
        style_ids: List[str] = list(schedule_cfg["style_ids"])
        mode = schedule_cfg.get("mode", "random")
        seed_offset = int(schedule_cfg.get("random_seed_offset", 0))

        # safety checks
        if n <= 0:
            raise ValueError(f"!! every_n requires n>0, got n={n}")
        if not style_ids:
            raise ValueError("!! every_n requires non-empty style_ids")
        if mode != "random":
            raise ValueError(f"!! every_n only supports mode='random', got mode='{mode}'")
        
        # define which chunk of size n we are in
        block_idx = round_idx // n
        rng = _rng_for_schedule(base_seed, seed_offset)

        # deterministically choose style for this block
        chosen = None
        for _ in range(block_idx + 1):
            chosen = rng.choice(style_ids)
        
        assert chosen is not None
        return chosen
    
    # STYLE CHANGES RANDOMLY WITH PROBABILITY P EACH ROUND (the new style is also chosen randomly)
    if schedule_type == "random_change":
        # load config values
        p_change = float(schedule_cfg["p_change"])
        style_ids: List[str] = list(schedule_cfg["style_ids"])
        seed_offset = int(schedule_cfg.get("random_seed_offset", 0))

        # safety checks
        if not (0.0 <= p_change <= 1.0):
            raise ValueError(f"!! random_change requires 0.0 <= p_change <= 1.0, got p_change={p_change}")
        if not style_ids:
            raise ValueError("!! random_change requires non-empty style_ids")
        
        rng = _rng_for_schedule(base_seed, seed_offset)
        current = rng.choice(style_ids)

        # init style at round 0
        if round_idx == 0:
            return current
        
        # for subsequent rounds, decide whether to change style (simulations kinda)
        for r in range(1, round_idx + 1):
            if rng.random() < p_change:
                current = rng.choice(style_ids)
        
        return current
    
    # for other cases
    raise ValueError(f"!! Unsupported style schedule type '{schedule_type}'")
        

# === 4. IMPLEMENT HIDDEN STYLE CHOICE ===

def get_hidden_style_prompt_for_turn(
        experiment_cfg: Dict[str, Any],
        styles_cfg: Dict[str, Any],
        speaker: str,
        round_idx: int,
        base_seed: int,
    ) -> Optional[str]:
    """
    Apply the style schedule only if the cuurent speaker is the hidden_speaker.
    Args:
        experiment_cfg (Dict[str, Any]): Experiment configuration dictionary.
        styles_cfg (Dict[str, Any]): Styles configuration dictionary.
        speaker (str): The role of the current speaker (A/B).
        round_idx (int): The current round index.
        base_seed (int): The base seed for randomness.
    Returns:
        Optional[str]: The hidden style prompt if applicable, else None.
    """
    # load config values
    hidden_speaker = experiment_cfg.get("hidden_speaker", None)

    # safety checks
    if hidden_speaker is None:
        return None
    if speaker != hidden_speaker:
        return None
        
    # determine style regime for this round
    schedule_cfg = experiment_cfg["schedule"]
    style_id = choose_style_id(schedule_cfg, round_idx, base_seed)

    # safety check
    if style_id not in styles_cfg:
        raise ValueError(f"!! Chosen style_id '{style_id}' not found in styles configuration.")
    
    # return the style prompt for the chosen style
    return styles_cfg[style_id]["prompt"]

def get_hidden_style_id_for_round(
    experiment_cfg: Dict[str, Any],
    speaker: str,
    round_idx: int,
    base_seed: int,
    ) -> Optional[str]:
    """
    Same as get_hidden_style_prompt_for_turn, but returns the style_id instead of the prompt text.
    Useful for logging into transcript.jsonl so we mark change points in analysis.
    """
    hidden_speaker = experiment_cfg.get("hidden_speaker", None)
    if hidden_speaker is None or speaker != hidden_speaker:
        return None
    return choose_style_id(experiment_cfg["schedule"], round_idx, base_seed)


# === 5. IMPLEMENT INVESTIGATOR MODE ===

def _render_investigator_prompt(
        investigator_cfg: Dict[str, Any],
        style_ids: List[str],
    ) -> str:
    """
    Fill {style_id_list} placeholder from YAML.
    Args:
        investigator_cfg (Dict[str, Any]): Investigator style configuration dictionary.
    Returns:
        str: The processed list of style ids from which the investigator has to guess.
    """
    # load investigator prompt template
    template = investigator_cfg["prompt"]

    # build style_id_list string
    style_id_list = ", ".join(style_ids)

    # return processed investigator prompt
    return template.replace("{style_id_list}", style_id_list)

def get_investigator_prompt_for_turn(
        experiment_cfg: Dict[str, Any],
        investigator_cfg: Dict[str, Any],
        styles_cfg: Dict[str, Any],
        speaker: str,
    ) -> Optional[str]:
    """
    If the investigator_mode is on, return the investigator instruction for the investigator speaker.
    Args:
        experiment_cfg (Dict[str, Any]): Experiment configuration dictionary.
        investigator_cfg (Dict[str, Any]): Investigator style configuration dictionary.
        styles_cfg (Dict[str, Any]): Styles configuration dictionary.
        speaker (str): The role of the current speaker (A/B).
    Returns:
        Optional[str]: The investigator style prompt if applicable, else None.
    """
    # load config values
    investigator_mode = bool(experiment_cfg.get("investigator_mode", False))
    investigator_speaker = experiment_cfg.get(
        "investigator_speaker",
        investigator_cfg.get("investigator_speaker_default", "B"),
    )
    all_style_ids = list(styles_cfg.keys())

    # safety checks
    if not investigator_mode:
        return None
    if speaker != investigator_speaker:
        return None

    # return the processed investigator prompt
    return _render_investigator_prompt(investigator_cfg, all_style_ids)

