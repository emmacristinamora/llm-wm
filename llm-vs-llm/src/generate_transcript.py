# src/generate_transcript.py

# === 1. IMPORTS ===

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import time
import torch

# internal
from .io_utils import build_run_paths, init_run_dir, append_jsonl
from .model_loader import load_tokenizer_and_model, set_global_seed
from .style_scheduler import(
    get_hidden_style_prompt_for_turn,
    get_investigator_prompt_for_turn,
    get_hidden_style_id_for_round,
)
from .prompts import build_next_prompt_plaintext, build_chat_messages


# === 2. UTILS ===

def speaker_to_role(speaker: str) -> str:
    """
    Map speaker labels ("A", "B") to roles used in prompts ("user", "assistant").
    Args:
        speaker (str): The speaker label ("A" or "B").
    Returns:
        str: The corresponding role ("user" or "assistant").
    """
    if speaker == "A":
        return "user"
    elif speaker == "B":
        return "assistant"
    else:
        raise ValueError(f"!! Unsupported speaker '{speaker}'. Supported speakers are: 'A', 'B'.")
    
def join_extra_instructions(*parts: Optional[str]) -> Optional[str]:
    """
    Merge multiple optional instruction blocks into a single instruction string (so we don't accidentally have hidden prompt and investigator into one).
    Args:
        *parts (Optional[str]): Multiple optional instruction strings.
    Returns:
        Optional[str]: Merged instruction string or None if all parts are None.
    """
    blocks = [p.strip() for p in parts if p and p.strip()] 
    if not blocks:
        return None
    return "\n\n".join(blocks)

def _strip_if_model_starts_writing_other_role(text: str) -> str:
    """
    GPT2 tend to continue with both sides. If we detect that the model started writing the other role, we strip that part.
    Args:
        text (str): The generated text from the model.
    Returns:
        str: The cleaned text with only the intended role's content.
    """
    for stop in ["\nUser:", "\nAssistant:"]:
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


# === 3. TRUNCATION MANAGEMENT ===

def truncate_history_to_max_tokens(
        tokenizer,
        *,
        use_chat_template: bool,
        topic_prompt: str,
        history: List[Dict[str, Any]],
        next_role: str,
        extra_instruction: Optional[str],
        max_context_tokens: int,
    ) -> List[Dict[str, Any]]:
    """
    Ensure we don't exceed max tokens by dropping oldest turns (useful when we do lots of rounds).
    Args:
        tokenizer: The tokenizer object.
        use_chat_template (bool): Whether to use chat template formatting.
        topic_prompt (str): The topic prompt.
        history (List[Dict[str, Any]]): The conversation history.
        next_role (str): The role of the next message to generate ("user" or "assistant").
        extra_instruction (Optional[str]): Any extra instruction to include.
        max_context_tokens (int): Maximum allowed tokens for context.
    Returns:
        List[Dict[str, Any]]: The truncated conversation history.
    Logic: While the total tokens exceed max_context_tokens, drop the oldest turn (2 messages).
    """
    if max_context_tokens is None or max_context_tokens <= 0:
        return history
    
    while True:
        if use_chat_template:
            msgs = build_chat_messages(
                topic_prompt = topic_prompt,
                history = history,
                hidden_style_prompt = extra_instruction,
            )

            # add_generation_prompt = True is needed for Qwen
            enc = tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt = True,
                tokenize = True,
                return_tensors = "pt",
            )
            prompt_len = int(enc.shape[-1])
        
        else:
            prompt_text = build_next_prompt_plaintext(
                topic_prompt = topic_prompt,
                history = history,
                next_role = next_role,
                hidden_style_prompt = extra_instruction,
            )
            prompt_len = int(tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[-1])
        
        if prompt_len <= max_context_tokens:
            return history
        
        # if we can't drop at least one round, stop
        if len(history) <= 2:
            return history
        
        # drop oldest round (2 messages)
        history = history[2:]


# === 4. GENERATE ONE TURN ===

def generate_one_turn(
        *,
        tokenizer,
        model,
        device: str,
        use_chat_template: bool,
        topic_prompt: str,
        history: List[Dict[str, Any]],
        next_role: str,
        extra_instruction: Optional[str],
        gen_cfg: Dict[str, Any],
    ) -> Tuple[str, int, int]:
    """
    Generate one message given the topic and history.
    Args:
        tokenizer: The tokenizer object.
        model: The model object.
        device (str): The device to run the model on.
        use_chat_template (bool): Whether to use chat template formatting.
        topic_prompt (str): The topic prompt.
        history (List[Dict[str, Any]]): The conversation history.
        next_role (str): The role of the next message to generate ("user" or "assistant").
        extra_instruction (Optional[str]): Any extra instruction to include.
        gen_cfg (Dict[str, Any]): Generation configuration parameters.
    Returns:
        Tuple[str, int, int]: The generated message text, number of input prompt tokens, number of generated tokens.
    """
    # 1. BUILD MODEL INPUT 

    # use_chat_template models like Qwen
    if use_chat_template:
        msgs = build_chat_messages(
            topic_prompt = topic_prompt,
            history = history,
            hidden_style_prompt = extra_instruction,
        )
        # special tokens for signaling generation
        enc = tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt = True,
            tokenize = True,
            return_tensors = "pt",
        )
        input_ids = enc.to(device)
        attention_mask = torch.ones_like(input_ids)

    # plaintext models like GPT2
    else:
        prompt_text = build_next_prompt_plaintext(
            topic_prompt = topic_prompt,
            history = history,
            next_role = next_role,
            hidden_style_prompt = extra_instruction,
        )
        enc = tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

    prompt_tokens = int(input_ids.shape[-1])

    # 2. GENERATE CONTINUATION TOKENS
    with torch.no_grad():
        out = model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = int(gen_cfg["max_new_tokens"]),
            do_sample = bool(gen_cfg["do_sample"]),
            temperature = float(gen_cfg["temperature"]),
            top_p = float(gen_cfg["top_p"]),
            repetition_penalty = float(gen_cfg["repetition_penalty"]),
            pad_token_id = tokenizer.eos_token_id,
        )
    
    new_ids = out[0, prompt_tokens:]
    gen_tokens = int(new_ids.shape[-1])

    # 3. DECODE AND CLEAN
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # for GPT2-like models, strip any continuation of the other role
    if not use_chat_template:
        text = _strip_if_model_starts_writing_other_role(text)
    else:
        text = text.strip()
    
    return text, prompt_tokens, gen_tokens


# === 5. GENERATE FULL TRANSCRIPT ===

def generate_full_transcript(
        cfg: Dict[str, Any],
        *,
        experiment_id: str,
        seed: int,
    ) -> None:
    """
    Generate a transcript,jsonl for one experiment and one seed. 
    Args:
        cfg (Dict[str, Any]): The full experiment configuration dictionary.
        experiment_id (str): The experiment identifier.
        seed (int): The random seed for the run.
    Output structure is managed by io_utils.build_run_paths():
        outputs/<experiment_id>/seed_001/<run_tag>/transcript.jsonl
    Logic:
        1. Load config blocks and set random seed
        2. Prepare output directory + paths and metadata
        3. Load model/tokenizer
        4. Define conversation state (topic prompt, history, who speaks first, turn_idx)
        5. For each round:
            1. Determine speaker, role, and per-turn instructions
            2. Truncate history if needed
            3. Generate one turn
            4. Update history
            5. Log to transcript.jsonl
    """
    # 1. LOAD CONFIG BLOCKS AND SET RANDOM SEED

    # general data
    global_cfg = cfg["global"]
    model_cfg = cfg["model"]
    gen_cfg = cfg["generation"]

    # experiment specific data
    styles_cfg = cfg["styles"]
    investigator_cfg = cfg["investigator"]
    exp_cfg = cfg["experiment"][experiment_id]

    # set random seed for reproducibility
    set_global_seed(seed)

    # 2. PREPARE OUTPUT DIRECTORY + PATHS AND METADATA

    # paths using io_utils
    paths = build_run_paths(
        output_dir = global_cfg["output_dir"],
        experiment_id = experiment_id,
        seed = seed,
        run_tag = global_cfg.get("run_tag", None),
    )

    # initialize run directory
    init_run_dir(
        paths,
        resolved_config = cfg,
        extra_meta = {"experiment_id": experiment_id, "seed": seed}
    )

    # 3. LOAD MODEL/TOKENIZER

    # load tokenizer and model
    loaded = load_tokenizer_and_model(
        model_name = model_cfg["name"],
        device = model_cfg.get("device", "auto"),
        device_map = model_cfg.get("device_map", "auto"),
        dtype = model_cfg.get("dtype", "float32"),
        hf_cache_dir = model_cfg.get("hf_cache_dir", None),
    )
    tokenizer = loaded.tokenizer
    model = loaded.model
    device = loaded.device

    # use_chat_template vs plaintext
    use_chat_template = bool(model_cfg.get("use_chat_template", False))

    # 4. DEFINE CONVERSATION STATE

    # load topic prompt
    topic_prompt = global_cfg["topic_prompt"]
    num_rounds = int(global_cfg["num_rounds"])
    max_context_tokens = int(gen_cfg.get("max_context_tokens", 10**9))

    # conversation history
    history: List[Dict[str, Any]] = []

    # who speaks first
    a_first = bool(global_cfg.get("a_speaks_first", True))

    # init turn index
    turn_idx = 0

    # 5. GENERATE CONVERSATION ROUND BY ROUND
    for round_idx in range(num_rounds):
        # each round has two turns: A then B (or B then A)
        for step_in_round in range(2):

            # 1. DETERMINE SPEAKER, ROLE, AND PER-TURN INSTRUCTIONS

            # speaker
            if a_first:
                speaker = "A" if step_in_round == 0 else "B"
            else:
                speaker = "B" if step_in_round == 0 else "A"
            
            # role
            role = speaker_to_role(speaker)

            # per turn instructions => hidden prompt
            hidden_style_prompt = get_hidden_style_prompt_for_turn(
                experiment_cfg = exp_cfg,
                styles_cfg = styles_cfg,
                speaker = speaker,
                round_idx = round_idx,
                base_seed = seed,
            )
            hidden_style_id = get_hidden_style_id_for_round(
                experiment_cfg = exp_cfg,
                speaker = speaker,
                round_idx = round_idx,
                base_seed = seed,
            )

            # per turn instructions => investigator mode
            investigator_prompt = get_investigator_prompt_for_turn(
                experiment_cfg = exp_cfg,
                investigator_cfg = investigator_cfg,
                styles_cfg = styles_cfg,
                speaker = speaker,
            )

            # merge instructions
            extra_instruction = join_extra_instructions(investigator_prompt, hidden_style_prompt)  # IF WE PUT INVESTIGATOR SECOND, IT MIGHT BE TOO SALIENT?

            # 2. TRUNCATE HISTORY IF NEEDED
            history = truncate_history_to_max_tokens(
                tokenizer = tokenizer,
                use_chat_template = use_chat_template,
                topic_prompt = topic_prompt,
                history = history,
                next_role = role,
                extra_instruction = extra_instruction,
                max_context_tokens = max_context_tokens,
            )

            # 3. GENERATE THE NEXT MESSAGE
            text, prompt_tokens, gen_tokens = generate_one_turn(
                tokenizer = tokenizer,
                model = model,
                device = device,
                use_chat_template = use_chat_template,
                topic_prompt = topic_prompt,
                history = history,
                next_role = role,
                extra_instruction = extra_instruction,
                gen_cfg = gen_cfg,
            )

            # 4. UPDATE HISTORY
            history.append({"role": role, "content": text})

            # 5. LOG TO TRANSCRIPT.JSONL
            record = {
                    "turn_idx": turn_idx,
                    "round_idx": round_idx,
                    "speaker": speaker,          # "A" or "B"
                    "role": role,                # "user" or "assistant"
                    "content": text,
                    "hidden_style_id": hidden_style_id,  # None if not applied this turn
                    "investigator_active": investigator_prompt is not None,
                    "prompt_tokens": prompt_tokens,
                    "gen_tokens": gen_tokens,
                    "ts_unix": int(time.time()),
            }
            append_jsonl(paths.transcript_path, record)

            # increment turn index
            turn_idx += 1