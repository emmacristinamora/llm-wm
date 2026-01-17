# src/prompts.py

# === 1. IMPORTS ===

from __future__ import annotations
from typing import Any, List, Dict, Optional


# === 2. FUNCTIONS ===

def build_system_prefix(topic_prompt: str) -> str:
    """
    For non-chat models, emulate a system prompt by prepending instructions to the topic prompt.
    Args:
        topic_prompt (str): The original topic prompt.
    Returns:
        str: The modified prompt with system instructions.
    """
    return f"Topic: {topic_prompt}\n\n"

def format_history_plaintext(messages: List[Dict[str, Any]]) -> str:
    """
    Plaintext format for gpt2 style models. Expected message dict keys: role in {"user", "assistant"}, content (str).
    Args:
        messages (List[Dict[str, Any]]): List of message dictionaries.
    Returns:
        str: Formatted conversation history as plaintext.
    """
    lines = []
    if not messages:
        return ""
    for m in messages:
        if m["role"] == "user":
            lines.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"Assistant: {m['content']}")
        else:
            raise ValueError(f"!! Unsupported role '{m['role']}' in message.")
    return "\n".join(lines).strip() + "\n"

def build_next_prompt_plaintext(
        topic_prompt: str,
        history: List[Dict[str, Any]],
        next_role: str,
        hidden_style_prompt: Optional[str] = None,
    ) -> str:
    """
    Build the full input text to feed the model to generate the next message in plaintext format.
    Args:
        topic_prompt (str): The topic prompt.
        history (List[Dict[str, Any]]): List of message dictionaries representing the conversation history.
        next_role (str): The role of the next message to generate ("user" or "assistant").
        hidden_style_prompt (Optional[str]): Applied only when the next speaker is the "hidden" one.
    Returns:
        str: The full input text for the model.
    """
    prefix = build_system_prefix(topic_prompt)

    style_block = ""
    if hidden_style_prompt:
        style_block = f"System: {hidden_style_prompt}\n\n"
    history_block = format_history_plaintext(history)

    if next_role == "user":
        # generate next User message
        return prefix + style_block + history_block + "User: "
    elif next_role == "assistant":
        # generate next Assistant message
        return prefix + style_block + history_block + "Assistant: "
    else:
        raise ValueError(f"!! Unsupported next_role '{next_role}'. Supported roles are: 'user', 'assistant'")
    

# === 3. GENERIC CHAT TEMPLATE SUPPORT FUNCTION ===

def build_chat_messages(
        topic_prompt: str,
        history: List[Dict[str, Any]],
        # next_role: str, ## MAYBE IMPLEMENT LATER?
        hidden_style_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
    """
    Build the full input messages to feed the Qwen model to generate the next message (good for tokenizer.apply_chat_template).
    Args:
        topic_prompt (str): The topic prompt.
        history (List[Dict[str, Any]]): List of message dictionaries representing the conversation history.
        # next_role (str): The role of the next message to generate ("user" or "assistant").
        hidden_style_prompt (Optional[str]): Applied only when the next speaker is the "hidden" one.
    Returns:
        List[Dict[str, str]]: The full input messages for the model.
    """
    msgs: List[Dict[str, str]] = [{"role": "system", "content": topic_prompt}]

    if hidden_style_prompt:
        msgs.append({"role": "system", "content": hidden_style_prompt})

    msgs.extend({"role": m["role"], "content": m["content"]} for m in history)

    # TO DO: add a generation cue message if needed by the model

    return msgs
