# src/model_loader.py

# === 1. IMPORTS ===

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np
from transformers import set_seed as hf_set_seed


# === 2. MODEL LOADER CLASS & FUNCTIONS ===

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# define dataclass
@dataclass
class LoadedModel:
    tokenizer: object
    model: object
    device: str
    torch_dtype: torch.dtype

# define functions 
def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """
    Resolve the string representation of a dtype to a torch.dtype.
    Args:
        dtype_str (str): The string representation of the dtype.
    Returns: 
        torch.dtype: The corresponding torch.dtype.
    """
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"!! Unsupported dtype '{dtype_str}'. Supported dtypes are: {list(_DTYPE_MAP.keys())}")
    return _DTYPE_MAP[dtype_str]

def _resolve_device(device_str: str) -> str:
    """
    Resolve the string representation of a device to a torch device string.
    Args:
        device_str (str): The string representation of the device ("cpu", "cuda", "auto")
    Returns:
        str: The corresponding torch device string.
    """
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device_str in ["cpu", "cuda"]:
        return device_str
    else:
        raise ValueError(f"!! Unsupported device '{device_str}'. Supported devices are: 'cpu', 'cuda', 'auto'")
    

# === 3. LOAD MODEL FUNCTIONS ===

def load_tokenizer_and_model(
        model_name: str,
        device: str = "auto",
        device_map: str = "auto",
        dtype: str = "float32",
        hf_cache_dir: Optional[str] = None,
    ) -> LoadedModel:
    """
    Loads the tokenier and model from Hugging Face Hub.
    Args:
        model_name (str): The name of the model to load from Hugging Face Hub.
        device (str, optional): The device to load the model onto ("cpu", "cuda", "auto"). Defaults to "auto".
        device_map (str, optional): The device map for loading the model. Defaults to "auto".
        dtype (str, optional): The data type for the model ("float16", "float32", "bfloat16"). Defaults to "float32".
        hf_cache_dir (Optional[str], optional): The cache directory for Hugging Face models. Defaults to None.
    Returns:
        LoadedModel: A dataclass containing the tokenizer, model, device, and torch_dtype.
    Notes: ensure pad_token is set.
    """

    # resolve device and dtype
    resolved_device = _resolve_device(device)
    torch_dtype = _resolve_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir = hf_cache_dir,
        use_fast = True,
        trust_remote_code = True  # safe for Qwen, harmless for gpt2
    )

    # set pad_token if not present (especially for decoder-only models like gpt2)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    
    # load model (for cpu and cuda cases)
    if resolved_device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir = hf_cache_dir,
            torch_dtype = torch_dtype,
            trust_remote_code = True
        )
        model.to(resolved_device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir = hf_cache_dir,
            device_map = device_map,
            torch_dtype = torch_dtype,
            trust_remote_code = True
        )
        
    model.eval()

    return LoadedModel(
        tokenizer = tokenizer,
        model = model,
        device = resolved_device,
        torch_dtype = torch_dtype
    )

def set_global_seed(seed: int) -> None:
    """
    Best-effort reproducibility across torch and cuda.
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
