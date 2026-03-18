# src/evaluate.py

# === IMPORTS ===

from typing import Any, Dict, List, Optional
import torch
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# === LOSS EVALUATION ===

@torch.no_grad()
def compute_eval_loss(
    model,
    dataloader,
    device,
    position_mode: str = "default",
    special_token_ids: Optional[List[int]] = None,
    build_position_ids_fn=None,
) -> float:
    model.eval()

    total_loss = 0.0
    total_examples = 0
    special_token_ids = special_token_ids or []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if position_mode == "shared_position":
            if build_position_ids_fn is None:
                raise ValueError("build_position_ids_fn must be provided when position_mode='shared_position'")
            forward_kwargs["position_ids"] = build_position_ids_fn(
                input_ids=input_ids,
                special_token_ids=special_token_ids,
            )

        outputs = model(**forward_kwargs)
        loss = outputs.loss

        batch_size = input_ids.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


# === GENERATION EVALUATION ===

@torch.no_grad()
def generate_predictions(
    model,
    tokenizer,
    examples: List[Dict[str, Any]],
    build_prompt_fn,
    device,
    generation_max_new_tokens: int = 80,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.95,
    position_mode: str = "default",
    special_token_ids: Optional[List[int]] = None,
    build_position_ids_fn=None,
    max_examples: Optional[int] = 100,
) -> List[Dict[str, Any]]:
    model.eval()

    generations = []
    special_token_ids = special_token_ids or []

    if max_examples is not None:
        examples = examples[:max_examples]

    for example in examples:
        prompt_text = build_prompt_fn(example)

        enc = tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=True,
            max_length=tokenizer.model_max_length if tokenizer.model_max_length < 100000 else 2048,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": generation_max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        if position_mode == "shared_position":
            if build_position_ids_fn is None:
                raise ValueError("build_position_ids_fn must be provided when position_mode='shared_position'")
            gen_kwargs["position_ids"] = build_position_ids_fn(
                input_ids=input_ids,
                special_token_ids=special_token_ids,
            )

        generated = model.generate(**gen_kwargs)
        continuation_ids = generated[0, input_ids.shape[1]:]
        prediction = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()

        generations.append(
            {
                "example_id": example.get("example_id"),
                "transcript_id": example.get("transcript_id"),
                "user_turn_number": example.get("user_turn_number"),
                "prompt_text": prompt_text,
                "gold_target": example.get("target_message", ""),
                "prediction": prediction,
            }
        )

    return generations


# === COSINE SIMILARITY ===

def compute_mean_cosine_similarity(
    generations: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return {
            "available": False,
            "mean_cosine_similarity": None,
            "per_example_similarity": None,
            "model_name": model_name,
            "note": "sentence-transformers not installed",
        }

    if len(generations) == 0:
        return {
            "available": True,
            "mean_cosine_similarity": None,
            "per_example_similarity": [],
            "model_name": model_name,
        }

    sentence_model = SentenceTransformer(model_name)

    gold_texts = [row["gold_target"] for row in generations]
    pred_texts = [row["prediction"] for row in generations]

    gold_embeddings = sentence_model.encode(gold_texts, convert_to_tensor=True, show_progress_bar=False)
    pred_embeddings = sentence_model.encode(pred_texts, convert_to_tensor=True, show_progress_bar=False)

    similarities = st_util.cos_sim(pred_embeddings, gold_embeddings).diagonal().cpu().tolist()
    mean_similarity = sum(similarities) / len(similarities)

    return {
        "available": True,
        "mean_cosine_similarity": mean_similarity,
        "per_example_similarity": similarities,
        "model_name": model_name,
    }


# === HIGH-LEVEL WRAPPER ===

def evaluate_run(
    model,
    tokenizer,
    eval_examples: List[Dict[str, Any]],
    eval_dataloader,
    build_prompt_fn,
    device,
    position_mode: str = "default",
    special_token_ids: Optional[List[int]] = None,
    build_position_ids_fn=None,
    generation_max_new_tokens: int = 80,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_generation_examples: int = 100,
    cosine_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    eval_loss = compute_eval_loss(
        model=model,
        dataloader=eval_dataloader,
        device=device,
        position_mode=position_mode,
        special_token_ids=special_token_ids,
        build_position_ids_fn=build_position_ids_fn,
    )

    generations = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        build_prompt_fn=build_prompt_fn,
        device=device,
        generation_max_new_tokens=generation_max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        position_mode=position_mode,
        special_token_ids=special_token_ids,
        build_position_ids_fn=build_position_ids_fn,
        max_examples=max_generation_examples,
    )

    cosine_metrics = compute_mean_cosine_similarity(
        generations=generations,
        model_name=cosine_model_name,
    )

    return {
        "eval_loss": eval_loss,
        "generations": generations,
        "cosine_metrics": cosine_metrics,
    }