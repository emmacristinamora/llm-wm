# src/io_utils.py

# === 1. IMPORTS ===

from __future__ import annotations
import csv 
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


# === 2. PATHS / RUN DIRECTORY ===

@dataclass(frozen=True)
class RunPaths:
    """
    Centralized paths for a single run so that we have a consistent structure.
    """
    run_dir: Path
    transcript_path: Path
    scores_path: Path
    resolved_config_path: Path
    meta_path: Path

def build_run_paths(
        output_dir: str | Path,
        experiment_id: str,
        seed: int,
        run_tag: Optional[str] = None,
    ) -> RunPaths:
    """
    Create a stable run directory structure: {output_dir}/{experiment_id}/seed_{seed:03d}/{run_tag_optional}/
    Args:
        output_dir (str | Path): Base output directory.
        experiment_id (str): Experiment identifier.
        seed (int): Random seed for the run.
        run_tag (Optional[str]): Optional tag for the run.
    Returns:
        RunPaths: Dataclass containing all relevant paths for the run.
    """

    base = Path(output_dir) / experiment_id / f"seed_{seed:03d}"
    if run_tag:
        base = base / run_tag

    run_dir = base
    transcript_path = run_dir / "transcript.jsonl"
    scores_path = run_dir / "scores.csv"
    resolved_config_path = run_dir / "resolved_config.yaml"
    meta_path = run_dir / "meta.json"

    return RunPaths(
        run_dir = run_dir,
        transcript_path = transcript_path,
        scores_path = scores_path,
        resolved_config_path = resolved_config_path,
        meta_path = meta_path,
    )

def ensure_dir(path: str | Path) -> None:
    """
    Ensure that a directory exists (Slurm won't do it automatically).
    Args:
        path (str | Path): The directory path to ensure.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    

# === 3. SAFE WRITES ===

def atomic_write_text(path: Path, text: str) -> None:
    """
    Atomically write text to a file.
    Args:
        path (Path): The file path to write to.
        text (str): The text content to write.
    """
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)

def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    """
    Save a dictionary as a YAML file atomically. Used to write resolved_config.yaml.
    Args:
        path (Path): The file path to write to.
        data (Dict[str, Any]): The dictionary to save.
    """
    yaml_text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    atomic_write_text(path, yaml_text)

def save_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Save a dictionary as a JSON file atomically. Used to write meta.json.
    Args:
        path (Path): The file path to write to.
        data (Dict[str, Any]): The dictionary to save.
    """
    json_text = json.dumps(data, indent=2, ensure_ascii=False)
    atomic_write_text(path, json_text)


# === 4. RUN METADATA ===

def _try_get_git_commit() -> Optional[str]:
    """
    Try to get the current git commit hash (good for tracing results back to the exact code version).
    Returns:
        Optional[str]: The git commit hash if available, else None.
    """
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return commit_hash
    except Exception:
        return None
    
def init_run_dir(
        paths: RunPaths,
        resolved_config: Dict[str, Any],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
    """
    Initialize the run directory and write resolved_config.yaml and meta.json.
    Args:
        paths (RunPaths): The run paths dataclass.
        resolved_config (Dict[str, Any]): The resolved configuration dictionary.
        extra_meta (Optional[Dict[str, Any]]): Additional metadata to include in meta.json.
    """

    ensure_dir(paths.run_dir)

    # save config
    save_yaml(paths.resolved_config_path, resolved_config)

    # prepare meta
    meta = {
        "created_at_unix": int(time.time()),
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _try_get_git_commit(),
        "run_dir": str(paths.run_dir),
    }
    if extra_meta:
        meta.update(extra_meta)

    # save meta
    save_json(paths.meta_path, meta)


# === 5. TRANSCRIPT WRITING (JSONL) ===

def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """
    Append one JSON record to a JSONL file. This ensures we save the transcript incrementally.
    Args:
        path (Path): The file path to append to.
        record (Dict[str, Any]): The record to append.
    """
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# === 6. SCORES WRITING (CSV) ===

def append_csv_row(path: Path, fieldnames: list[str], row: Dict[str, Any]) -> None:
    """
    Append one row to a CSV file. This ensures we save scores incrementally.
    Args:
        path (Path): The file path to append to.
        fieldnames (list[str]): The list of field names (CSV headers).
        row (Dict[str, Any]): The row data to append.
    """
    ensure_dir(path.parent)
    write_header = not path.exists()

    with path.open("a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)