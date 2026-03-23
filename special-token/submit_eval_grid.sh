#!/bin/bash
#SBATCH --job-name=eval_grid
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_grid_%j.out
#SBATCH --error=logs/eval_grid_%j.err
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmamora2003@gmail.com

set -euo pipefail

echo "[info] job_id=${SLURM_JOB_ID:-no_slurm}"
echo "[info] node=$(hostname)"
echo "[info] start=$(date)"
echo "[info] partition=${SLURM_JOB_PARTITION:-unknown}"

# repo root
REPO_ROOT="/home/3210604/projects/llm-wm/special-token"
cd "$REPO_ROOT"

mkdir -p logs
mkdir -p data
mkdir -p data/runs

# env
source ../llm-vs-llm/.venv/bin/activate

# --- Cache locations (cluster-friendly) ---
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false

# make runs more stable/consistent
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python=$(which python)"
echo "[info] gpu info:"
nvidia-smi || true

# sanity checks
if [ ! -f "run_eval_grid.py" ]; then
  echo "[error] run_eval_grid.py not found in ${REPO_ROOT} — aborting"
  exit 1
fi

if [ ! -f "evaluate_special_token.py" ]; then
  echo "[error] evaluate_special_token.py not found in ${REPO_ROOT} — aborting"
  exit 1
fi

if [ ! -s "data/examples.jsonl" ]; then
  echo "[error] data/examples.jsonl is missing or empty — aborting"
  exit 1
fi

if [ ! -d "data/runs" ] || [ -z "$(find data/runs -maxdepth 2 -name 'run_summary.json' -print -quit)" ]; then
  echo "[error] data/runs has no completed training runs (no run_summary.json found) — aborting"
  exit 1
fi

echo "[info] launching eval grid..."

python -u run_eval_grid.py \
  --repo_root . \
  --examples_path data/examples.jsonl \
  --runs_root data/runs \
  --evals_root data/evals \
  --personas Emma,Maria \
  --styles WhatIf \
  --allowed_personas Emma,Maria \
  --allowed_styles PickOne,WhatIf \
  --topics career_learning,productivity_habits,ethics_decision_making \
  --token_counts 0 15 \
  --token_placements after_context \
  --position_modes default shared_position \
  --include_baseline \
  --include_trained \
  --generation_max_new_tokens 128 \
  --save_per_example \
  --seed 42

echo "[info] end=$(date)"