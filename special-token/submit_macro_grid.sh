#!/bin/bash
#SBATCH --job-name=macro_grid
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/macro_grid_%j.out
#SBATCH --error=logs/macro_grid_%j.err
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

# Optional: make runs more stable/consistent
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python=$(which python)"
echo "[info] gpu info:"
nvidia-smi || true

# sanity checks
if [ ! -f "run_macro_grid.py" ]; then
  echo "[error] run_macro_grid.py not found in ${REPO_ROOT} — aborting"
  exit 1
fi

if [ ! -f "train_special_token.py" ]; then
  echo "[error] train_special_token.py not found in ${REPO_ROOT} — aborting"
  exit 1
fi

if [ ! -s "data/examples.jsonl" ]; then
  echo "[error] data/examples.jsonl is missing or empty — aborting"
  exit 1
fi

echo "[info] launching macro grid..."

python -u run_macro_grid.py \
  --repo_root . \
  --examples_path data/examples.jsonl \
  --runs_root data/runs \
  --base_persona_ids Emma Maria \
  --style_ids PickOne WhatIf \
  --topic_ids \
    career_learning \
    productivity_habits \
    relationships_communication \
    health_wellbeing \
    travel_living_abroad \
    personal_finance_basics \
    tech_everyday \
    current_events_civic \
    ethics_decision_making \
    creative_projects \
  --run_baseline \
  --baseline_token_placement after_context \
  --baseline_position_mode default \
  --num_special_tokens_list 1 3 5 10 15 \
  --token_placements after_context \
  --position_modes shared_position \
  --model_name Qwen/Qwen2.5-0.5B \
  --max_length 1024 \
  --batch_size 1 \
  --num_epochs 3 \
  --learning_rate 5e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --grad_accum_steps 1 \
  --max_grad_norm 0.5 \
  --eval_every_steps 20 \
  --seed 42

echo "[info] end=$(date)"