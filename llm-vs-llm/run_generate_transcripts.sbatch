#!/bin/bash
#SBATCH --job-name=gen_transcripts
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/gen_transcripts_%j.out
#SBATCH --error=logs/gen_transcripts_%j.err
#SBATCH --requeue

set -euo pipefail

echo "[info] job_id=${SLURM_JOB_ID}"
echo "[info] node=$(hostname)"
echo "[info] start=$(date)"
echo "[info] partition=${SLURM_JOB_PARTITION:-unknown}"

# --- Go to repo root ---
REPO_ROOT="/home/3210604/projects/llm-wm/llm-vs-llm"
cd "$REPO_ROOT"

# --- Make sure logs exists ---
mkdir -p logs
mkdir -p src/data/conversations

# --- Activate environment ---
source .venv/bin/activate

# --- Cache locations (cluster-friendly) ---
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

# Optional: make runs more stable/consistent
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --- Run ---
python -u src/generate_transcripts.py \
  --experiments_path experiments.jsonl \
  --num_experiments 15 \
  --conversations_per_experiment 5 \
  --num_turns 6 \
  --assistant_max_new_tokens 1200 \
  --assistant_temp 0.7 \
  --user_max_new_tokens 220 \
  --user_temp 0.8 \
  --seed 0 \
  --verbose --print_every 1

echo "[info] end=$(date)"