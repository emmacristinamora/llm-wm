#!/bin/bash
#SBATCH --job-name=gen_transcripts
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/gen_transcripts_%j.out
#SBATCH --error=logs/gen_transcripts_%j.err

set -euo pipefail

echo "[info] job_id=${SLURM_JOB_ID}"
echo "[info] node=$(hostname)"
echo "[info] start=$(date)"

# --- Go to repo root ---
REPO_ROOT="/home/3210604/projects/llm-wm/llm-vs-llm"
cd "$REPO_ROOT"

# --- Make sure logs exists ---
mkdir -p logs
mkdir -p src/data/conversations

# --- Activate environment (edit if yours differs) ---
source .venv/bin/activate

# Optional: be explicit about cache locations (nice on clusters)
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Optional: avoid tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# --- Run ---
python -u src/generate_transcripts.py \
  --experiments_path experiments.jsonl \
  --num_experiments 1 \
  --conversations_per_experiment 1 \
  --num_turns 3 \
  --assistant_max_new_tokens 200 \
  --assistant_temp 0.7 \
  --user_max_new_tokens 150 \
  --user_temp 0.8 \
  --seed 0

echo "[info] end=$(date)"