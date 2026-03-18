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

# repo root
REPO_ROOT="/home/3210604/projects/llm-wm/special-token/synthetic-conversations"
cd "$REPO_ROOT"

mkdir -p logs
mkdir -p data

# env
source ../../llm-vs-llm/.venv/bin/activate

# --- Cache locations (cluster-friendly) ---
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false

# Optional: make runs more stable/consistent
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# OUT_CONV="data/conversations/with_cat_persona_job${SLURM_JOB_ID}.jsonl"
# OUT_INV="data/conversations/investigation_meta_job${SLURM_JOB_ID}.jsonl"

# echo "[info] out_conv=${OUT_CONV}"
# echo "[info] out_inv=${OUT_INV}"

# experiment generation
python -u generate_experiments.py

if [ ! -s data/experiments.jsonl ]; then
  echo "[error] data/experiments.jsonl is empty — aborting"
  exit 1
fi

# transcript generation
python -u generate_transcripts.py \
    --num_turns 15 \
    --conversations_per_experiment 1 \
    --assistant_max_new_tokens 900 \
    --user_max_new_tokens 400

echo "[info] end=$(date)"