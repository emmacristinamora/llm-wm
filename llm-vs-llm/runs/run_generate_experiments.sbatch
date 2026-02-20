#!/bin/bash
#SBATCH --job-name=gen_experiments
#SBATCH --partition=gpu
#SBATCH --account=3210604
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/gen_experiments_%j.out
#SBATCH --error=logs/gen_experiments_%j.err

set -euo pipefail

cd ~/projects/llm-wm/llm-vs-llm
source .venv/bin/activate
mkdir -p logs

# cache model files in your home (recommended)
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE

# exact model
export QWEN_MODEL="Qwen/Qwen3-8B"

python -u src/generate_experiments.py
