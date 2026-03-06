#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:
#SBATCH --requeue


set -euo pipefail

mkdir -p logs

# Load conda (since we're already inside the container)
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH


conda activate llmr

cd /scratch/sv2279/nyu-llm-reasoners-a2/student
mkdir -p results/logs


# ── Question (b): forward + backward, warmup=5 ───────────────────────────────
echo '===== Question (b): forward + backward, warmup=5 ====='
uv run benchmark.py \
    --model_size all \
    --context_length 128 \
    --mode forward_backward \
    --warmup 5 \
    --steps 10 \
    --device cuda \
    --dtype float32 \
    --run_name q_b_fwd_bwd_warmup5

echo '===== Question (b): forward only, warmup=5 ====='
uv run benchmark.py \
    --model_size all \
    --context_length 128 \
    --mode forward \
    --warmup 5 \
    --steps 10 \
    --device cuda \
    --dtype float32 \
    --run_name q_b_fwd_only_warmup5

# ── Question (c): no warmup ───────────────────────────────────────────────────
echo '===== Question (c): no warmup ====='
uv run benchmark.py \
    --model_size all \
    --context_length 128 \
    --mode forward_backward \
    --warmup 0 \
    --steps 10 \
    --device cuda \
    --dtype float32 \
    --run_name q_c_warmup0

# ── Question (c): 1 warmup step ──────────────────────────────────────────────
echo '===== Question (c): 1 warmup step ====='
uv run benchmark.py \
    --model_size all \
    --context_length 128 \
    --mode forward_backward \
    --warmup 1 \
    --steps 10 \
    --device cuda \
    --dtype float32 \
    --run_name q_c_warmup1

# ── Question (c): 2 warmup steps ─────────────────────────────────────────────
echo '===== Question (c): 2 warmup steps ====='
uv run benchmark.py \
    --model_size all \
    --context_length 128 \
    --mode forward_backward \
    --warmup 2 \
    --steps 10 \
    --device cuda \
    --dtype float32 \
    --run_name q_c_warmup2

# ── Question 1.1.5 (c): 2 warmup steps ─────────────────────────────────────────────

echo '===== 1.1.5c: BF16 mixed precision: forward + backward ====='
uv run benchmark.py \
    --model_size all \
    --context_length 128 \
    --mode forward_backward \
    --warmup 5 \
    --steps 10 \
    --device cuda \
    --dtype float32 \
    --mixed_precision \
    --run_name q_mixed_bf16_autocast

echo 'All done. Results in results/'
"