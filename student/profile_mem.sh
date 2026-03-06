#!/bin/bash
#SBATCH --job-name=memory_profile
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

set -euo pipefail
mkdir -p logs
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
conda activate llmr
cd /scratch/sv2279/nyu-llm-reasoners-a2/student
mkdir -p memory_snapshots

# ── (a) + (b): forward and training, FP32, context lengths 128/256/512 ───────
for ctx in 128 256 512; do
    echo "=== xl ctx=$ctx FORWARD (FP32) ==="
    uv run python benchmark.py \
        --model_size xl \
        --context_length $ctx \
        --mode forward \
        --dtype float32 \
        --steps 1 \
        --memory_profiling 1 \
        --memory_output memory_snapshots/xl_ctx${ctx}_fwd_fp32.pickle

    echo "=== xl ctx=$ctx TRAINING (FP32) ==="
    uv run python benchmark.py \
        --model_size xl \
        --context_length $ctx \
        --mode forward_backward \
        --dtype float32 \
        --steps 1 \
        --memory_profiling 1 \
        --memory_output memory_snapshots/xl_ctx${ctx}_training_fp32.pickle
done

# ── (c): mixed precision BF16, context lengths 128/256/512 ───────────────────
for ctx in 128 256 512; do
    echo "=== xl ctx=$ctx FORWARD (BF16 mixed) ==="
    uv run python benchmark.py \
        --model_size xl \
        --context_length $ctx \
        --mode forward \
        --dtype float32 \
        --steps 1 \
        --mixed_precision 1 \
        --memory_profiling 1 \
        --memory_output memory_snapshots/xl_ctx${ctx}_fwd_bf16.pickle

    echo "=== xl ctx=$ctx TRAINING (BF16 mixed) ==="
    uv run python benchmark.py \
        --model_size xl \
        --context_length $ctx \
        --mode forward_backward \
        --dtype float32 \
        --steps 1 \
        --mixed_precision 1 \
        --memory_profiling 1 \
        --memory_output memory_snapshots/xl_ctx${ctx}_training_bf16.pickle
done

echo "All done. Snapshots in memory_snapshots/"