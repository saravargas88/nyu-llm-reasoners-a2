


#test.sh
set -euo pipefail


# Load conda (since we're already inside the container)
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH


conda activate llmr

cd /scratch/sv2279/nyu-llm-reasoners-a2/student
mkdir -p results/logs


# ── Question (b): forward + backward, warmup=5 ───────────────────────────────
echo '===== Question (b): forward + backward, warmup=5 ====='
uv run benchmark.py \
    --model_size 2.7B \
    --context_length 128 \
    --mode forward_backward \
    --warmup 5 \
    --steps 10 \
    --device cuda \
    --dtype float32 \
    --run_name test_train_2.7B