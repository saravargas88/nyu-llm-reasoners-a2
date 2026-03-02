#!/bin/bash
#SBATCH --job-name=benchmark_ab
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

mkdir -p logs

singularity exec --bind /scratch --nv \
--overlay /scratch/sv2279/overlay-25GB-500K.ext3:r \
/scratch/sv2279/ubuntu-20.04.3.sif \
/bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
set -euo pipefail
conda activate llmr
cd /scratch/sv2279/repo_dir

mkdir -p results/logs

# ── Question (b): forward + backward, warmup=5 ───────────────────────────────
echo '===== Question (b): forward + backward, warmup=5 ====='
python benchmark.py \
    --model_size all \
    --context_length 512 \
    --mode forward_backward \
    --warmup 5 \
    --steps 10 \
    --device cuda \
    --dtype float32 \
    --run_name q_b_fwd_bwd_warmup5

echo '===== Question (b): forward only, warmup=5 ====='
python benchmark.py \
    --model_size all \
    --context_length 512 \
    --mode forward \
    --warmup 5 \
    --steps 10 \
    --device cuda \
    --dtype float32 \
    --run_name q_b_fwd_only_warmup5

# ── Question (c): no warmup ───────────────────────────────────────────────────
echo '===== Question (c): no warmup ====='
python benchmark.py \
    --model_size all \
    --context_length 512 \
    --mode forward_backward \
    --warmup 0 \
    --steps 10 \
    --device cuda \
    --dtype float32 \
    --run_name q_c_warmup0

# ── Question (c): 1 warmup step ──────────────────────────────────────────────
echo '===== Question (c): 1 warmup step ====='
python benchmark.py \
    --model_size all \
    --context_length 512 \
    --mode forward_backward \
    --warmup 1 \
    --steps 10 \
    --device cuda \
    --dtype float32 \
    --run_name q_c_warmup1

# ── Question (c): 2 warmup steps ─────────────────────────────────────────────
echo '===== Question (c): 2 warmup steps ====='
python benchmark.py \
    --model_size all \
    --context_length 512 \
    --mode forward_backward \
    --warmup 2 \
    --steps 10 \
    --device cuda \
    --dtype float32 \
    --run_name q_c_warmup2

echo 'All done. Results in results/'
"