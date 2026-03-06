#!/bin/bash
set -e

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


echo "Running attention benchmark..."
uv run attention_bench.py 2>&1 | tee benchmark_results.log

echo ""
echo "Done! Results saved to benchmark_results.log"