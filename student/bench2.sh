#!/bin/bash
# bench2.sh
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PATH=/share/apps/cuda/11.1.74/bin:$PATH

NSYS=$(find ~/tools/nsight-systems -name nsys -type f | head -1)
echo "Using nsys: $NSYS"
$NSYS --version

SIZES="small medium large xl 2.7B"
CONTEXTS="128 256 512 1024"

mkdir -p nsight_log

for size in $SIZES; do
    for ctx in $CONTEXTS; do

        echo "========================================"
        echo "Profiling $size ctx=$ctx forward..."
        echo "========================================"
        $NSYS profile \
            --trace=cuda,nvtx,osrt \
            --stats=true \
            --force-overwrite=true \
            --output=nsight_log/profile_${size}_ctx${ctx}_fwd \
            uv run python profiling.py \
            --model_size $size \
            --context_length $ctx \
            --mode forward \
            2>&1 | tee nsight_log/stats_${size}_ctx${ctx}_fwd.txt

        echo "========================================"
        echo "Profiling $size ctx=$ctx forward+backward..."
        echo "========================================"
        $NSYS profile \
            --trace=cuda,nvtx,osrt \
            --stats=true \
            --force-overwrite=true \
            --output=nsight_log/profile_${size}_ctx${ctx}_fwdbwd \
            uv run python profiling.py \
            --model_size $size \
            --context_length $ctx \
            --mode forward_backward \
            2>&1 | tee nsight_log/stats_${size}_ctx${ctx}_fwdbwd.txt

    done
done
```

The key additions are `2>&1 | tee nsight_log/stats_....txt` which saves all the terminal output (including the `--stats=true` kernel summary) to a text file per run. That means you'll have the kernel timing data in plain text to answer questions (b), (c), (d), (e) without needing the GUI at all.

The `--stats=true` output will show you a **CUDA GPU Kernel Summary** directly in the terminal that looks like:
```
 Time(%)  Total Time (ns)  Instances  Avg (ns)  ...  Name
 -------  ---------------  ---------  --------  ...  ----
   45.2%      123456789       144      856657   ...  ampere_sgemm_...
   12.3%       33456789       144      232339   ...  softmax_kernel...