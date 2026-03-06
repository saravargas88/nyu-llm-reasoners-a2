#!/bin/bash

SIZES="small medium large xl 2.7B"
CTX=1024
WARMUP=5
STEPS=10
MODE="forward_backward"
OUTPUT_DIR="results/bf16_comparison"

mkdir -p $OUTPUT_DIR

# Header
echo "Size,Precision,fwd_mean_ms,fwd_std_ms,bwd_mean_ms,bwd_std_ms" > $OUTPUT_DIR/comparison.csv

for SIZE in $SIZES; do
    echo "========================================"
    echo "Running $SIZE FP32..."
    echo "========================================"
    uv run python benchmark.py \
        --model_size $SIZE \
        --context_length $CTX \
        --warmup $WARMUP \
        --steps $STEPS \
        --mode $MODE \
        --dtype float32 \
        --mixed_precision 0 \
        --run_name ${SIZE}_fp32 \
        2>&1 | tee $OUTPUT_DIR/${SIZE}_fp32.txt

    echo "========================================"
    echo "Running $SIZE BF16..."
    echo "========================================"
    uv run python benchmark.py \
        --model_size $SIZE \
        --context_length $CTX \
        --warmup $WARMUP \
        --steps $STEPS \
        --mode $MODE \
        --dtype float32 \
        --mixed_precision 1 \
        --run_name ${SIZE}_bf16 \
        2>&1 | tee $OUTPUT_DIR/${SIZE}_bf16.txt
done

echo "========================================"
echo "Done! Results saved to $OUTPUT_DIR/"
echo "========================================"

# Print side by side summary
echo ""
echo "SUMMARY:"
echo "Size | Precision | fwd mean (ms) | bwd mean (ms)"
echo "-----|-----------|---------------|---------------"
for SIZE in $SIZES; do
    for PREC in fp32 bf16; do
        FWD=$(grep "fwd: mean=" $OUTPUT_DIR/${SIZE}_${PREC}.txt | tail -1 | grep -oP 'mean=\K[0-9.]+')
        BWD=$(grep "bwd: mean=" $OUTPUT_DIR/${SIZE}_${PREC}.txt | tail -1 | grep -oP 'mean=\K[0-9.]+')
        echo "$SIZE | $PREC | $FWD | $BWD"
    done
done