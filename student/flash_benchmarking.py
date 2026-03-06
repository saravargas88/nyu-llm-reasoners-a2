"""
flash_benchmarking.py - Benchmarks FlashAttention-2 (Triton) vs vanilla PyTorch attention.

Usage:
    uv run python flash_benchmarking.py --output results/flash_bench.tex
"""

import torch
import triton.testing
import math
import argparse
import itertools
import os

from flash_attention_triton import FlashAttentionTriton

SEQ_LENS   = [128, 1024, 8192, 65536]
D_SIZES    = [16, 128]
PRECISIONS = [torch.float32, torch.bfloat16]
BATCH_SIZE = 1
IS_CAUSAL  = True
DEVICE     = "cuda"
WARMUP     = 25
REPS       = 100


def vanilla_attn(Q, K, V, is_causal=True):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.bmm(Q, K.transpose(-2, -1)) * scale
    if is_causal:
        N = Q.shape[1]
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask.unsqueeze(0), float('-inf'))
    P = torch.softmax(S, dim=-1)
    return torch.bmm(P, V)

def flash_triton(Q, K, V, is_causal=True):
    return FlashAttentionTriton.apply(Q, K, V, is_causal)


def make_inputs(seq_len, d, dtype):
    def t():
        return torch.randn(BATCH_SIZE, seq_len, d, device=DEVICE, dtype=dtype).contiguous().requires_grad_(True)
    return t(), t(), t()


def bench_forward(fn, Q, K, V):
    Q = Q.detach().requires_grad_(False)
    K = K.detach().requires_grad_(False)
    V = V.detach().requires_grad_(False)
    return triton.testing.do_bench(lambda: fn(Q, K, V), warmup=WARMUP, rep=REPS)


def bench_backward(fn, Q, K, V):
    Q = Q.detach().requires_grad_(True)
    K = K.detach().requires_grad_(True)
    V = V.detach().requires_grad_(True)
    O  = fn(Q, K, V)
    dO = torch.randn_like(O)

    def bwd():
        if Q.grad is not None: Q.grad.zero_()
        if K.grad is not None: K.grad.zero_()
        if V.grad is not None: V.grad.zero_()
        O.backward(dO, retain_graph=True)

    return triton.testing.do_bench(bwd, warmup=WARMUP, rep=REPS)


def bench_fwd_bwd(fn, Q, K, V):
    Q  = Q.detach().requires_grad_(True)
    K  = K.detach().requires_grad_(True)
    V  = V.detach().requires_grad_(True)
    dO = torch.randn(BATCH_SIZE, Q.shape[1], Q.shape[2], device=DEVICE, dtype=Q.dtype)

    def run():
        if Q.grad is not None: Q.grad.zero_()
        if K.grad is not None: K.grad.zero_()
        if V.grad is not None: V.grad.zero_()
        O = fn(Q, K, V)
        O.backward(dO)

    return triton.testing.do_bench(run, warmup=WARMUP, rep=REPS)


def run_benchmarks():
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {BATCH_SIZE}, Causal: {IS_CAUSAL}\n")

    header = ["seq_len", "d", "dtype", "impl", "fwd (ms)", "bwd (ms)", "fwd+bwd (ms)"]
    rows   = []
    fmt    = "{:>8} {:>4} {:>9} {:>12}  {:>10} {:>10} {:>12}"

    print(fmt.format(*header))
    print("-" * 75)

    impls = [("triton", flash_triton), ("vanilla", vanilla_attn)]

    for seq_len, d, dtype in itertools.product(SEQ_LENS, D_SIZES, PRECISIONS):
        dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
        Q, K, V   = make_inputs(seq_len, d, dtype)

        for impl_name, fn in impls:
            if impl_name == "vanilla" and seq_len > 16384:
                row = [seq_len, d, dtype_str, impl_name, "OOM", "OOM", "OOM"]
                rows.append(row)
                print(fmt.format(*row))
                continue

            try:
                fwd_ms    = bench_forward(fn, Q, K, V)
                bwd_ms    = bench_backward(fn, Q, K, V)
                fwdbwd_ms = bench_fwd_bwd(fn, Q, K, V)
                row = [seq_len, d, dtype_str, impl_name,
                       f"{fwd_ms:.3f}", f"{bwd_ms:.3f}", f"{fwdbwd_ms:.3f}"]
                rows.append(row)
                print(fmt.format(*row))

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row = [seq_len, d, dtype_str, impl_name, "OOM", "OOM", "OOM"]
                rows.append(row)
                print(fmt.format(*row))

            except Exception as e:
                row = [seq_len, d, dtype_str, impl_name, "ERR", "ERR", "ERR"]
                rows.append(row)
                print(fmt.format(*row))
                print(f"  -> {e}")

    return rows


def save_latex(rows, output_path):
    gpu = torch.cuda.get_device_name(0)
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{FlashAttention-2 (Triton) vs vanilla PyTorch attention, batch=1, causal=True, GPU: {gpu}}}")
    lines.append(r"\label{tab:flash_bench}")
    lines.append(r"\begin{tabular}{rrrllll}")
    lines.append(r"\toprule")
    lines.append(r"seq\_len & $d$ & dtype & impl & fwd (ms) & bwd (ms) & fwd+bwd (ms) \\")
    lines.append(r"\midrule")

    prev_key = None
    for row in rows:
        seq_len, d, dtype, impl, fwd, bwd, fwdbwd = row
        cur_key = (seq_len, d, dtype)
        if prev_key is not None and cur_key != prev_key:
            lines.append(r"[3pt]")
        lines.append(f"{seq_len} & {d} & {dtype} & {impl} & {fwd} & {bwd} & {fwdbwd} \\\\")
        prev_key = cur_key

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved LaTeX table to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/flash_bench.tex")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    rows = run_benchmarks()
    save_latex(rows, args.output)