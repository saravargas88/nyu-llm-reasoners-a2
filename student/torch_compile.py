"""
Section 1.3: Benchmarking JIT-Compiled Attention with torch.compile
Extends the vanilla attention benchmark to include a compiled version.
"""

import torch
import torch.nn.functional as F
import itertools
import math


# ── Attention implementation ──────────────────────────────────────────────────

def attention(Q, K, V):
    """
    Vanilla scaled dot-product attention.
    Q, K, V: (batch, seq_len, d_model)
    """
    scale = math.sqrt(Q.shape[-1])
    scores = torch.bmm(Q, K.transpose(-2, -1)) / scale
    attn_weights = F.softmax(scores, dim=-1)
    return torch.bmm(attn_weights, V)


# Compiled version — torch.compile wraps the function once at module level
compiled_attention = torch.compile(attention)


# ── Benchmarking helpers ──────────────────────────────────────────────────────

def time_passes(fn, n=100):
    """Warm up 3 times, then time n passes with CUDA events."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n  # ms per call


def make_inputs(batch, seq_len, d_model, device, dtype):
    q = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    return q, k, v


def bench_one(attn_fn, batch, seq_len, d_model, device, dtype, n=100):
    """
    Returns (fwd_ms, bwd_ms) for a given attention function,
    or (nan, nan) on OOM.
    """
    try:
        Q, K, V = make_inputs(batch, seq_len, d_model, device, dtype)

        # Forward
        def fwd():
            return attn_fn(Q, K, V)
        fwd_ms = time_passes(fwd, n)

        # Backward — needs a fresh graph each call
        grad_out = torch.ones(batch, seq_len, d_model, device=device, dtype=dtype)

        def bwd():
            nonlocal Q, K, V
            Q, K, V = make_inputs(batch, seq_len, d_model, device, dtype)
            out = attn_fn(Q, K, V)
            out.backward(grad_out)

        bwd_ms = time_passes(bwd, n)
        return fwd_ms, bwd_ms

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan"), float("nan")


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark():
    assert torch.cuda.is_available(), "CUDA required"

    BATCH       = 8
    N_PASSES    = 100
    d_models    = [16, 32, 64, 128]
    seq_lengths = [256, 1024, 4096, 8192, 16384]
    device      = "cuda"
    dtype       = torch.float32

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch={BATCH} | Passes={N_PASSES}\n")

    # Trigger compilation once before the benchmark loop so compile time
    # is not included in the timed runs.
    print("Warming up torch.compile (this takes ~30s the first time)...")
    _q, _k, _v = make_inputs(BATCH, 256, 16, device, dtype)
    compiled_attention(_q, _k, _v)
    torch.cuda.synchronize()
    print("Done.\n")

    header = (
        f"{'d_model':>8} {'seq_len':>8} "
        f"{'fwd_vanilla':>13} {'fwd_compiled':>13} "
        f"{'bwd_vanilla':>13} {'bwd_compiled':>13}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for d_model, seq_len in itertools.product(d_models, seq_lengths):
        torch.cuda.empty_cache()

        v_fwd, v_bwd = bench_one(attention,          BATCH, seq_len, d_model, device, dtype, N_PASSES)
        c_fwd, c_bwd = bench_one(compiled_attention, BATCH, seq_len, d_model, device, dtype, N_PASSES)

        def fmt(x):
            return f"{x:>13.3f}" if not math.isnan(x) else f"{'OOM':>13}"

        print(
            f"{d_model:>8} {seq_len:>8} "
            f"{fmt(v_fwd)} {fmt(c_fwd)} "
            f"{fmt(v_bwd)} {fmt(c_bwd)}"
        )
        results.append((d_model, seq_len, v_fwd, c_fwd, v_bwd, c_bwd))

    return results


if __name__ == "__main__":
    run_benchmark()