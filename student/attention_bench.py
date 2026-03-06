"""
Section 1.2: Benchmarking PyTorch Attention
Benchmarks vanilla attention at various head dims and sequence lengths.
"""

import torch
import torch.nn.functional as F
import itertools
import math

# ── Attention implementation ──────────────────────────────────────────────────

def attention(Q, K, V, mask=None):
    """
    Vanilla scaled dot-product attention (no multihead, no head dimension).
    Q, K, V: (batch, seq_len, d_model)
    Returns: (batch, seq_len, d_model)
    """
    d_k = Q.shape[-1]
    scale = math.sqrt(d_k)

    # (batch, seq_len, seq_len)
    scores = torch.bmm(Q, K.transpose(-2, -1)) / scale

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    return torch.bmm(attn_weights, V)


# ── Benchmarking helpers ──────────────────────────────────────────────────────

def time_passes(fn, n=100):
    """Warm up then time n passes."""
    # Warmup
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


def measure_memory_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


# ── Main benchmark loop ───────────────────────────────────────────────────────

def run_benchmark():
    assert torch.cuda.is_available(), "CUDA required"

    BATCH        = 8
    N_PASSES     = 100
    d_models     = [16, 32, 64, 128]
    seq_lengths  = [256, 1024, 4096, 8192, 16384]
    device       = "cuda"
    dtype        = torch.float32

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {BATCH} | Passes: {N_PASSES}\n")

    header = (
        f"{'d_model':>8} {'seq_len':>8} "
        f"{'fwd_ms':>10} {'mem_MB':>10} {'bwd_ms':>10} {'status':>12}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for d_model, seq_len in itertools.product(d_models, seq_lengths):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # ── Forward pass timing ──────────────────────────────────────────
            Q = torch.randn(BATCH, seq_len, d_model, device=device, dtype=dtype, requires_grad=False)
            K = torch.randn(BATCH, seq_len, d_model, device=device, dtype=dtype, requires_grad=False)
            V = torch.randn(BATCH, seq_len, d_model, device=device, dtype=dtype, requires_grad=False)

            # Need grad for backward; recreate with requires_grad
            def make_inputs():
                q = torch.randn(BATCH, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
                k = torch.randn(BATCH, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
                v = torch.randn(BATCH, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
                return q, k, v

            Q, K, V = make_inputs()

            # Time forward
            def fwd():
                return attention(Q, K, V)

            fwd_ms = time_passes(fwd, N_PASSES)

            # ── Memory before backward ───────────────────────────────────────
            # Do one forward to build graph, measure memory
            Q2, K2, V2 = make_inputs()
            out = attention(Q2, K2, V2)
            torch.cuda.synchronize()
            mem_before_bwd_mb = measure_memory_mb()

            # ── Backward pass timing ─────────────────────────────────────────
            grad_out = torch.ones_like(out)

            def bwd():
                # We need a fresh graph each time
                nonlocal Q2, K2, V2, out
                Q2, K2, V2 = make_inputs()
                out = attention(Q2, K2, V2)
                out.backward(grad_out)

            bwd_ms = time_passes(bwd, N_PASSES)

            status = "OK"

        except torch.cuda.OutOfMemoryError:
            fwd_ms = float("nan")
            mem_before_bwd_mb = float("nan")
            bwd_ms = float("nan")
            status = "OOM"
            torch.cuda.empty_cache()

        results.append((d_model, seq_len, fwd_ms, mem_before_bwd_mb, bwd_ms, status))
        print(
            f"{d_model:>8} {seq_len:>8} "
            f"{fwd_ms:>10.3f} {mem_before_bwd_mb:>10.1f} {bwd_ms:>10.3f} {status:>12}"
        )

    return results


# ── Memory accounting helper ──────────────────────────────────────────────────

def memory_accounting(batch=8, seq_len=256, d_model=16, dtype_bytes=4):
    """
    Estimate memory for one attention forward pass (no multihead).
    Follows Assignment-1 Transformer memory equations.
    """
    B, N, d = batch, seq_len, d_model

    # Q, K, V stored for backward: 3 × B × N × d
    qkv_bytes    = 3 * B * N * d * dtype_bytes

    # Attention scores S = QK^T / sqrt(d): B × N × N
    scores_bytes  = B * N * N * dtype_bytes

    # Softmax output P (same shape as S)
    softmax_bytes = B * N * N * dtype_bytes

    total_bytes   = qkv_bytes + scores_bytes + softmax_bytes
    total_mb      = total_bytes / 1024**2

    print(f"\n--- Memory accounting for d_model={d_model}, seq_len={seq_len} ---")
    print(f"  Q/K/V (fwd activations): {qkv_bytes/1024**2:.2f} MB")
    print(f"  Attention scores S:      {scores_bytes/1024**2:.2f} MB")
    print(f"  Softmax output P:        {softmax_bytes/1024**2:.2f} MB")
    print(f"  Total estimated:         {total_mb:.2f} MB")
    print()
    print("  Memory scales as O(N^2) in seq_len for the attention matrices.")
    print("  Doubling seq_len quadruples memory for scores/softmax.")
    print("  To eliminate: use FlashAttention (tiled, never materialises N×N matrix).\n")


if __name__ == "__main__":
    results = run_benchmark()

    # Do accounting for a config near the OOM boundary
    # Adjust to whichever (d_model, seq_len) first hits OOM on your machine
    memory_accounting(batch=8, seq_len=8192, d_model=16)
    memory_accounting(batch=8, seq_len=4096, d_model=128)