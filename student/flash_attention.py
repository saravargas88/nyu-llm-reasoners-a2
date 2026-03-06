"""
flash_attention.py

Part (a): Pure PyTorch implementation of FlashAttention-2 forward pass.
No Triton — this is for debugging and understanding before writing the kernel.
"""
from student.flash_attention_backward import flash_backward
import torch
import math



class FlashAttentionPyTorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Q, K, V: (batch, seq_len, d)
        is_causal: if True, apply causal mask (ignored for part a, handled in part c)
        Returns: O (batch, seq_len, d)
        """
        B, N_q, d = Q.shape
        _,  N_k, _ = K.shape

        scale = 1.0 / math.sqrt(d)

        # Tile sizes — must be >= 16, use powers of 2
        Q_TILE = 32
        K_TILE = 32

        # Output and logsumexp buffers
        O = torch.zeros_like(Q)                        # (B, N_q, d)
        L = torch.full((B, N_q), float("-inf"),        # (B, N_q) — will hold logsumexp
                       device=Q.device, dtype=torch.float32)

        # Number of tiles
        T_q = math.ceil(N_q / Q_TILE)
        T_k = math.ceil(N_k / K_TILE)

        for i in range(T_q):
            # ── Query tile bounds ────────────────────────────────────────────
            q_start = i * Q_TILE
            q_end   = min(q_start + Q_TILE, N_q)

            # Load query tile: (B, Q_TILE, d)
            Qi = Q[:, q_start:q_end, :]

            # Running accumulators for this query tile
            bq = q_end - q_start
            Oi = torch.zeros(B, bq, d,  device=Q.device, dtype=torch.float32)
            li = torch.zeros(B, bq,     device=Q.device, dtype=torch.float32)
            mi = torch.full( (B, bq),   float("-inf"),
                             device=Q.device, dtype=torch.float32)

            for j in range(T_k):
                # ── Key/Value tile bounds ────────────────────────────────────
                k_start = j * K_TILE
                k_end   = min(k_start + K_TILE, N_k)

                Kj = K[:, k_start:k_end, :]   # (B, K_TILE, d)
                Vj = V[:, k_start:k_end, :]   # (B, K_TILE, d)

                # ── Attention scores for this tile ───────────────────────────
                # Sij: (B, bq, bk)
                Sij = torch.bmm(Qi.float(), Kj.float().transpose(-2, -1)) * scale

                # ── Causal mask ──────────────────────────────────────────────
                if is_causal:
                    # Global query and key indices
                    q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)  # (bq, 1)
                    k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)  # (1, bk)
                    mask  = q_idx < k_idx   # True where we should mask OUT
                    Sij   = Sij.masked_fill(mask.unsqueeze(0), -1e6)

                # ── Online softmax update ────────────────────────────────────
                # New row-wise max
                mij_new = torch.maximum(mi, Sij.max(dim=-1).values)  # (B, bq)

                # Unnormalized softmax values for this tile
                Pij = torch.exp(Sij - mij_new.unsqueeze(-1))          # (B, bq, bk)

                # Rescale old accumulator and update
                rescale = torch.exp(mi - mij_new)                     # (B, bq)
                Oi = rescale.unsqueeze(-1) * Oi + torch.bmm(Pij, Vj.float())
                li = rescale * li + Pij.sum(dim=-1)
                mi = mij_new

            # ── Normalize and write output tile ─────────────────────────────
            Oi = Oi / li.unsqueeze(-1)
            O[:, q_start:q_end, :] = Oi.to(Q.dtype)

            # Logsumexp: m + log(l)
            L[:, q_start:q_end] = mi + torch.log(li)

        # Save for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None  # None for is_causal gradient



# ── Convenience wrapper ───────────────────────────────────────────────────────

def flash_attention_pytorch(Q, K, V, is_causal=False):
    return FlashAttentionPyTorch.apply(Q, K, V, is_causal)


# ── Quick correctness check ───────────────────────────────────────────────────

def vanilla_attention(Q, K, V, is_causal=False):
    """Reference implementation using full N×N matrix."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.bmm(Q.float(), K.float().transpose(-2, -1)) * scale
    if is_causal:
        N = Q.shape[1]
        mask = torch.triu(torch.ones(N, N, device=Q.device), diagonal=1).bool()
        S = S.masked_fill(mask.unsqueeze(0), -1e6)
    P = torch.softmax(S, dim=-1)
    return torch.bmm(P, V.float()).to(Q.dtype)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, N, d = 2, 64, 32
    Q = torch.randn(B, N, d, device=device, dtype=torch.float32)
    K = torch.randn(B, N, d, device=device, dtype=torch.float32)
    V = torch.randn(B, N, d, device=device, dtype=torch.float32)

    # Non-causal
    out_flash   = flash_attention_pytorch(Q, K, V, is_causal=False)
    out_vanilla = vanilla_attention(Q, K, V, is_causal=False)
    print("Non-causal max diff:", (out_flash - out_vanilla).abs().max().item())

    # Causal
    out_flash_c   = flash_attention_pytorch(Q, K, V, is_causal=True)
    out_vanilla_c = vanilla_attention(Q, K, V, is_causal=True)
    print("Causal max diff:    ", (out_flash_c - out_vanilla_c).abs().max().item())