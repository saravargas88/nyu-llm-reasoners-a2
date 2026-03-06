"""
flash_attention_backward.py

FlashAttention-2 backward pass in pure PyTorch with torch.compile.
Follows Equations 13-19 from the assignment.
Importable by both FlashAttentionPyTorch and FlashAttentionTriton.
"""

import torch
import math


def _flash_backward_inner(Q, K, V, O, dO, L, is_causal):
    """
    Inputs:
        Q, K, V : (B, N, d)  — saved from forward
        O       : (B, N, d)  — saved from forward
        dO      : (B, N, d)  — gradient w.r.t. output
        L       : (B, N)     — logsumexp saved from forward
        is_causal: bool
    Returns:
        dQ, dK, dV each of shape (B, N, d)
    """
    B, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # ── Eq 13-14: recompute S and P from Q, K, L ────────────────────────────
    # S = QK^T / sqrt(d)
    S = torch.bmm(Q.float(), K.float().transpose(-2, -1)) * scale  # (B, N, N)

    if is_causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device), diagonal=1).bool()
        S = S.masked_fill(mask.unsqueeze(0), -1e6)

    # P = exp(S - L)  — L is (B, N), broadcast over key dim
    P = torch.exp(S - L.unsqueeze(-1).float())                      # (B, N, N)

    # ── Eq 15: dV = P^T dO ──────────────────────────────────────────────────
    dV = torch.bmm(P.transpose(-2, -1), dO.float())                 # (B, N, d)

    # ── Eq 16: dP = dO V^T ──────────────────────────────────────────────────
    dP = torch.bmm(dO.float(), V.float().transpose(-2, -1))         # (B, N, N)

    # ── D vector: D = rowsum(O * dO) ────────────────────────────────────────
    D = (O.float() * dO.float()).sum(dim=-1)                        # (B, N)

    # ── Eq 17: dS = P * (dP - D) ────────────────────────────────────────────
    dS = P * (dP - D.unsqueeze(-1))                                 # (B, N, N)

    if is_causal:
        dS = dS.masked_fill(mask.unsqueeze(0), 0.0)

    # ── Eq 18: dQ = dS K / sqrt(d) ──────────────────────────────────────────
    dQ = torch.bmm(dS, K.float()) * scale                           # (B, N, d)

    # ── Eq 19: dK = dS^T Q / sqrt(d) ────────────────────────────────────────
    dK = torch.bmm(dS.transpose(-2, -1), Q.float()) * scale         # (B, N, d)

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


# Compile for speed
flash_backward = torch.compile(_flash_backward_inner)