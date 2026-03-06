"""
flash_attention_triton.py - FlashAttention-2 forward pass in Triton with causal masking.
"""
from student.flash_attention_backward import flash_backward
import torch
import math
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D:           tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal:   tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index      = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Qi = tl.load(Q_block_ptr)

    # running accumulators for online softmax
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,),   dtype=tl.float32)
    mi = tl.full( (Q_TILE_SIZE,),   float("-inf"), dtype=tl.float32)

    q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Kj = tl.load(K_block_ptr)
        Vj = tl.load(V_block_ptr)

        Sij = tl.dot(Qi.to(tl.float32), tl.trans(Kj).to(tl.float32)) * scale

        if is_causal:
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] < k_idx[None, :]
            Sij = Sij + tl.where(causal_mask, -1e6, 0.0)

        mij_new = tl.maximum(mi, tl.max(Sij, axis=1))
        Pij     = tl.exp(Sij - mij_new[:, None])
        rescale = tl.exp(mi - mij_new)

        Oi = rescale[:, None] * Oi
        Oi = tl.dot(Pij.to(Vj.dtype), Vj, acc=Oi).to(tl.float32)
        li = rescale * li + tl.sum(Pij, axis=1)
        mi = mij_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    Oi = Oi / li[:, None]
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, mi + tl.log(li))


class FlashAttentionTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_q, d = Q.shape
        _,  N_k, _ = K.shape

        assert Q.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        O = torch.empty_like(Q)
        L = torch.empty((B, N_q), device=Q.device, dtype=torch.float32)

        grid = (math.ceil(N_q / Q_TILE_SIZE), B)

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=N_q, N_KEYS=N_k,
            scale=1.0 / math.sqrt(d),
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None


def flash_attention_triton(Q, K, V, is_causal=False):
    return FlashAttentionTriton.apply(Q, K, V, is_causal)


def vanilla_attention(Q, K, V, is_causal=False):
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
    B, N, d = 2, 128, 64
    Q = torch.randn(B, N, d, device="cuda", dtype=torch.float32).contiguous()
    K = torch.randn(B, N, d, device="cuda", dtype=torch.float32).contiguous()
    V = torch.randn(B, N, d, device="cuda", dtype=torch.float32).contiguous()

    out_triton  = flash_attention_triton(Q, K, V, is_causal=False)
    out_vanilla = vanilla_attention(Q, K, V, is_causal=False)
    print("Non-causal max diff:", (out_triton - out_vanilla).abs().max().item())

    out_triton_c  = flash_attention_triton(Q, K, V, is_causal=True)
    out_vanilla_c = vanilla_attention(Q, K, V, is_causal=True)
    print("Causal max diff:    ", (out_triton_c - out_vanilla_c).abs)