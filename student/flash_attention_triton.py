"""
FlashAttention-2 Triton kernel implementation.
"""

import math
import torch
import triton
import triton.language as tl
from triton import cdiv

from student.flash_attention_backward import flashbackward

@triton.jit
def flash_fwd_kernel(
     Q_ptr, K_ptr, V_ptr,
     O_ptr, L_ptr,
     stride_qb, stride_qq,
     stride_kb, stride_kk,
     stride_vb, stride_vk,
     stride_ob, stride_oq,
     stride_lb, stride_lq,
    stride_qd,
    stride_kd,
    stride_vd,
    stride_od,
     N_QUERIES, N_KEYS,
     scale,
     D: tl.constexpr,
     Q_TILE_SIZE: tl.constexpr,
     K_TILE_SIZE: tl.constexpr,

    is_causal: tl.constexpr
    
     ):

    
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
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

    
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)

    
    mi = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)  #runnign max
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  #running out
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  

    #casual mask query indices
    q_start = query_tile_index * Q_TILE_SIZE
    q_indices = q_start + tl.arange(0, Q_TILE_SIZE) 

    #loop over k v tiles
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_start = j * K_TILE_SIZE

        #k and v tiles load
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") 
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") 

        # attention scores
        Sij = tl.dot(Qi.to(tl.float32), tl.trans(Kj.to(tl.float32))) * scale

        
        if is_causal:
            k_indices = k_start + tl.arange(0, K_TILE_SIZE) 
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            Sij = tl.where(causal_mask, Sij, -1e6)

        # Online softmax update
        mij = tl.max(Sij, axis=1) 
        mi_new = tl.maximum(mi, mij)

        # p = exp(Sij - m_new)
        Pij_tilde = tl.exp(Sij - mi_new[:, None])  
        li_new = tl.exp(mi - mi_new) * li + tl.sum(Pij_tilde, axis=1)
        Oi = tl.exp(mi - mi_new)[:, None] * Oi + tl.dot(Pij_tilde.to(tl.float32), Vj.to(tl.float32))

        # Update
        mi = mi_new
        li = li_new

        #advance K and V block pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    
    Oi = Oi / li[:, None]

    # Compute logsumexp:
    Li = mi + tl.log(li)

    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li.to(L_block_ptr.type.element_ty), boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:

        if not Q.is_cuda:
            try:
                from student.flash_attention_pytorch import flash_attention_forward_tiled
            except ImportError:
                from flash_attention import flash_attention_forward_tiled

            O, L = flash_attention_forward_tiled(Q, K, V, is_causal=is_causal)
            ctx.save_for_backward(Q, K, V, O, L)
            ctx.is_causal = is_causal
            return O

        batch_size, n_queries, d = Q.shape
        _, n_keys, _ = K.shape

        
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        
        O = torch.empty_like(Q)
        L = torch.empty(batch_size, n_queries, device=Q.device, dtype=Q.dtype)

        
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        
        if n_queries < Q_TILE_SIZE:
            Q_TILE_SIZE = triton.next_power_of_2(n_queries)
        if n_keys < K_TILE_SIZE:
            K_TILE_SIZE = triton.next_power_of_2(n_keys)

        
        Q_TILE_SIZE = max(16, Q_TILE_SIZE)
        K_TILE_SIZE = max(16, K_TILE_SIZE)

        scale = 1.0 / math.sqrt(d)

        
        n_q_tiles = cdiv(n_queries, Q_TILE_SIZE)
        n_k_tiles = cdiv(n_keys, K_TILE_SIZE)

        #launch grid
        grid = (n_q_tiles, batch_size)

        # Launch kernel
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            stride_qb=Q.stride(0),
            stride_qq=Q.stride(1),
            stride_qd=Q.stride(2),
            stride_kb=K.stride(0),
            stride_kk=K.stride(1),
            stride_kd=K.stride(2),
            stride_vb=V.stride(0),
            stride_vk=V.stride(1),
            stride_vd=V.stride(2),
            stride_ob=O.stride(0),
            stride_oq=O.stride(1),
            stride_od=O.stride(2),
            stride_lb=L.stride(0),
            stride_lq=L.stride(1),
            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            scale=scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
            num_warps=4,
            num_stages=1,
        )
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        flashbackward
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        dQ, dK, dV = flashbackward(Q, K, V, O, dO, L, is_causal)

        return dQ, dK, dV, None
