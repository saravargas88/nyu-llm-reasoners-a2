"""
FlashAttention-2 implementation in PyTorch.
"""

import math
import torch
from torch import Tensor


from student.flash_attention_backward import flashbackward

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        
        O, L = flashforward(ctx, Q, K, V, is_causal)
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        dQ, dK, dV = flashbackward(Q, K, V, O, dO, L, is_causal)
        
        return dQ, dK, dV, None


def flashforward(ctx, Q, K, V, is_causal: bool = False):
    Bq, Bk = 32, 32
    
    batch_size, n_queries, d = Q.shape
    bacth, n_keys, _ = K.shape

    scale = 1.0 / math.sqrt(d)
    
    qtiles = math.ceil(n_queries / Bq)
    ktiles = math.ceil(n_keys / Bk)

    O = torch.zeros_like(Q)
    
    L = torch.zeros(batch_size, n_queries, device=Q.device, dtype=Q.dtype)

    # over query tiles
    for i in range(qtiles):
        
        q_start = i * Bq
        q_end = min((i + 1) * Bq, n_queries)
        Qi = Q[:, q_start:q_end, :] 

        Oi = torch.zeros(batch_size, q_end - q_start, d, device=Q.device, dtype=torch.float32)
        li = torch.zeros(batch_size, q_end - q_start, device=Q.device, dtype=torch.float32)
        mi = torch.full((batch_size, q_end - q_start), float("-inf"), device=Q.device, dtype=torch.float32)
        
        #over key tiels
        for j in range(ktiles):
            k_start = j * Bk
            k_end = min((j + 1) * Bk, n_keys)
            Kj = K[:, k_start:k_end, :]  
            Vj = V[:, k_start:k_end, :] 

            
            Sij = torch.bmm(Qi.float(), Kj.float().transpose(-2, -1)) * scale

            
            if is_causal:
                
                q_indices = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                k_indices = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                
                causal_mask = q_indices >= k_indices  
                Sij = torch.where(
                    causal_mask.unsqueeze(0), Sij, torch.tensor(float("-inf"), device=Q.device,dtype=torch.float32)
                )
            
            mij = Sij.max(dim=-1).values 
            mi_new = torch.maximum(mi, mij)
            Pij_tilde = torch.exp(Sij - mi_new.unsqueeze(-1))  
            li_new = torch.exp(mi - mi_new) * li + Pij_tilde.sum(dim=-1)
            Oi = torch.exp(mi - mi_new).unsqueeze(-1) * Oi + torch.einsum('bik,bkd->bid', Pij_tilde, Vj.float())

            
            mi = mi_new
            li = li_new

        Oi = Oi / li.unsqueeze(-1)
        Li = mi + torch.log(li)

        
        O[:, q_start:q_end, :] = Oi.to(Q.dtype)
        L[:, q_start:q_end] = Li.to(Q.dtype)

    return O, L

