"""
flash_attention_backward.py
FlashAttention-2 backward pass in pure PyTorch with torch.compile.
"""

import torch
import math


def _flash_backward_inner(Q, K, V, O, dO, L, is_causal):
    B, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # S = QK^T / sqrt(d)
    S = torch.bmm(Q.float(), K.float().transpose(-2, -1)) * scale  # (B, N, N)

    if is_causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device), diagonal=1).bool()
        S = S.masked_fill(mask.unsqueeze(0), -1e6)

  
    P = torch.exp(S - L.unsqueeze(-1).float())                     
    dV = torch.bmm(P.transpose(-2, -1), dO.float())                
    dP = torch.bmm(dO.float(), V.float().transpose(-2, -1))         

    
    D = (O.float() * dO.float()).sum(dim=-1)                       

    
    dS = P * (dP - D.unsqueeze(-1))                                

    if is_causal:
        dS = dS.masked_fill(mask.unsqueeze(0), 0.0)

   
    dQ = torch.bmm(dS, K.float()) * scale                          

    
    dK = torch.bmm(dS.transpose(-2, -1), Q.float()) * scale         

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)

flash_backward = torch.compile(_flash_backward_inner)