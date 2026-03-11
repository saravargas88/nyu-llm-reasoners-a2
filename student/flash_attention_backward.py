
import torch 

@torch.compile

def flashbackward(Q, K, V, O, dO, L, is_causal=False):
    
    
    batch_size, n_queries, d = Q.shape
    b, n_keys, d_ = K.shape
    scale = 1.0 / math.sqrt(d)

    D = (dO * O).sum(dim=-1)  # (batch, n_queries)

    #Eq 13
    S = torch.einsum('bqd,bkd->bqk', Q.float(), K.float()) * scale

    P = torch.exp(S - L.unsqueeze(-1))  # (batch, n_queries, n_keys)
    
    #  causal mask if needed
    if is_causal:
        q_idx = torch.arange(n_queries, device=Q.device).unsqueeze(1)
        k_idx = torch.arange(n_keys, device=Q.device).unsqueeze(0)
        causal_mask = q_idx >= k_idx
        P = P * causal_mask.unsqueeze(0)

    
    dV = torch.einsum('bqk,bqd->bkd', P, dO.float())

    dP = torch.einsum('bqd,bkd->bqk', dO.float(), V.float())


    dQ = torch.einsum('bqk,bkd->bqd', dS, K.float()) * scale

    # Eq 19: dK = dS^T Q / sqrt(d)
    dK = torch.einsum('bqk,bqd->bkd', dS, Q.float()) * scale

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)
    

