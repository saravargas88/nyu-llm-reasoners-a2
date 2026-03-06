# profiling.py
import argparse
import math
import torch
import torch.cuda.nvtx as nvtx
from einops import einsum
import a1_basics.model
from a1_basics.nn_utils import softmax
from benchmark import benchmark_script, MODEL_SIZES

# --- Annotated attention ---
@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(K, Q, V, mask=None):
    with nvtx.range("computing attention scores"):
        d_k = K.shape[-1]
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")
    return output

a1_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

# --- Patch benchmark_script to add NVTX ranges around warmup vs timed steps ---
import benchmark as _bm
_orig = _bm.benchmark_script

def patched_benchmark_script(**kwargs):
    # We need to wrap run_step internals, but since they're closures we
    # patch at the model level instead — wrap the model's forward
    result = _orig(**kwargs)
    return result

# Patch the model forward to label forward/backward in NVTX
_orig_forward = a1_basics.model.BasicsTransformerLM.forward

def labeled_forward(self, x):
    with nvtx.range("forward pass"):
        return _orig_forward(self, x)

a1_basics.model.BasicsTransformerLM.forward = labeled_forward

# --- Args ---
p = argparse.ArgumentParser()
p.add_argument("--model_size", default="small")
p.add_argument("--context_length", type=int, default=128)
p.add_argument("--mode", default="forward", choices=["forward", "forward_backward"])
args = p.parse_args()

benchmark_script(
    model_cfg=MODEL_SIZES[args.model_size],
    context_length=args.context_length,
    warmup=2,
    steps=3,
    mode=args.mode,
    device="cuda",
    dtype=torch.float32,
)