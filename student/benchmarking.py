

import argparse
from os import sync
import os
import statistics
from contextlib import nullcontext
from timeit import default_timer as timer

from examples.fused_vs_unfused import benchmark
import numpy as np
import pandas as pd
import torch

from a1_basics.model import BasicsTransformerLM
from a1_basics.data import get_batch
from a1_basics.optimizer import AdamW

# End to end benchmarking of the forward and backward passes of the model 


#since we have so many args we use argparse: 
import argparse

MODEL_SIZES = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end Transformer benchmarking (A2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument("--model_size", default="small")
    p.add_argument("--d_model",    type=int, default=None)
    p.add_argument("--d_ff",       type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--num_heads",  type=int, default=None)
    # Run config
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--warmup",         type=int, default=5,
                   help="Un-timed warm-up steps before measurement")
    p.add_argument("--steps",          type=int, default=10,
                   help="Timed measurement steps")
    p.add_argument("--mode", default="forward_backward",
                   choices=["forward", "forward_backward"])
    p.add_argument("--device", default="cuda")
    # Precision
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"],
                   help="Base model / parameter dtype")
    p.add_argument("--mixed_precision", action="store_true",
                   help="Enable torch.autocast BF16 mixed precision (§1.1.5)")
    # Output
    p.add_argument("--latex_out", default=None,
                   help="Path to save LaTeX table, e.g. results.tex")
    return p.parse_args()


#Functions to convert results to a df and print as md table save latex
def results_to_df(rows):
    records = []
    for r in rows:
        records.append({
            "Size": r["size"],
            "Context len": r["context_length"],
            "Mode": r["mode"],
            "fwd mean (ms)": f"{r['fwd_mean_s']*1000:.2f} ± {r['fwd_std_s']*1000:.2f}",
            "bwd mean (ms)": f"{r['bwd_mean_s']*1000:.2f} ± {r['bwd_std_s']*1000:.2f}",
        })
    return pd.DataFrame(records)
        
    
def print_markdown_table(df: pd.DataFrame):
    print("\n" + df.to_markdown(index=False))

def save_latex(df: pd.DataFrame, path: str):
    df.to_latex(path, index=False,
                caption="Transformer benchmarking results",
                label="tab:benchmarks")
    print(f"  LaTeX table saved to {path}")
    



from timeit import default_timer as timer

def benchmark_script(
    model_cfg: dict, # take hyper parameters from the table 
    context_length: int = 512,
    warmup: int = 5,
    steps: int = 10,
    mode: str = "forward_backward",   
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    
) -> dict: 
    #generate random batch of data 

    model = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=context_length,
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        rope_theta=10_000,
    ).to(device=device, dtype=dtype)
    
    #so either forward pass alone (eval) or ALSO backward pass (train)
    model.train() if mode == "forward_backward" else model.eval()

    
    
    #parameters the model has : 
    nparams= sum(p.numel() for p in model.parameters())
    print(f"  Params  : {nparams / 1e6:.1f}M")

    
    #generate random batch of data 
    #Batch size: 4
    fake_dataset = np.random.randint(
        0, 10_000,
        size=(4 * context_length * 4,),
        dtype=np.int32,
    )
    batch, _ = get_batch(fake_dataset, 4, context_length, str(device))
    
    #for backward passes initialize the optimizer and loss function
    optim = AdamW(model.parameters(), lr=1e-4) if mode == "forward_backward" else None
    
    def warmup_step():
        if optim:
            optim.zero_grad(set_to_none=True)
        output= model(batch)
        
        if mode == "forward_backward":
            output.float().mean().backward()
            optim.step()
            
        torch.cuda.synchronize()
        

    def step(): 
        if optim:
            optim.zero_grad(set_to_none=True)

        # ── Forward ──
        start_time = timer()
        
        output = model(batch)
        
        torch.cuda.synchronize()
        
        
        for_end = timer()

        # ── Backward ──
        if mode == "forward_backward":
            loss = output.float().mean()
            loss.backward()
            optim.step()
            torch.cuda.synchronize()
        back_end = timer()

        forward_time = for_end - start_time
        backward_time = back_end - for_end if mode == "forward_backward" else 0.0
        
        return forward_time, backward_time
            
    #run w warm up steps and then time the execution of n steps 
    for w in range(warmup): 
        warmup_step()
        print(f"Warmup step {w+1}/{warmup} completed")
        
    forward_times = []    
    backward_times = []
    for s in range(steps): 
        fwd_time, bwd_time = step()
        forward_times.append(fwd_time)
        backward_times.append(bwd_time)
        
        if mode == "forward_backward":
            print(f"  Step {s+1:>2}/{steps}  fwd={fwd_time*1000:.2f} ms  "
                  f"bwd={bwd_time*1000:.2f} ms  total={(fwd_time+bwd_time)*1000:.2f} ms")
        else:
            print(f"  Step {s+1:>2}/{steps}  fwd={fwd_time*1000:.2f} ms")
    
    def _stats(times):
        return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)
    
    fwd_mean, fwd_std = _stats(forward_times)
    bwd_mean, bwd_std = _stats(backward_times)

    return dict(
        fwd_mean_s=fwd_mean,
        fwd_std_s=fwd_std,
        bwd_mean_s=bwd_mean,
        bwd_std_s=bwd_std,
        backward_times=backward_times,
        forward_times=forward_times,
    )

import os
def main(): 
    out_dir = f"results/{args.model_size}_ctx{args.context_length}_{args.mode}"
    os.makedirs(out_dir, exist_ok=True)

    args = parse_args()
    
    #go through model sizes run benchmark script save results 

    if args.model_size == "all":
        sizes = list(MODEL_SIZES.keys())
    elif args.model_size == "custom":
        for k in ["d_model", "d_ff", "num_layers", "num_heads"]:
            if getattr(args, k) is None:
                raise ValueError(f"--model_size custom requires --{k}")
        MODEL_SIZES["custom"] = dict(
            d_model=args.d_model, d_ff=args.d_ff,
            num_layers=args.num_layers, num_heads=args.num_heads,
        )
        sizes = ["custom"]
    else:
        sizes = [args.model_size]

    all_results = []
    
    for size_name in sizes:
        cfg = MODEL_SIZES[size_name]
        print(f"\n{'='*60}")
        print(f"  {size_name}  |  ctx={args.context_length}"
              f"  warmup={args.warmup}  steps={args.steps}  mode={args.mode}")
        print(f"{'='*60}")

        res = benchmark_script(
            model_cfg=cfg,
            context_length=args.context_length,
            warmup=args.warmup,
            steps=args.steps,
            mode=args.mode,
            device=args.device,
           
        )
        res.update(
            size=size_name,
            context_length=args.context_length,
            mode=args.mode,
            warmup=args.warmup,
            mixed_prec=args.mixed_precision,
        )
        all_results.append(res)
        print(f"\n  ▶  fwd: mean={res['fwd_mean_s']*1000:.2f} ms  std={res['fwd_std_s']*1000:.2f} ms")
        if args.mode == "forward_backward":
            print(f"  ▶  bwd: mean={res['bwd_mean_s']*1000:.2f} ms  std={res['bwd_std_s']*1000:.2f} ms")

    df = results_to_df(all_results)
    
    df.to_csv(f"{out_dir}/results.csv", index=False)
    df.to_markdown(f"{out_dir}/results.md", index=False)
    df.to_latex(f"{out_dir}/results.tex", index=False,
            caption="Benchmark results", label="tab:benchmarks")

    print(f"\nResults saved to {out_dir}/")

    if args.latex_out:
        save_latex(df, args.latex_out)


if __name__ == "__main__":
    main()

        
    
        
        