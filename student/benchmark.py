#benchmark.py
import argparse
import os
import statistics
from contextlib import nullcontext
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch


from a1_basics.model import BasicsTransformerLM
from a1_basics.data import get_batch
from a1_basics.optimizer import AdamW


# End-to-end benchmarking of the forward and backward passes of the model
MODEL_SIZES = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end Transformer benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # model config
    p.add_argument("--model_size",  default="small",
                   help="One of: small, medium, large, xl, 2.7B, all, custom")
    p.add_argument("--d_model",     type=int, default=None)
    p.add_argument("--d_ff",        type=int, default=None)
    p.add_argument("--num_layers",  type=int, default=None)
    p.add_argument("--num_heads",   type=int, default=None)

    
    # specific run config
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--warmup",         type=int, default=5,
                   help="Number of un-timed warm-up steps (0 to disable)")
    p.add_argument("--steps",          type=int, default=10,
                   help="Number of timed measurement steps")
    p.add_argument("--mode",           default="forward_backward",
                   choices=["forward", "forward_backward"])
    p.add_argument("--device",         default="cuda")

    
   
    p.add_argument("--dtype",          default="float32",
                   choices=["float32", "float16", "bfloat16"])

    
    p.add_argument("--mixed_precision", type= int , default=0)
    p.add_argument("--memory_profiling",    type= int, default=0)
    p.add_argument("--memory_output", default= None)
    
    # Output
    p.add_argument("--run_name",    default=None,
                   help="Name for the results folder, e.g. 'q_b_warmup5'")
    p.add_argument("--latex_out",   default=None,
                   help="Optional extra path to save a LaTeX table")
    return p.parse_args()


#helpers for result saving and formatting
def results_to_df(rows: list[dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        records.append({
            "Size":         r["size"],
            "Params (M)":   f"{r['num_params_M']:.0f}",
            "Context len":  r["context_length"],
            "Warmup steps": r["warmup"],
            "Mode":         r["mode"],
            "fwd mean (ms)": f"{r['fwd_mean_s']*1000:.2f} ± {r['fwd_std_s']*1000:.2f}",
            "bwd mean (ms)": f"{r['bwd_mean_s']*1000:.2f} ± {r['bwd_std_s']*1000:.2f}",
        })
    return pd.DataFrame(records)


def save_results(df: pd.DataFrame, out_dir: str, latex_out: str = None):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/results.csv", index=False)
    
    df.to_latex(f"{out_dir}/results.tex", index=False,
                caption="Transformer benchmarking results",
                label="tab:benchmarks")
    print(f"\n  Results saved to {out_dir}/")
    if latex_out:
        df.to_latex(latex_out, index=False,
                    caption="Transformer benchmarking results",
                    label="tab:benchmarks")
        print(f"  LaTeX copy saved to {latex_out}")


# Actucal benchmarking code
def benchmark_script(
        model_cfg: dict,
        context_length: int = 512,
        warmup: int = 5,
        steps: int = 10,
        mode: str = "forward_backward",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        use_mixed_precision: int = 0, 
        memory_profiling: int =0, 
        memory_output: str = "memory_snapshots/", 
    ) -> dict:

    model = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=context_length,
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        rope_theta=10_000,
    ).to(device=device, dtype=dtype)

    model.train() if mode == "forward_backward" else model.eval()

    nparams = sum(p.numel() for p in model.parameters())
    print(f"  Params: {nparams / 1e6:.1f}M")

    fake_dataset = np.random.randint(
        0, 10_000, size=(4 * context_length * 4,), dtype=np.int32
    )
    batch, _ = get_batch(fake_dataset, 4, context_length, str(device))

    optim = AdamW(model.parameters(), lr=1e-4) if mode == "forward_backward" else None

    def run_warmup_step():
        if optim:
            optim.zero_grad(set_to_none=True)
        output = model(batch)
        if mode == "forward_backward":
            output.float().mean().backward()
            optim.step()
        torch.cuda.synchronize()

    def run_step():
        if optim:
            optim.zero_grad(set_to_none=True)

        #before running the passes choose if outcast or no 
        if use_mixed_precision:
            precision_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
          
        else:
            precision_ctx = nullcontext()
        

        #forwards
        start = timer()
        with precision_ctx:
            output = model(batch)
        torch.cuda.synchronize() #forward phase for memory 
        fwd_end = timer()

        # back
        if mode == "forward_backward":
            #separate the mem prof for optimizer and back ward setp
            output.float().mean().backward()
            torch.cuda.synchronize()  # backward boundary
            optim.step()
            torch.cuda.synchronize()  # optimizer boundary
            optim.zero_grad(set_to_none=True) 
            torch.cuda.synchronize()
        
        bwd_end = timer()

        fwd_time = fwd_end - start
        bwd_time = bwd_end - fwd_end if mode == "forward_backward" else 0.0
        return fwd_time, bwd_time


    # wamrup steps (not timed)
    for w in range(warmup):
        run_warmup_step()
        print(f"  Warmup step {w+1}/{warmup}")

    #since for memory profiling we only care after warmup reset peak mem stats 
    torch.cuda.reset_peak_memory_stats(device)

    if memory_profiling: 
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        
                    
        
    # timed steps (if there is mem profiling we do 1 only)
    num_steps = 1 if memory_profiling else steps
    forward_times, backward_times = [], []
    for s in range(steps):
        fwd, bwd = run_step()
        forward_times.append(fwd)
        backward_times.append(bwd)
        
        if mode == "forward_backward":
            print(f"  Step {s+1:>2}/{steps}  "
                  f"fwd={fwd*1000:.2f} ms  bwd={bwd*1000:.2f} ms  "
                  f"total={(fwd+bwd)*1000:.2f} ms")
        else: #only one
            print(f"  Step {s+1:>2}/{steps}  fwd={fwd*1000:.2f} ms")

    #dump memory prof pickl
    if(memory_profiling): 
        os.makedirs(os.path.dirname(memory_output), exist_ok = True)
        torch.cuda.memory._dump_snapshot(memory_output)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Snapshot saved in {memory_output}")

    
    def _stats(times):
        return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)

    fwd_mean, fwd_std = _stats(forward_times)
    bwd_mean, bwd_std = _stats(backward_times)

    
    

    return dict(
        num_params_M=nparams / 1e6,
        fwd_mean_s=fwd_mean,
        fwd_std_s=fwd_std,
        bwd_mean_s=bwd_mean,
        bwd_std_s=bwd_std,
        forward_times=forward_times,
        backward_times=backward_times,
    )




def main():
    args = parse_args()

    # Resolve model sizes to run
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

    # Output directory
    run_name = args.run_name or f"{args.model_size}_ctx{args.context_length}_{args.mode}_warmup{args.warmup}"
    out_dir  = f"results/{run_name}"

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    all_results = []

    for size_name in sizes:
        cfg = MODEL_SIZES[size_name]
        print(f"\n{'='*60}")
        print(f"  {size_name}  |  ctx={args.context_length}  "
              f"warmup={args.warmup}  steps={args.steps}  mode={args.mode}")
        print(f"{'='*60}")

        try: 

            snap_path = (
                f"memory_snapshots/"
                f"{size_name}_ctx{args.context_length}_"
                f"{args.mode}_"
                f"{'amp' if args.mixed_precision else args.dtype}"
                f".pickle"
            )
            
            res = benchmark_script(
                model_cfg=cfg,
                context_length=args.context_length,
                warmup=args.warmup,
                steps=args.steps,
                mode=args.mode,
                device=args.device,
                dtype=dtype,
                use_mixed_precision=args.mixed_precision, 
                memory_profiling = args.memory_profiling, 
                memory_output=snap_path
            )

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM — skipping {size_name}")
            torch.cuda.empty_cache()
            continue

    
        #res.update adds keys for each run (to keep metadata)
        res.update(
            size=size_name,
            context_length=args.context_length,
            mode=args.mode,
            warmup=args.warmup,
        )
        
        all_results.append(res)

        print(f"\n   fwd: mean={res['fwd_mean_s']*1000:.2f} ms  std={res['fwd_std_s']*1000:.2f} ms")
        if args.mode == "forward_backward":
            print(f"   bwd: mean={res['bwd_mean_s']*1000:.2f} ms  std={res['bwd_std_s']*1000:.2f} ms")

    #df = results_to_df(all_results)
    #print("\nResults:")
    #print(df.to_string(index=False))
    #save_results(df, out_dir, latex_out=args.latex_out)


if __name__ == "__main__":
    main()