#!/usr/bin/env python3
import os
import sys
import time
import json
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DELHI_SRC = os.path.join(PROJECT_ROOT, "src", "delhi")
if DELHI_SRC not in sys.path:
    sys.path.insert(0, DELHI_SRC)

import config as c
from Model.new_model import SpatioTemporalModel

def run_benchmark():
    print("=" * 60)
    print("DySTCA COMPUTATIONAL BENCHMARK")
    print("=" * 60)
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Device: {device_name}")
    if device_name == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    model = SpatioTemporalModel(
        num_nodes=c.NUM_STATIONS,
        input_dim=c.INPUT_DIM,
        embedding_dim=c.EMBEDDING_DIM,
        sequence_length=c.SEQUENCE_LENGTH,
        prediction_horizon=c.PREDICTION_HORIZON,
        decay_constant=c.GRAPH_DECAY_CONSTANT,
        similarity_threshold=c.GRAPH_SIMILARITY_THRESHOLD,
        num_heads=c.GAT_HEADS,
        use_hyper_heuristic=False
    ).to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Calculate size in MB
    param_size = sum(p.element_size() * p.numel() for p in model.parameters())
    buffer_size = sum(b.element_size() * b.numel() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    print(f"Model Size: {size_mb:.2f} MB")
    
    # Dummy data
    B = c.BATCH_SIZE
    dummy_records = [[[{ "Station_ID": n, "Latitude": 0.0, "Longitude": 0.0, "wind_u": 0.0, "wind_v": 0.0 } for n in range(c.NUM_STATIONS)] for _ in range(c.SEQUENCE_LENGTH)] for _ in range(B)]
    x = torch.randn(B, c.SEQUENCE_LENGTH, c.NUM_STATIONS, c.INPUT_DIM).to(device)
    
    # Forward Pass Latency
    forward_times = []
    with torch.no_grad():
        for _ in range(2): # warmup
            model(x, dummy_records)
        for _ in range(5):
            if device_name == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            model(x, dummy_records)
            if device_name == "cuda": torch.cuda.synchronize()
            t1 = time.time()
            forward_times.append(t1 - t0)
            
    fwd_mean = np.mean(forward_times) * 1000
    
    # Backward Pass Latency
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    backward_times = []
    optimizer_times = []
    
    for _ in range(2): # warmup
        out = model(x, dummy_records)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    for _ in range(5):
        out = model(x, dummy_records)
        loss = out.sum()
        
        if device_name == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        loss.backward()
        if device_name == "cuda": torch.cuda.synchronize()
        t1 = time.time()
        optimizer.step()
        if device_name == "cuda": torch.cuda.synchronize()
        t2 = time.time()
        optimizer.zero_grad()
        
        backward_times.append(t1 - t0)
        optimizer_times.append(t2 - t1)
        
    bwd_mean = np.mean(backward_times) * 1000
    opt_mean = np.mean(optimizer_times) * 1000
    print(f"Forward Pass: {fwd_mean:.2f} ms")
    print(f"Backward Pass: {bwd_mean:.2f} ms")
    print(f"Optimizer Step: {opt_mean:.2f} ms")
    
    peak_mem = 0
    if device_name == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(0) / 1024**2
        print(f"Peak GPU Memory: {peak_mem:.2f} MB")
        
    out_dir = os.path.join(PROJECT_ROOT, "results", "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    
    results = {
        "parameters": param_count,
        "model_size_mb": size_mb,
        "forward_latency_ms": {"mean": float(fwd_mean)},
        "backward_latency_ms": {"mean": float(bwd_mean)},
        "optimizer_step_latency_ms": {"mean": float(opt_mean)},
        "gpu_memory_mb": float(peak_mem) if device_name == "cuda" else "NOT AVAILABLE",
        "device": device_name,
        "config": {
            "batch_size": B,
            "sequence_length": c.SEQUENCE_LENGTH,
            "stations": c.NUM_STATIONS,
            "features": c.INPUT_DIM
        }
    }
    
    with open(os.path.join(out_dir, "computational_benchmark.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    import pandas as pd
    pd.DataFrame([results]).to_csv(os.path.join(out_dir, "computational_benchmark.csv"), index=False)
        
    print(f"Saved benchmark results to {out_dir}")

if __name__ == "__main__":
    run_benchmark()
