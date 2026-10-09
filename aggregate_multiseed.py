import os
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MULTISEED_DIR = os.path.join(PROJECT_ROOT, "results", "multiseed")

def aggregate():
    if not os.path.exists(MULTISEED_DIR):
        print(f"Directory not found: {MULTISEED_DIR}")
        return
        
    seeds = [42, 123, 2024, 3407, 9999]
    metrics = {"mae": [], "rmse": [], "r2": [], "smape": []}
    
    valid_runs = 0
    for seed in seeds:
        summary_file = os.path.join(MULTISEED_DIR, f"seed_{seed}", "metrics_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                data = json.load(f)
                for k in metrics:
                    if k in data:
                        metrics[k].append(data[k])
            valid_runs += 1
            
    if valid_runs == 0:
        print("No multiseed results found to aggregate.")
        return
        
    agg = {}
    for k, v in metrics.items():
        if len(v) > 0:
            agg[f"{k}_mean"] = float(np.mean(v))
            agg[f"{k}_std"] = float(np.std(v))
            agg[f"{k}_min"] = float(np.min(v))
            agg[f"{k}_max"] = float(np.max(v))
            
    agg["valid_runs"] = valid_runs
    
    out_dir = os.path.join(MULTISEED_DIR, "aggregate")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "multiseed_summary.json"), "w") as f:
        json.dump(agg, f, indent=4)
        
    df = pd.DataFrame([agg])
    df.to_csv(os.path.join(out_dir, "multiseed_summary.csv"), index=False)
    print(f"Aggregated {valid_runs} runs successfully.")

if __name__ == "__main__":
    aggregate()
