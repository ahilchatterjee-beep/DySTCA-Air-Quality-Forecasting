import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_experiment(cmd):
    print(f"Starting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"Finished: {' '.join(cmd)} - {status}")
    if status == "FAIL":
        print(f"ERROR log for {' '.join(cmd)}:")
        print(result.stderr[-2000:])
    return status

if __name__ == "__main__":
    commands = [
        # Baseline
        [sys.executable, "src/delhi/Training/reproduce_baseline_run.py"],
        # Multiseed
        [sys.executable, "src/delhi/Training/multiseed_train.py", "--seed", "42"],
        [sys.executable, "src/delhi/Training/multiseed_train.py", "--seed", "123"],
        [sys.executable, "src/delhi/Training/multiseed_train.py", "--seed", "2024"],
        [sys.executable, "src/delhi/Training/multiseed_train.py", "--seed", "3407"],
        [sys.executable, "src/delhi/Training/multiseed_train.py", "--seed", "9999"],
        # Ablations
        [sys.executable, "run_ablation.py", "--variant", "full"],
        [sys.executable, "run_ablation.py", "--variant", "wo_tcn"],
        [sys.executable, "run_ablation.py", "--variant", "wo_gat"],
        [sys.executable, "run_ablation.py", "--variant", "wo_cross_attention"],
        [sys.executable, "run_ablation.py", "--variant", "wo_dynamic_graph"],
        [sys.executable, "run_ablation.py", "--variant", "autoregressive"],
        # Greece
        [sys.executable, "run_greece.py"]
    ]
    
    print("Running experiments in parallel (max 4 workers)...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_experiment, commands))
        
    print("\nAll experiments finished.")
