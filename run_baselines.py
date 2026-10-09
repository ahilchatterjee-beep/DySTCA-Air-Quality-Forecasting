import os
import subprocess
import sys

def main():
    print("============================================================")
    print("RUNNING TABLE 4 COMPARATIVE BASELINES (SVR, GRU-ED, STGCN)")
    print("============================================================")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    delhi_src = os.path.join(script_dir, "src", "delhi", "Baselines")
    
    # 1. Run SVR
    print("\n--- Running SVR ---")
    svr_script = os.path.join(delhi_src, "train_svr.py")
    subprocess.run([sys.executable, svr_script])
    
    # 2. Run GRU-ED
    print("\n--- Running GRU-ED ---")
    dl_script = os.path.join(delhi_src, "train_dl_baseline.py")
    subprocess.run([sys.executable, dl_script, "--model", "GRU-ED"])
    
    # 3. Run STGCN
    print("\n--- Running STGCN ---")
    subprocess.run([sys.executable, dl_script, "--model", "STGCN"])
    
    print("\n============================================================")
    print("ALL BASELINES COMPLETED.")
    print("Results saved to results/baselines/")
    print("============================================================")

if __name__ == "__main__":
    main()
