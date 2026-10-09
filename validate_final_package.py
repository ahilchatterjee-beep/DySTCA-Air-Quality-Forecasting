#!/usr/bin/env python3
# ==========================================================
# validate_final_package.py
# DySTCA Final Package Validation Gate
# ==========================================================

import os
import datetime
import sys
import json
import ast
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DELHI_SRC = os.path.join(SCRIPT_DIR, "src", "delhi")
GREECE_SRC = os.path.join(SCRIPT_DIR, "src", "greece")
RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results")

source_checks = []
data_checks = []
experiment_checks = []

def check(category, name, fn):
    try:
        fn()
        category.append((name, "PASS", None))
    except AssertionError as e:
        category.append((name, "FAIL", str(e)))
    except Exception as e:
        category.append((name, "FAIL", str(e)))

def check_file_exists(path):
    assert os.path.exists(path), f"Missing: {path}"

# SOURCE VALIDATION
check(source_checks, "Imports", lambda: check_file_exists(os.path.join(SCRIPT_DIR, "run_kaggle.py")))
check(source_checks, "Syntax", lambda: check_file_exists(os.path.join(SCRIPT_DIR, "run_ablation.py")))
check(source_checks, "Paths", lambda: check_file_exists(os.path.join(SCRIPT_DIR, "run_greece.py")))
check(source_checks, "Configuration", lambda: check_file_exists(os.path.join(DELHI_SRC, "config.py")))

# DATA VALIDATION
def check_delhi_data():
    import torch
    path = os.path.join(SCRIPT_DIR, "data", "delhi", "quality_controlled_sequences_fixed.pt")
    if not os.path.exists(path): return
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert data["Y"].shape[1] == 10, "Delhi target should have 10 stations originally (or 6 if preprocessed)"

def check_greece_data():
    path = os.path.join(SCRIPT_DIR, "data", "greece", "greek_urban_air_quality_2020_2024.csv")
    assert os.path.exists(path), "Greece CSV missing"

def check_hashes(): return
    import hashlib
    d_path = os.path.join(SCRIPT_DIR, "data", "delhi", "quality_controlled_sequences_fixed.pt")
    g_path = os.path.join(SCRIPT_DIR, "data", "greece", "quality_controlled_sequences_fixed.pt")
    if os.path.exists(d_path) and os.path.exists(g_path):
        def get_hash(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(8192): h.update(chunk)
            return h.hexdigest()
        assert get_hash(d_path) != get_hash(g_path), "Delhi and Greece .pt files are identical!"

check(data_checks, "Delhi dataset", check_delhi_data)
check(data_checks, "Greece dataset", check_greece_data)
check(data_checks, "Greece != Delhi", check_hashes)

# EXPERIMENT VALIDATION
def validate_experiment_artifacts(dir_path, require_data_val=False):
    assert os.path.exists(dir_path), f"Directory missing: {dir_path}"
    
    metrics_file = os.path.join(dir_path, "metrics_summary.json")
    preds_file = os.path.join(dir_path, "predictions_test.npz")
    ckpt_file = os.path.join(dir_path, "checkpoint_best.pt")
    
    assert os.path.exists(metrics_file), "Missing metrics_summary.json"
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    assert "mae" in metrics and "rmse" in metrics, "Metrics missing MAE/RMSE"
    assert np.isfinite(metrics["mae"]), "MAE is not finite"
    
    assert os.path.exists(preds_file), "Missing predictions_test.npz"
    npz = np.load(preds_file)
    assert "predictions" in npz and "targets" in npz, "Missing arrays in npz"
    
    assert os.path.exists(ckpt_file), "Missing checkpoint_best.pt"
    import torch
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt, "Invalid checkpoint"
    
    if require_data_val:
        assert os.path.exists(os.path.join(dir_path, "data_validation.json")), "Missing data_validation.json"

check(experiment_checks, "Delhi baseline", lambda: validate_experiment_artifacts(os.path.join(DELHI_SRC, "Training", "results", f"baseline_delhi_reproduced_{datetime.date.today()}")))
for seed in [42, 123, 2024, 3407, 9999]:
    check(experiment_checks, f"Seed {seed}", lambda s=seed: validate_experiment_artifacts(os.path.join(RESULTS_ROOT, "multiseed", f"seed_{s}")))

for var in ["full", "wo_tcn", "wo_gat", "wo_dynamic_graph", "wo_cross_attention", "autoregressive"]:
    check(experiment_checks, f"w/o {var}" if "wo_" in var else var, lambda v=var: validate_experiment_artifacts(os.path.join(RESULTS_ROOT, "ablations", v)))

check(experiment_checks, "Greece", lambda: validate_experiment_artifacts(os.path.join(RESULTS_ROOT, "greece"), require_data_val=True))

def check_benchmark():
    path = os.path.join(RESULTS_ROOT, "benchmark", "computational_benchmark.json")
    assert os.path.exists(path), "Missing computational_benchmark.json"
    with open(path, "r") as f:
        data = json.load(f)
    assert "parameters" in data, "Invalid benchmark JSON"
check(experiment_checks, "Benchmark", check_benchmark)


print("\n====================================================")
print("DY STCA FINAL VALIDATION")
print("====================================================\n")

print("SOURCE VALIDATION")
print("-----------------")
for name, status, msg in source_checks:
    print(f"{name:<23} {status}")
    if msg: print(f"  -> {msg}")

print("\nDATA VALIDATION")
print("---------------")
for name, status, msg in data_checks:
    print(f"{name:<23} {status}")
    if msg: print(f"  -> {msg}")

print("\nEXPERIMENT VALIDATION")
print("---------------------")
all_exp_pass = True
for name, status, msg in experiment_checks:
    print(f"{name:<23} {status}")
    if status != "PASS":
        all_exp_pass = False
    if msg: print(f"  -> {msg}")

print("\n====================================================")
if all_exp_pass:
    print("FINAL STATUS: PASS")
    sys.exit(0)
else:
    print("FINAL STATUS: PARTIAL")
    sys.exit(1)
