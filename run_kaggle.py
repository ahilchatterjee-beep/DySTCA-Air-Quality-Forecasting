#!/usr/bin/env python3
# ==========================================================
# run_kaggle.py
# DySTCA — Single Kaggle Entry Point
#
# Usage:
#   python run_kaggle.py --experiment baseline
#   python run_kaggle.py --experiment multiseed
#   python run_kaggle.py --experiment ablation
#   python run_kaggle.py --experiment greece
#   python run_kaggle.py --experiment benchmark
#   python run_kaggle.py --experiment all
#
# This script locates datasets automatically (from
# /kaggle/input/* or ./data/*) and runs experiments.
# ==========================================================

import os
import sys
import argparse
import subprocess
import time
import json
import platform

# ================================================================
# PATH RESOLUTION
# ================================================================

# The script is at: DySTCA_FINAL_KAGGLE/run_kaggle.py
# Project root = directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

# Source directories
DELHI_SRC = os.path.join(PROJECT_ROOT, "src", "delhi")
GREECE_SRC = os.path.join(PROJECT_ROOT, "src", "greece")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

os.makedirs(RESULTS_ROOT, exist_ok=True)

# ================================================================
# DATASET DISCOVERY
# ================================================================

def discover_delhi_data():
    """Find Delhi dataset. Returns path or None."""
    candidates = [
        os.path.join(DATA_ROOT, "delhi", "quality_controlled_sequences_fixed.pt"),
        os.path.join(DELHI_SRC, "data", "quality_controlled_sequences_fixed.pt"),
        os.path.join(DATA_ROOT, "quality_controlled_sequences_fixed.pt"),
    ]
    # Also search Kaggle input directories
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            for f in files:
                if f == "quality_controlled_sequences_fixed.pt":
                    candidates.insert(0, os.path.join(root, f))

    for path in candidates:
        if os.path.exists(path):
            print(f"[OK] Delhi data found: {path}")
            return path
    print("[MISSING] Delhi data not found.")
    return None


def discover_greece_data():
    """Find Greece dataset. Returns path or None."""
    candidates = [
        os.path.join(DATA_ROOT, "greece", "quality_controlled_sequences_fixed.pt"),
        os.path.join(GREECE_SRC, "data", "quality_controlled_sequences_fixed.pt"),
    ]
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                if "quality_controlled" in f and "greece" in root.lower():
                    candidates.insert(0, os.path.join(root, f))

    for path in candidates:
        if os.path.exists(path):
            print(f"[OK] Greece data found: {path}")
            return path
    print("[MISSING] Greece data not found.")
    return None


def discover_delhi_csv():
    """Find Delhi CSV. Returns path or None."""
    candidates = [
        os.path.join(DATA_ROOT, "delhi-weather-aqi-2025.csv"),
        os.path.join(DELHI_SRC, "delhi-weather-aqi-2025.csv"),
        os.path.join(DATA_ROOT, "delhi", "delhi-weather-aqi-2025.csv"),
    ]
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                if "delhi" in f.lower() and f.endswith(".csv"):
                    candidates.insert(0, os.path.join(root, f))

    for path in candidates:
        if os.path.exists(path):
            print(f"[OK] Delhi CSV found: {path}")
            return path
    return None


# ================================================================
# ENVIRONMENT REPORT
# ================================================================

def report_environment():
    import torch
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"Python      : {sys.version}")
    print(f"Platform    : {platform.platform()}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA        : {'Available - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 60)


# ================================================================
# EXPERIMENT RUNNERS
# ================================================================

def run_baseline(delhi_data_path):
    print("\n" + "=" * 60)
    print("RUNNING: Delhi Baseline Experiment")
    print("=" * 60)

    if delhi_data_path is None:
        print("NOT RUN -- DATASET UNAVAILABLE: Delhi data not found.")
        return False

    script = os.path.join(DELHI_SRC, "Training", "reproduce_baseline_run.py")
    if not os.path.exists(script):
        print(f"FAILED -- Script not found: {script}")
        return False

    env = os.environ.copy()
    env["PYTHONPATH"] = DELHI_SRC + os.pathsep + env.get("PYTHONPATH", "")

    # Create symlink or copy data to expected location
    expected_pt = os.path.join(DELHI_SRC, "data", "quality_controlled_sequences_fixed.pt")
    if not os.path.exists(expected_pt) and delhi_data_path != expected_pt:
        os.makedirs(os.path.dirname(expected_pt), exist_ok=True)
        try:
            os.symlink(delhi_data_path, expected_pt)
        except Exception:
            import shutil
            shutil.copy2(delhi_data_path, expected_pt)

    result = subprocess.run(
        [sys.executable, script],
        cwd=DELHI_SRC,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(f"FAILED:\n{result.stderr[-2000:]}")
        return False

    # Verify artifacts
    out_dir = os.path.join(DELHI_SRC, "Training", "results")
    results_dirs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))] if os.path.exists(out_dir) else []
    print(f"Result dirs: {results_dirs}")
    return True


def run_multiseed(delhi_data_path):
    print("\n" + "=" * 60)
    print("RUNNING: Multi-Seed Experiment (seeds: 42, 123, 2024, 3407, 9999)")
    print("=" * 60)

    if delhi_data_path is None:
        print("NOT RUN -- DATASET UNAVAILABLE: Delhi data not found.")
        return False

    script = os.path.join(DELHI_SRC, "Training", "multiseed_train.py")
    if not os.path.exists(script):
        print(f"FAILED -- Script not found: {script}")
        return False

    seeds = [42, 123, 2024, 3407, 9999]
    seed_results = {}

    env = os.environ.copy()
    env["PYTHONPATH"] = DELHI_SRC + os.pathsep + env.get("PYTHONPATH", "")

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        result = subprocess.run(
            [sys.executable, script, "--seed", str(seed)],
            cwd=DELHI_SRC,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        if result.returncode != 0:
            print(f"FAILED seed {seed}:\n{result.stderr[-1000:]}")
            seed_results[seed] = "FAILED"
        else:
            print(f"Seed {seed} completed.")
            print(result.stdout[-500:])
            seed_results[seed] = "COMPLETED"

    print("\nMulti-seed summary:", seed_results)
    return all(v == "COMPLETED" for v in seed_results.values())


def run_ablation(variant="all"):
    print("\n" + "=" * 60)
    print(f"RUNNING: Ablation Experiment (variant={variant})")
    print("=" * 60)

    script = os.path.join(PROJECT_ROOT, "run_ablation.py")
    if not os.path.exists(script):
        print(f"NOT RUN -- run_ablation.py not found at {script}")
        return False

    args = [sys.executable, script]
    if variant == "all":
        args.append("--all")
    else:
        args.extend(["--variant", variant])

    env = os.environ.copy()
    env["PYTHONPATH"] = DELHI_SRC + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(f"FAILED:\n{result.stderr[-2000:]}")
        return False
    return True


def run_greece(greece_data_path):
    print("\n" + "=" * 60)
    print("RUNNING: Greece Evaluation")
    print("=" * 60)

    if greece_data_path is None:
        print("NOT RUN -- DATASET UNAVAILABLE: Greece data not found.")
        return False

    script = os.path.join(PROJECT_ROOT, "run_greece.py")
    if not os.path.exists(script):
        print(f"NOT RUN -- run_greece.py not found at {script}")
        return False

    env = os.environ.copy()
    env["PYTHONPATH"] = GREECE_SRC + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(f"FAILED:\n{result.stderr[-2000:]}")
        return False
    return True


def run_benchmark(delhi_data_path):
    print("\n" + "=" * 60)
    print("RUNNING: Computational Benchmark")
    print("=" * 60)

    script = os.path.join(PROJECT_ROOT, "benchmark.py")
    if not os.path.exists(script):
        print(f"NOT RUN -- benchmark.py not found at {script}")
        return False

    env = os.environ.copy()
    env["PYTHONPATH"] = DELHI_SRC + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"FAILED:\n{result.stderr[-1000:]}")
        return False
    return True


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="DySTCA Kaggle Experiment Runner")
    parser.add_argument(
        "--experiment",
        choices=["baseline", "multiseed", "ablation", "greece", "benchmark", "all"],
        required=True,
        help="Which experiment to run"
    )
    parser.add_argument(
        "--variant",
        default="all",
        help="Ablation variant (only used when --experiment ablation)"
    )
    args = parser.parse_args()

    report_environment()

    delhi_data = discover_delhi_data()
    greece_data = discover_greece_data()

    experiment = args.experiment
    results = {}

    if experiment == "baseline" or experiment == "all":
        results["baseline"] = run_baseline(delhi_data)

    if experiment == "multiseed" or experiment == "all":
        results["multiseed"] = run_multiseed(delhi_data)

    if experiment == "ablation" or experiment == "all":
        results["ablation"] = run_ablation(args.variant if experiment == "ablation" else "all")

    if experiment == "greece" or experiment == "all":
        results["greece"] = run_greece(greece_data)

    if experiment == "benchmark" or experiment == "all":
        results["benchmark"] = run_benchmark(delhi_data)

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for exp, success in results.items():
        status = "COMPLETED" if success else "FAILED or NOT RUN"
        print(f"  {exp:<20}: {status}")
    print("=" * 60)

    # Save overall status
    summary_path = os.path.join(RESULTS_ROOT, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "results": {k: ("COMPLETED" if v else "FAILED") for k, v in results.items()}
        }, f, indent=4)
    print(f"\nSaved run summary to {summary_path}")


if __name__ == "__main__":
    main()
