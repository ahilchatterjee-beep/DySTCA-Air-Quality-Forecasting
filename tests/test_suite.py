#!/usr/bin/env python3
# ==========================================================
# tests/test_suite.py
# DySTCA Automated Test Suite
#
# Usage:
#   python tests/test_suite.py
#
# Runs: import, shape, forward, backward, checkpoint,
#       evaluation, and reproducibility tests.
# ==========================================================

import os
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DELHI_SRC = os.path.join(PROJECT_ROOT, "src", "delhi")
GREECE_SRC = os.path.join(PROJECT_ROOT, "src", "greece")

sys.path.insert(0, DELHI_SRC)

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results = []


def test(name, fn):
    try:
        fn()
        results.append((name, "PASS", None))
        print(f"  {PASS} {name}")
    except Exception as e:
        tb = traceback.format_exc()
        results.append((name, "FAIL", str(e)))
        print(f"  {FAIL} {name}: {e}")
        print(f"       {tb.splitlines()[-1]}")


# ==========================================================
# TEST 1: Import tests
# ==========================================================

print("\n" + "=" * 60)
print("TEST GROUP 1: Import Tests")
print("=" * 60)


def test_import_torch():
    import torch
    assert torch.__version__


def test_import_numpy():
    import numpy as np
    assert np.__version__


def test_import_sklearn():
    from sklearn.preprocessing import StandardScaler
    StandardScaler()


def test_import_delhi_config():
    import config as c
    assert c.PREDICTION_HORIZON > 0
    assert c.NUM_STATIONS > 0
    assert c.SEQUENCE_LENGTH > 0


def test_import_delhi_model():
    from Model.new_model import SpatioTemporalModel
    assert SpatioTemporalModel is not None


def test_import_delhi_temporal():
    from Temporal.new_TE import TemporalEncoder
    assert TemporalEncoder is not None


def test_import_delhi_spatial():
    from Spatial.new_spatial_encoder import SpatialEncoder
    assert SpatialEncoder is not None


def test_import_delhi_fusion():
    from Fusion.fusion_layer import FusionLayer
    assert FusionLayer is not None


def test_import_delhi_prediction():
    from Prediction.prediction_layer import PredictionLayer
    assert PredictionLayer is not None


test("torch import", test_import_torch)
test("numpy import", test_import_numpy)
test("sklearn import", test_import_sklearn)
test("delhi config import", test_import_delhi_config)
test("delhi model import", test_import_delhi_model)
test("delhi temporal import", test_import_delhi_temporal)
test("delhi spatial import", test_import_delhi_spatial)
test("delhi fusion import", test_import_delhi_fusion)
test("delhi prediction import", test_import_delhi_prediction)


# ==========================================================
# TEST 2: Data Tests
# ==========================================================

print("\n" + "=" * 60)
print("TEST GROUP 2: Data Tests")
print("=" * 60)


def test_delhi_pt_exists():
    pt_path = os.path.join(DELHI_SRC, "data", "quality_controlled_sequences_fixed.pt")
    assert os.path.exists(pt_path), f"Delhi .pt file missing: {pt_path}"


def test_delhi_pt_shape():
    import torch
    pt_path = os.path.join(DELHI_SRC, "data", "quality_controlled_sequences_fixed.pt")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    X = data["X"]
    Y = data["Y"]
    assert X.ndim == 4, f"Expected 4D X, got {X.shape}"
    assert Y.ndim == 3, f"Expected 3D Y, got {Y.shape}"
    print(f"         Delhi X: {X.shape}, Y: {Y.shape}")


def test_greece_pt_exists():
    pt_path = os.path.join(GREECE_SRC, "data", "quality_controlled_sequences_fixed.pt")
    assert os.path.exists(pt_path), f"Greece .pt file missing: {pt_path}"


def test_greece_pt_shape():
    import torch
    pt_path = os.path.join(GREECE_SRC, "data", "quality_controlled_sequences_fixed.pt")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    X = data["X"]
    Y = data["Y"]
    assert X.ndim == 4, f"Expected 4D X, got {X.shape}"
    assert Y.ndim == 3, f"Expected 3D Y, got {Y.shape}"
    print(f"         Greece X: {X.shape}, Y: {Y.shape}")


test("delhi pt exists", test_delhi_pt_exists)
test("delhi pt shape", test_delhi_pt_shape)
test("greece pt exists", test_greece_pt_exists)
test("greece pt shape", test_greece_pt_shape)


# ==========================================================
# TEST 3: Model Tests
# ==========================================================

print("\n" + "=" * 60)
print("TEST GROUP 3: Model Shape Tests")
print("=" * 60)


def test_delhi_model_instantiation():
    import torch
    from Model.new_model import SpatioTemporalModel
    import config as c

    model = SpatioTemporalModel(
        num_nodes=c.NUM_STATIONS,
        input_dim=c.INPUT_DIM,
        embedding_dim=c.EMBEDDING_DIM,
        sequence_length=c.SEQUENCE_LENGTH,
        prediction_horizon=c.PREDICTION_HORIZON,
        decay_constant=0.5,
        similarity_threshold=0.2,
        num_heads=c.GAT_HEADS,
        use_hyper_heuristic=False
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"         Parameters: {param_count:,}")
    assert param_count > 0


def test_delhi_forward_pass():
    import torch
    from Model.new_model import SpatioTemporalModel
    import config as c

    model = SpatioTemporalModel(
        num_nodes=c.NUM_STATIONS,
        input_dim=c.INPUT_DIM,
        embedding_dim=c.EMBEDDING_DIM,
        sequence_length=c.SEQUENCE_LENGTH,
        prediction_horizon=c.PREDICTION_HORIZON,
        decay_constant=0.5,
        similarity_threshold=0.2,
        num_heads=c.GAT_HEADS,
        use_hyper_heuristic=False
    )
    model.eval()

    # Create dummy station records
    B = 2
    dummy_records = []
    for b in range(B):
        seq = []
        for t in range(c.SEQUENCE_LENGTH):
            step = []
            for n in range(c.NUM_STATIONS):
                step.append({
                    "Station_ID": n,
                    "Latitude": 28.6 + n * 0.1,
                    "Longitude": 77.0 + n * 0.1,
                    "wind_u": 0.0,
                    "wind_v": 0.0
                })
            seq.append(step)
        dummy_records.append(seq)

    x = torch.zeros(B, c.SEQUENCE_LENGTH, c.NUM_STATIONS, c.INPUT_DIM)
    with torch.no_grad():
        out = model(x, dummy_records)

    assert out.shape == (B, c.NUM_STATIONS, c.PREDICTION_HORIZON), \
        f"Expected ({B}, {c.NUM_STATIONS}, {c.PREDICTION_HORIZON}), got {out.shape}"
    print(f"         Output shape: {out.shape} [CORRECT]")


def test_delhi_backward_pass():
    import torch
    from Model.new_model import SpatioTemporalModel
    import config as c

    model = SpatioTemporalModel(
        num_nodes=c.NUM_STATIONS,
        input_dim=c.INPUT_DIM,
        embedding_dim=c.EMBEDDING_DIM,
        sequence_length=c.SEQUENCE_LENGTH,
        prediction_horizon=c.PREDICTION_HORIZON,
        decay_constant=0.5,
        similarity_threshold=0.2,
        num_heads=c.GAT_HEADS,
        use_hyper_heuristic=False
    )
    model.train()

    B = 2
    dummy_records = []
    for b in range(B):
        seq = []
        for t in range(c.SEQUENCE_LENGTH):
            step = []
            for n in range(c.NUM_STATIONS):
                step.append({
                    "Station_ID": n,
                    "Latitude": 28.6 + n * 0.1,
                    "Longitude": 77.0 + n * 0.1,
                    "wind_u": 0.0,
                    "wind_v": 0.0
                })
            seq.append(step)
        dummy_records.append(seq)

    x = torch.zeros(B, c.SEQUENCE_LENGTH, c.NUM_STATIONS, c.INPUT_DIM)
    y = torch.zeros(B, c.NUM_STATIONS, c.PREDICTION_HORIZON)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    out = model(x, dummy_records)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()
    print(f"         Loss: {loss.item():.6f}")
    assert loss.item() >= 0.0


def test_checkpoint_save_load():
    import torch
    import tempfile
    from Model.new_model import SpatioTemporalModel
    import config as c

    model = SpatioTemporalModel(
        num_nodes=c.NUM_STATIONS,
        input_dim=c.INPUT_DIM,
        embedding_dim=c.EMBEDDING_DIM,
        sequence_length=c.SEQUENCE_LENGTH,
        prediction_horizon=c.PREDICTION_HORIZON,
        decay_constant=0.5,
        similarity_threshold=0.2,
        num_heads=c.GAT_HEADS,
        use_hyper_heuristic=False
    )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    try:
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print("         Checkpoint save/load: OK")
    finally:
        os.unlink(ckpt_path)


test("delhi model instantiation", test_delhi_model_instantiation)
test("delhi forward pass", test_delhi_forward_pass)
test("delhi backward pass", test_delhi_backward_pass)
test("checkpoint save/load", test_checkpoint_save_load)


# ==========================================================
# TEST 4: Metric Tests
# ==========================================================

print("\n" + "=" * 60)
print("TEST GROUP 4: Metric Calculation Tests")
print("=" * 60)


def test_metric_calculation():
    import numpy as np
    preds = np.array([10.0, 20.0, 30.0, 40.0])
    targets = np.array([12.0, 18.0, 33.0, 38.0])

    err = preds - targets
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)

    assert 0 <= mae < 100, f"Unexpected MAE: {mae}"
    assert 0 <= rmse < 100, f"Unexpected RMSE: {rmse}"
    assert -10 <= r2 <= 1.0, f"Unexpected R2: {r2}"
    print(f"         MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")


test("metric calculation", test_metric_calculation)


# ==========================================================
# TEST 5: Reproducibility
# ==========================================================

print("\n" + "=" * 60)
print("TEST GROUP 5: Reproducibility Test")
print("=" * 60)


def test_seed_reproducibility():
    import torch
    import numpy as np
    import random

    def get_output(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        x = torch.randn(3, 5)
        return x.sum().item()

    out1 = get_output(42)
    out2 = get_output(42)
    out3 = get_output(123)

    assert abs(out1 - out2) < 1e-10, "Same seed should give same output"
    assert abs(out1 - out3) > 1e-6, "Different seeds should give different outputs"
    print(f"         Seed 42 run1={out1:.6f}, run2={out2:.6f} [identical]")
    print(f"         Seed 123 run={out3:.6f} [different]")


test("seed reproducibility", test_seed_reproducibility)


# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

passed = sum(1 for _, status, _ in results if status == "PASS")
failed = sum(1 for _, status, _ in results if status == "FAIL")
total = len(results)

for name, status, msg in results:
    mark = PASS if status == "PASS" else FAIL
    print(f"  {mark} {name}")

print("-" * 60)
print(f"  Total: {total} | Passed: {passed} | Failed: {failed}")
print("=" * 60)

if failed == 0:
    print("  OVERALL: PASS")
    sys.exit(0)
else:
    print("  OVERALL: FAIL")
    sys.exit(1)
