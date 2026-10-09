import copy
import torch
import numpy as np

INPUT_PT = "data/quality_controlled_sequences.pt"
OUTPUT_PT = "data/quality_controlled_sequences_fixed.pt"

print("=" * 72)
print("REPAIRING STATION-RECORD ALIGNMENT")
print("=" * 72)

data = torch.load(INPUT_PT, map_location="cpu", weights_only=False)

required = [
    "X", "Y", "sequence_indices", "station_ids",
    "timestamps", "feature_columns", "station_record_sequences"
]
missing = [k for k in required if k not in data]
if missing:
    raise KeyError(f"Missing required keys: {missing}")

X = data["X"].detach().cpu()
sequence_indices = torch.as_tensor(data["sequence_indices"], dtype=torch.long)
station_ids = list(data["station_ids"])
timestamps = np.asarray(data["timestamps"])
feature_columns = list(data["feature_columns"])
old_records = data["station_record_sequences"]

B, T, N, F = X.shape

if F != len(feature_columns):
    raise ValueError(
        f"Feature mismatch: X has {F} features but feature_columns has "
        f"{len(feature_columns)} entries."
    )
if N != len(station_ids):
    raise ValueError(
        f"Station mismatch: X has {N} nodes but station_ids has "
        f"{len(station_ids)} entries."
    )
if len(old_records) != B:
    raise ValueError(
        f"Record mismatch: {len(old_records)} station-record sequences "
        f"for {B} X samples."
    )

# Static station metadata is safe to reuse by Station_ID.
# Dynamic values MUST come directly from X so that the spatial and temporal
# branches see the same sample/timestep/station.
metadata_by_id = {}

for sample in old_records:
    for timestep in sample:
        for record in timestep:
            sid = record["Station_ID"]
            if sid not in metadata_by_id:
                metadata_by_id[sid] = {
                    "Station_ID": sid,
                    "Station": record.get("Station", str(sid)),
                    "Latitude": float(record["Latitude"]),
                    "Longitude": float(record["Longitude"]),
                }

missing_metadata = [sid for sid in station_ids if sid not in metadata_by_id]
if missing_metadata:
    raise ValueError(
        f"Could not recover static metadata for station IDs: {missing_metadata}"
    )

fixed_records = []

for b in range(B):
    start = int(sequence_indices[b].item())
    sample_records = []

    for t in range(T):
        absolute_index = start + t
        timestamp = timestamps[absolute_index]
        timestep_records = []

        for n, sid in enumerate(station_ids):
            record = dict(metadata_by_id[sid])
            record["Timestamp"] = timestamp

            # Dynamic station features come from the EXACT X tensor location.
            for f, column in enumerate(feature_columns):
                record[column] = float(X[b, t, n, f].item())

            # Backward-compatible alias used by the current NodeFeatures code.
            # In this dataset Rainfall is the available rainfall feature.
            if "TOT-RF (mm)" not in record:
                record["TOT-RF (mm)"] = float(record.get("Rainfall", 0.0))

            timestep_records.append(record)

        sample_records.append(timestep_records)

    fixed_records.append(sample_records)

print(f"Samples repaired   : {len(fixed_records)}")
print(f"Timesteps/sample   : {T}")
print(f"Stations/timestep  : {N}")
print(f"Features aligned   : {F}")

# ------------------------------------------------------------
# HARD VALIDATION: station records must reproduce X exactly.
# ------------------------------------------------------------
max_abs_error = 0.0
mismatches = 0

for b in range(B):
    for t in range(T):
        for n, sid in enumerate(station_ids):
            record = fixed_records[b][t][n]

            if record["Station_ID"] != sid:
                raise AssertionError(
                    f"Station order mismatch at sample={b}, t={t}, n={n}"
                )

            expected_ts = timestamps[int(sequence_indices[b]) + t]
            if str(record["Timestamp"]) != str(expected_ts):
                raise AssertionError(
                    f"Timestamp mismatch at sample={b}, t={t}, n={n}"
                )

            for f, column in enumerate(feature_columns):
                xv = float(X[b, t, n, f].item())
                rv = float(record[column])
                error = abs(xv - rv)
                max_abs_error = max(max_abs_error, error)
                if error > 1e-6:
                    mismatches += 1

print("\nVALIDATION")
print("-" * 72)
print("Dynamic-value mismatches :", mismatches)
print("Maximum absolute error   :", max_abs_error)

if mismatches != 0:
    raise AssertionError(
        "Repair validation failed: station records still differ from X."
    )

# Show the first repaired point explicitly.
print("\nFIRST REPAIRED POINT")
print("-" * 72)
print("Sequence index :", int(sequence_indices[0]))
print("Timestamp      :", fixed_records[0][0][0]["Timestamp"])
print("Station ID     :", fixed_records[0][0][0]["Station_ID"])
print("X PM2.5        :", float(X[0, 0, 0, feature_columns.index("PM2.5 (µg/m³)")]))
print("Record PM2.5   :", fixed_records[0][0][0]["PM2.5 (µg/m³)"])
print("X Wind_U       :", float(X[0, 0, 0, feature_columns.index("Wind_U")]))
print("Record Wind_U  :", fixed_records[0][0][0]["Wind_U"])
print("X Wind_V       :", float(X[0, 0, 0, feature_columns.index("Wind_V")]))
print("Record Wind_V  :", fixed_records[0][0][0]["Wind_V"])

fixed_data = copy.copy(data)
fixed_data["station_record_sequences"] = fixed_records
fixed_data["station_record_alignment_fixed"] = True
fixed_data["station_record_alignment_source"] = (
    "Dynamic features reconstructed directly from X using sequence_indices; "
    "static station metadata retained by Station_ID."
)

torch.save(fixed_data, OUTPUT_PT)

print("\n" + "=" * 72)
print("REPAIR COMPLETE")
print("=" * 72)
print("Saved:", OUTPUT_PT)
print("\nNEXT STEP:")
print("Set QUALITY_DATASET_PATH in preprocessing/pre_updated.py to:")
print(f'    "{OUTPUT_PT}"')
print("=" * 72)
