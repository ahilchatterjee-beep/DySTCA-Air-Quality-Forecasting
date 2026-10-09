# DySTCA Data Validation Report

## 1. Overview
This report confirms that the DySTCA datasets correctly implement the temporal and spatial structures required for the multi-horizon forecasting experiments, specifically enforcing the raw structures of the original raw observations.

## 2. Delhi Dataset (delhi-weather-aqi-2025.csv)
- **Stations**: 6
- **Features**: 14 (meteorological + cyclic temporal embeddings)
- **Native Sampling Interval**: 1 hour
- **Lookback**: 168 hours (168 steps)
- **Forecast**: 24 hours (24 steps)

The processed `.pt` sequence generated matches:
`X: [samples, 168, 6, 14]`
`Y: [samples, 6, 24]`

## 3. Greece Dataset (greek_urban_air_quality_2020_2024.csv)
The Greek dataset was independently preprocessed to verify its properties and ensure the spatial/temporal structure is genuinely different from the Delhi topology.
- **Stations**: 31
- **Features**: 14
- **Native Sampling Interval**: 3 hours (Confirmed via strict `assert mode_diff_hours == 3.0`)
- **Lookback**: 72 hours (24 native steps of 3 hours each)
- **Forecast**: 24 hours (8 native steps of 3 hours each)

The processed sequence tensor shapes match:
`X: [samples, 24, 31, 14]`
`Y: [samples, 31, 8]`

**Key Note**: The data pipeline strictly preserves the native 3-hour recording interval rather than interpolating to 1-hour boundaries, ensuring real-world deployment realism for the Greek geographic topology.

## 4. Integrity Assertion
- The preprocessing hashes for the Delhi `quality_controlled_sequences_fixed.pt` and the Greece `quality_controlled_sequences_fixed.pt` were cryptographically compared and verified to be entirely distinct artifacts, solving the previous duplication bug.
- Validation scripts enforce `assert delhi_hash != greece_hash`.
