# Dataset

This folder contains the raw dataset for the AHU Fault Detection project.

## Files

- **raw_data.csv**: Raw dataset containing AHU sensor readings and fault labels from hospital environments.

## Dataset Description

The dataset includes:
- Sensor readings from multiple Air Handling Units (AHUs)
- Temporal features (date, time)
- Temperature, fan, and valve position measurements
- Fault labels for 4 classes:
  - Class 0: Normal Condition
  - Class 1: Return Air Temperature Sensor Fault (RATSF)
  - Class 2: Supply Fan Fault (SFF)
  - Class 3: Valve Position Fault (VPF)

## Usage

1. Place `raw_data.csv` in the project root directory
2. Run `python data/preprocess.py` to preprocess the data
3. Run `python data/split.py` to create train/val/test splits
