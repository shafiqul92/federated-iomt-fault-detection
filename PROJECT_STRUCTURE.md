# Project Structure Documentation

## Overview

This document provides a detailed breakdown of the project structure and organization.

## Directory Structure

```
fault-detection-iomt/
│
├── config/                          # Configuration Management
│   ├── __init__.py                  # Config loader utilities
│   └── config.yaml                  # Main configuration file
│
├── data/                            # Data Processing
│   ├── preprocess.py                # Data preprocessing and cleaning
│   └── split.py                     # Train/val/test splitting with SMOTE
│
├── models/                          # Model Architectures
│   ├── __init__.py                  # Model package exports
│   └── hybrid_model.py              # Hybrid LSTM+CNN+Attention model
│
├── training/                        # Training Scripts
│   ├── train_centralized.py         # Centralized baseline training
│   ├── train_fedavg.py              # Federated Averaging training
│   └── train_edge_aware.py          # Edge-aware federated learning
│
├── evaluation/                      # Evaluation & Visualization
│   └── stream.py                    # Real-time streaming comparison
│
├── utils/                           # Utility Functions
│   └── __init__.py                  # Helper functions (model finding, etc.)
│
├── processed_splits_advanced/       # Processed Data (Generated)
│   ├── centralized/                 # Centralized training splits
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── federated/                   # Federated learning splits
│   │   ├── ahu_1/
│   │   ├── ahu_2/
│   │   └── ...
│   └── metadata.json                # Dataset metadata
│
├── plots/                           # Generated Visualizations (Generated)
│   ├── checkpoints/                 # Checkpoint plots
│   ├── 01_complete_streaming_analysis.png
│   ├── 02_confusion_matrices.png
│   └── ...
│
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── README.md                        # Main documentation
└── PROJECT_STRUCTURE.md             # This file
```

## Module Descriptions

### config/
Contains all configuration files and utilities.

- **config.yaml**: Central configuration file with all hyperparameters, paths, and settings
- **__init__.py**: Provides `load_config()` function to load YAML configurations

### data/
Data preprocessing and preparation scripts.

- **preprocess.py**: 
  - Loads raw CSV data
  - Handles missing values (seasonal NaN patterns)
  - Feature engineering (temporal features, one-hot encoding)
  - Normalizes column names
  - Outputs `preprocessed_data.csv`

- **split.py**:
  - Creates stratified train/val/test splits
  - Per-AHU splits for federated learning
  - SMOTE augmentation for imbalanced classes
  - Augments rare classes across AHUs
  - Handles edge cases (single-class AHUs, insufficient data)

### models/
Neural network model definitions.

- **hybrid_model.py**:
  - `HybridLSTMCNNAttention`: Main model class
  - `AttentionLayer`: Multi-head self-attention
  - `ResidualBlock`: Residual connection with layer norm
  - `create_model()`: Factory function for model creation

### training/
Training scripts for different learning paradigms.

- **train_centralized.py**:
  - Traditional centralized training
  - Uses all data together
  - Best for baseline comparison

- **train_fedavg.py**:
  - Federated Averaging algorithm
  - Distributes training across AHU clients
  - Aggregates model updates via weighted averaging

- **train_edge_aware.py**:
  - Resource-aware federated learning
  - Considers device constraints (CPU, battery, bandwidth, memory)
  - Adaptive training strategies
  - Offloading decisions
  - Comprehensive ablation study support

### evaluation/
Evaluation and comparison scripts.

- **stream.py**:
  - Real-time prediction streaming
  - Compares all three model types
  - Generates comprehensive visualizations
  - Creates checkpoint plots during processing
  - Auto-detects latest model files

### utils/
Common utility functions.

- **__init__.py**:
  - `find_latest_model()`: Find latest model checkpoint
  - `find_all_model_patterns()`: Find all model types
  - `get_timestamp_string()`: Generate timestamp strings

## Data Flow

```
raw_data.csv
    ↓
[preprocess.py]
    ↓
preprocessed_data.csv
    ↓
[split.py]
    ↓
processed_splits_advanced/
    ├── centralized/ (train, val, test)
    └── federated/ (per-AHU splits)
    ↓
[Training Scripts]
    ↓
*.pth model files
    ↓
[evaluation/stream.py]
    ↓
plots/ (visualizations)
```

## File Naming Conventions

### Models
- Centralized: `best_model_centralized_baseline.pth`
- FedAvg: `model_fedavg_YYYYMMDD_HHMMSS.pth`
- Edge-Aware: `model_edge_aware_<config>_YYYYMMDD_HHMMSS.pth`

### Results
- FedAvg: `results_fedavg_YYYYMMDD_HHMMSS.json`
- Edge-Aware: `results_edge_aware_<config>_YYYYMMDD_HHMMSS.json`

### Data
- Raw: `raw_data.csv`
- Preprocessed: `preprocessed_data.csv`
- Splits: `processed_splits_advanced/{centralized|federated}/{split}.csv`

## Configuration System

All configurations are centralized in `config/config.yaml`:

- **Model settings**: Architecture parameters, dropout, etc.
- **Training settings**: Batch size, learning rate, epochs
- **Federated settings**: Rounds, local epochs
- **Edge-aware settings**: Device specs, thresholds, strategies
- **Paths**: Data paths (can be overridden in code)

## Import Paths

After reorganization, import paths have changed:

**Before:**
```python
from train_centralized_baseline import HybridLSTMCNNAttention
```

**After:**
```python
import sys
sys.path.append('.')
from models.hybrid_model import HybridLSTMCNNAttention
# Or
from models import HybridLSTMCNNAttention
```

## Generated Files (Gitignored)

These files are generated during execution and should not be committed:

- `*.pth` - Model checkpoints
- `*.json` - Result files
- `plots/` - Visualization outputs
- `processed_splits_advanced/` - Processed data
- `preprocessed_data.csv` - Preprocessed data
- `*.log` - Log files

## Migration Notes

When updating imports in existing scripts:

1. Model imports:
   ```python
   # Old
   from train_centralized_baseline import HybridLSTMCNNAttention
   
   # New
   from models.hybrid_model import HybridLSTMCNNAttention
   ```

2. Configuration:
   ```python
   # New approach
   from config import load_config
   config = load_config()
   ```

3. Utilities:
   ```python
   from utils import find_latest_model
   model_path = find_latest_model('model_fedavg_*.pth')
   ```

## Future Improvements

Potential enhancements to the structure:

1. **Tests**: Add `tests/` directory with unit tests
2. **Scripts**: Add `scripts/` for one-off utilities
3. **Notebooks**: Add `notebooks/` for exploratory analysis
4. **Logging**: Add structured logging configuration
5. **Documentation**: Generate API docs with Sphinx

