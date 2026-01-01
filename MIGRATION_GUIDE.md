# Migration Guide: Reorganized Project Structure

## What Changed?

The project has been reorganized into a more maintainable structure:

### Before
```
fault-detection-iomt/
├── preprocess.py
├── split.py
├── train_centralized_baseline.py
├── train_fedavg.py
├── train_edge_aware_fl.py
├── stream.py
└── [model code duplicated in each file]
```

### After
```
fault-detection-iomt/
├── config/           # Configuration files
├── data/             # Data processing scripts
├── models/           # Shared model code
├── training/         # Training scripts
├── evaluation/       # Evaluation scripts
└── utils/            # Utility functions
```

## Required Changes to Scripts

### 1. Update Model Imports

**Old way (model code duplicated):**
```python
# Model was defined in each training file
class HybridLSTMCNNAttention(nn.Module):
    ...
```

**New way (shared model):**
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_model import HybridLSTMCNNAttention
# Or use the factory function
from models import create_model
```

### 2. Update Data Paths

**Old:**
```python
DATA_FILE = 'preprocessed_data.csv'
SPLITS_DIR = 'processed_splits_advanced'
```

**New:**
```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / 'preprocessed_data.csv'
SPLITS_DIR = PROJECT_ROOT / 'processed_splits_advanced'
```

### 3. Update Script Execution Paths

Since scripts are now in subdirectories, update relative paths:

**Old:**
```python
train_df = pd.read_csv('processed_splits_advanced/centralized/train.csv')
```

**New:**
```python
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
train_df = pd.read_csv(BASE_DIR / 'processed_splits_advanced/centralized/train.csv')
```

## Quick Fix Script

I've created a helper script `setup_paths.py` that adds the project root to Python path. You can use it like this:

```python
# At the top of your script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now you can import normally
from models import HybridLSTMCNNAttention
```

## Running Scripts

### From Project Root

All scripts should be run from the project root directory:

```bash
# From project root
python data/preprocess.py
python data/split.py
python training/train_centralized.py
python training/train_fedavg.py
python training/train_edge_aware.py
python evaluation/stream.py
```

### Or Use Python Module Syntax

```bash
python -m data.preprocess
python -m data.split
python -m training.train_centralized
# etc.
```

## Model File Auto-Detection

The `evaluation/stream.py` script now uses auto-detection for model files:

**Old (hardcoded):**
```python
FEDAVG_MODEL_PATH = 'model_fedavg_20251117_195739.pth'
```

**New (auto-detect):**
```python
from utils import find_latest_model
FEDAVG_MODEL_PATH = find_latest_model('model_fedavg_*.pth') or 'model_fedavg_20251117_195739.pth'
```

## Configuration System

New centralized configuration in `config/config.yaml`:

**Old:**
```python
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
# Hardcoded everywhere
```

**New:**
```python
from config import load_config
config = load_config()
BATCH_SIZE = config['training']['batch_size']
LEARNING_RATE = config['training']['learning_rate']
```

## Checklist for Updating Scripts

- [ ] Update model imports to use `from models import HybridLSTMCNNAttention`
- [ ] Update data paths to use `Path(__file__).parent.parent` for project root
- [ ] Add project root to `sys.path` at the top of scripts
- [ ] Update any hardcoded paths to use relative paths from project root
- [ ] Replace hardcoded model paths with auto-detection (for stream.py)
- [ ] Test that scripts run from project root directory

## Next Steps

1. **Update training scripts**: Fix imports in `training/*.py` files
2. **Update evaluation script**: Fix imports in `evaluation/stream.py`
3. **Test all scripts**: Ensure everything runs correctly
4. **Update documentation**: Ensure README reflects new structure

## Need Help?

If you encounter import errors:

1. Make sure you're running scripts from the project root
2. Check that `sys.path` includes the project root
3. Verify that `models/__init__.py` exists and exports correctly
4. Use absolute imports: `from models.hybrid_model import HybridLSTMCNNAttention`

