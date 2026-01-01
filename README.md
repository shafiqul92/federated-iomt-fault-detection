# Fault Detection in Air Handling Units (AHU) - IoMT System

A comprehensive fault detection system for Air Handling Units in hospital environments using centralized and federated learning approaches with edge-aware resource management.

## 🎯 Overview

This project implements three machine learning paradigms for AHU fault detection:

1. **Centralized Baseline**: Traditional centralized training with all data
2. **FedAvg**: Federated Averaging for distributed learning across multiple AHUs
3. **Edge-Aware FL**: Resource-aware federated learning considering CPU, battery, bandwidth, and memory constraints

### Fault Classes
- **Class 0**: Normal Condition
- **Class 1**: Return Air Temperature Sensor Fault (RATSF)
- **Class 2**: Supply Fan Fault (SFF)
- **Class 3**: Valve Position Fault (VPF)

## 📁 Project Structure

```
fault-detection-iomt/
├── config/              # Configuration files
│   ├── config.yaml      # Main configuration
│   └── __init__.py
├── data/                # Data processing scripts
│   ├── preprocess.py    # Data preprocessing
│   └── split.py         # Train/val/test splitting with SMOTE
├── models/              # Model architectures
│   ├── hybrid_model.py  # Hybrid LSTM+CNN+Attention model
│   └── __init__.py
├── training/            # Training scripts
│   ├── train_centralized.py
│   ├── train_fedavg.py
│   └── train_edge_aware.py
├── evaluation/          # Evaluation and visualization
│   └── stream.py        # Real-time streaming comparison
├── utils/               # Utility functions
│   └── __init__.py
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (tested with 3.13.9)
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shafiqul92/federated-iomt-fault-detection.git
   cd federated-iomt-fault-detection
   ```

2. **Create virtual environment**
   
   **Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
   **Linux/MacOS:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

1. **Place your raw data file** (`raw_data.csv`) in the root directory

2. **Preprocess the data**
   ```bash
   python data/preprocess.py
   ```
   This will generate `preprocessed_data.csv` with cleaned and feature-engineered data.

3. **Split data for training**
   ```bash
   python data/split.py
   ```
   This creates:
   - `processed_splits_advanced/centralized/` - Centralized training splits
   - `processed_splits_advanced/federated/` - Per-AHU splits for federated learning

## 🏋️ Training

### 1. Centralized Baseline Training

Train a centralized model on all data:

```bash
python training/train_centralized.py
```

**Output:**
- `best_model_centralized_baseline.pth` - Best model checkpoint
- Training logs and metrics

### 2. Federated Averaging (FedAvg)

Train using federated learning with FedAvg:

```bash
python training/train_fedavg.py
```

**Note:** Requires the centralized baseline model for initialization (optional but recommended).

**Output:**
- `model_fedavg_<timestamp>.pth` - Trained model
- `results_fedavg_<timestamp>.json` - Evaluation metrics

### 3. Edge-Aware Federated Learning

Train with resource-aware federated learning:

```bash
python training/train_edge_aware.py
```

This approach considers:
- CPU capability
- Battery/power availability
- Network bandwidth
- Memory constraints

**Output:**
- `model_edge_aware_<timestamp>.pth` - Trained model
- `results_edge_aware_<timestamp>.json` - Metrics and device statistics

**Configuration:** Edit `config/config.yaml` to customize edge device specifications and resource thresholds.

## 📊 Evaluation & Visualization

### Real-Time Streaming Comparison

Compare all three models on streaming data:

```bash
python evaluation/stream.py
```

This script:
- Loads all three trained models
- Processes test data in real-time
- Generates comprehensive visualizations:
  - Real-time accuracy/F1 curves
  - Confusion matrices
  - Per-class performance metrics
  - Model agreement analysis
  - Convergence analysis
- Creates checkpoint plots during processing

**Output:** All plots saved in `plots/` directory

**Note:** The script automatically detects the latest model files. Ensure models are trained before running.

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

- **Model architecture** (embedding dimensions, layers, etc.)
- **Training hyperparameters** (batch size, learning rate, epochs)
- **Federated learning** settings (rounds, local epochs)
- **Edge device specifications** (CPU, battery, bandwidth, memory)
- **Resource thresholds** for edge-aware FL

## 📈 Model Architecture

The project uses a **Hybrid LSTM-CNN-Attention** model:

- **CNN Branch**: Extracts local patterns from features
- **LSTM Branch**: Captures temporal dependencies
- **Multi-Head Attention**: Focuses on important features
- **Residual Connections**: Enables deeper networks

**Parameters:** ~5-10M (CPU-friendly)

## 🔧 Advanced Usage

### Custom Edge Device Configuration

Edit `config/config.yaml` under `edge_aware.devices`:

```yaml
devices:
  ahu_1: [cpu_score, battery%, bandwidth_mbps, memory_mb]
  ahu_2: [70, 75, 30, 1536]
  # ...
```

### Ablation Studies

The edge-aware training script supports multiple ablation study configurations:

- **Adaptive epochs**: Adjust training epochs based on device capability
- **Adaptive weights**: Weight aggregation by device resources
- **Smart selection**: Select devices based on resource availability
- **Offloading strategies**: CPU-based, battery-based, network-based

Edit `config/config.yaml` or modify `training/train_edge_aware.py` to explore different configurations.

## 📚 Documentation

- **Edge-Aware Policy**: See `edge_aware_policy.md` for detailed policy documentation
- **Code Documentation**: Docstrings are provided in all major modules

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- Dataset: AHU fault detection data from hospital environments
- Libraries: PyTorch, scikit-learn, imbalanced-learn

## 📧 Contact

[Add your contact information]

---

**Note:** Make sure to have sufficient disk space for processed data and model checkpoints (~1-2 GB recommended).
