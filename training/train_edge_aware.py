"""
Edge-Aware FL Training - CUDA Version
Considers: CPU capability, battery, bandwidth, memory
Uses centralized baseline model as initialization
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Import model from shared module
from models.hybrid_model import HybridLSTMCNNAttention

# ============================================================================
# CONFIG
# ============================================================================

SPLITS_DIR = PROJECT_ROOT / 'processed_splits_advanced/federated'
TEST_DIR = PROJECT_ROOT / 'processed_splits_advanced/centralized'
BASELINE_MODEL_PATH = PROJECT_ROOT / 'best_model_centralized_baseline.pth'

NUM_ROUNDS = 40
BASE_LOCAL_EPOCHS = 5

BATCH_SIZE = 128
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

# ============================================================================
# ABLATION STUDY CONFIGURATION
# ============================================================================

# Study 1: Edge-Aware Component Ablation
# Options: 'baseline', 'adaptive_epochs', 'adaptive_weights', 'smart_selection', 'full_edge_aware'
ABLATION_STUDY_1 = 'full_edge_aware'

ENABLE_ADAPTIVE_EPOCHS = True      # Study 1.2
ENABLE_ADAPTIVE_WEIGHTS = True     # Study 1.3
ENABLE_SMART_SELECTION = True      # Study 1.4

# Study 2: Offloading Strategy Ablation
# Options: 'none', 'cpu_based', 'battery_based', 'network_based', 'hybrid'
OFFLOADING_STRATEGY = 'network_based'

# Study 3: Weight Aggregation Ablation
# Options: 'data_only', 'data_cpu', 'data_battery', 'full'
WEIGHT_STRATEGY = 'full'

# Study 4: Resource Threshold Ablation
# Options: 'strict', 'moderate', 'lenient'
THRESHOLD_MODE = 'moderate'

# Study 5: Epoch Adaptation Ablation
# Options: 'fixed', 'linear', 'conservative', 'aggressive'
EPOCH_ADAPTATION = 'conservative'

# Set thresholds based on mode
if THRESHOLD_MODE == 'strict':
    MIN_BATTERY, MIN_CPU = 40, 50
elif THRESHOLD_MODE == 'lenient':
    MIN_BATTERY, MIN_CPU = 10, 20
else:  # moderate (default)
    MIN_BATTERY, MIN_CPU = 20, 30


# ============================================================================
# EDGE DEVICES
# ============================================================================

class EdgeDevice:
    def __init__(self, name, cpu_score, battery, bandwidth, memory):
        self.name = name
        self.cpu_score = cpu_score
        self.battery = battery
        self.bandwidth = bandwidth
        self.memory = memory
        self.training_count = 0
        self.total_time = 0

    def can_train(self, min_battery, min_cpu):
        return self.battery >= min_battery and self.cpu_score >= min_cpu

    def consume_battery(self, training_time):
        drain = training_time / 60.0  # 1% per minute
        self.battery = max(0, self.battery - drain)
        self.training_count += 1
        self.total_time += training_time

    def recharge(self, amount=10):
        self.battery = min(100, self.battery + amount)

    def get_epochs(self, base_epochs, adaptation_mode='conservative'):
        """
        Get adapted epochs based on CPU score and adaptation mode
        
        Modes:
        - 'fixed': No adaptation, return base_epochs
        - 'linear': epochs = base_epochs * (cpu_score / 100)
        - 'conservative': epochs = base_epochs * (0.5 + 0.5 * cpu_score/100)
        - 'aggressive': epochs = base_epochs * (0.3 + 0.7 * cpu_score/100)
        """
        if adaptation_mode == 'fixed':
            return base_epochs
        
        factor = self.cpu_score / 100.0
        
        if adaptation_mode == 'linear':
            adapted = base_epochs * factor
        elif adaptation_mode == 'aggressive':
            adapted = base_epochs * (0.3 + 0.7 * factor)
        else:  # conservative (default)
            adapted = base_epochs * (0.5 + 0.5 * factor)
        
        return max(1, int(adapted))

    def get_weight(self, data_size, weight_strategy='full'):
        """
        Get aggregation weight based on strategy
        
        Strategies:
        - 'data_only': w = data_size (standard FedAvg)
        - 'data_cpu': w = data_size * (0.7 + 0.3*cpu)
        - 'data_battery': w = data_size * (0.8 + 0.2*battery)
        - 'full': w = data_size * (0.5 + 0.3*cpu + 0.2*battery)
        """
        if weight_strategy == 'data_only':
            return float(data_size)
        
        cpu_factor = self.cpu_score / 100.0
        battery_factor = self.battery / 100.0
        
        if weight_strategy == 'data_cpu':
            return data_size * (0.7 + 0.3 * cpu_factor)
        elif weight_strategy == 'data_battery':
            return data_size * (0.8 + 0.2 * battery_factor)
        else:  # full
            return data_size * (0.5 + 0.3 * cpu_factor + 0.2 * battery_factor)
    
    def should_offload(self, offload_strategy='none'):
        """
        Determine if training should be offloaded based on strategy
        
        Strategies:
        - 'none': Never offload (train locally)
        - 'cpu_based': Offload if CPU < 50
        - 'battery_based': Offload if battery < 20%
        - 'network_based': Offload if bandwidth < 10 Mbps
        - 'hybrid': Combined decision (CPU < 50 OR battery < 20% OR bandwidth < 10)
        """
        if offload_strategy == 'none':
            return False
        elif offload_strategy == 'cpu_based':
            return self.cpu_score < 50
        elif offload_strategy == 'battery_based':
            return self.battery < 20
        elif offload_strategy == 'network_based':
            return self.bandwidth < 10
        elif offload_strategy == 'hybrid':
            return (self.cpu_score < 50 or 
                   self.battery < 20 or 
                   self.bandwidth < 10)
        return False


# Edge devices for each AHU
EDGE_DEVICES = {
    'ahu_1': EdgeDevice('ahu_1', cpu_score=85, battery=90, bandwidth=50, memory=2048),
    'ahu_2': EdgeDevice('ahu_2', cpu_score=70, battery=75, bandwidth=30, memory=1536),
    'ahu_3': EdgeDevice('ahu_3', cpu_score=90, battery=95, bandwidth=100, memory=4096),
    'ahu_4': EdgeDevice('ahu_4', cpu_score=60, battery=60, bandwidth=20, memory=1024),
    'ahu_5': EdgeDevice('ahu_5', cpu_score=80, battery=85, bandwidth=40, memory=2048),
    'ahu_6': EdgeDevice('ahu_6', cpu_score=75, battery=70, bandwidth=35, memory=1536),
    'ahu_8': EdgeDevice('ahu_8', cpu_score=65, battery=65, bandwidth=25, memory=1024),
}


# ============================================================================
# DATASET & MODEL (same as FedAvg)
# ============================================================================

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.X = X.values.astype(np.float32)
        else:
            self.X = X.astype(np.float32)
        if isinstance(y, pd.Series):
            self.y = y.values.astype(np.int64)
        else:
            self.y = y.astype(np.int64)
        self.X = np.nan_to_num(self.X, nan=0.0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


# ============================================================================
# MODEL
# ============================================================================
# Model is imported from hybrid_lstm.py (HybridLSTMCNNAttention)


def train_local(model, loader, epochs, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    start = time.time()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return time.time() - start


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds.extend(outputs.max(1)[1].cpu().numpy())
            labels.extend(y.cpu().numpy())
    return preds, labels


def average_states(states, weights):
    total = sum(weights)
    avg = {}
    for key in states[0].keys():
        avg[key] = sum(states[i][key] * (weights[i] / total) for i in range(len(states)))
    return avg


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("EDGE-AWARE FL - CUDA VERSION")
    print("="*80)
    
    # Use CUDA if available
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    print("\nLoading data...")

    ahus = sorted([d for d in os.listdir(str(SPLITS_DIR)) if os.path.isdir(SPLITS_DIR / d)])
    print(f"AHUs: {ahus}")

    clients = {}
    input_dim = None
    num_classes = None

    for ahu in ahus:
        if ahu not in EDGE_DEVICES:
            continue

        train_df = pd.read_csv(SPLITS_DIR / ahu / 'train.csv')
        val_df = pd.read_csv(SPLITS_DIR / ahu / 'val.csv')

        # Drop label and any metadata columns
        ahu_cols = [col for col in train_df.columns if col.startswith('ahu_')]
        drop_cols = ['label'] + ahu_cols
        extra_cols = ['id', 'label_full', 'AHU_name']
        for col in extra_cols:
            if col in train_df.columns and col not in drop_cols:
                drop_cols.append(col)
        
        X_tr = train_df.drop(columns=drop_cols, errors='ignore')
        y_tr = train_df['label'].values
        X_val = val_df.drop(columns=drop_cols, errors='ignore')
        y_val = val_df['label'].values

        if input_dim is None:
            input_dim = X_tr.shape[1]
        
        if num_classes is None:
            num_classes = len(np.unique(y_tr))

        print(f"{ahu}: train={len(X_tr)}, val={len(X_val)}")

        clients[ahu] = {
            'train': DataLoader(SimpleDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True),
            'val': DataLoader(SimpleDataset(X_val, y_val), batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS, pin_memory=True),
            'size': len(X_tr)
        }

    test_df = pd.read_csv(TEST_DIR / 'test.csv')
    # Drop same columns as training data
    ahu_cols = [col for col in test_df.columns if col.startswith('ahu_')]
    drop_cols = ['label'] + ahu_cols
    extra_cols = ['id', 'label_full', 'AHU_name']
    for col in extra_cols:
        if col in test_df.columns and col not in drop_cols:
            drop_cols.append(col)
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df['label'].values
    test_loader = DataLoader(SimpleDataset(X_test, y_test), batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Test: {len(X_test)}")

    print(f"\nEdge Devices:")
    for name, dev in EDGE_DEVICES.items():
        if name in clients:
            print(f"  {name}: CPU={dev.cpu_score}, Battery={dev.battery}%, BW={dev.bandwidth}Mbps")

    # Model - Use same architecture as centralized baseline
    print(f"\nCreating model...")
    
    # Try to infer num_classes from baseline model if it exists
    baseline_num_classes = None
    if os.path.exists(BASELINE_MODEL_PATH):
        print(f"Checking baseline model at {BASELINE_MODEL_PATH}...")
        try:
            state_dict = torch.load(BASELINE_MODEL_PATH, map_location=DEVICE)
            # Infer num_classes from the classifier output layer
            if 'classifier.8.weight' in state_dict:
                baseline_num_classes = state_dict['classifier.8.weight'].shape[0]
                print(f"  Baseline model has {baseline_num_classes} classes")
            elif 'classifier.2.weight' in state_dict:
                baseline_num_classes = state_dict['classifier.2.weight'].shape[0]
                print(f"  Baseline model has {baseline_num_classes} classes")
        except Exception as e:
            print(f"  Could not read baseline model: {e}")
    
    # Use baseline num_classes if available, otherwise use data's num_classes
    if baseline_num_classes is not None:
        model_num_classes = baseline_num_classes
        print(f"Using num_classes={model_num_classes} from baseline model")
    else:
        model_num_classes = num_classes
        print(f"Using num_classes={model_num_classes} from data")
    
    print(f"Input dim: {input_dim}, Num classes: {model_num_classes}")
    model = HybridLSTMCNNAttention(
        input_dim=input_dim,
        num_classes=model_num_classes,
        embed_dim=256,
        cnn_channels=[128, 256, 384],
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        num_attention_heads=8,
        num_residual_blocks=2,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Load centralized baseline model weights
    if BASELINE_MODEL_PATH.exists():
        print(f"Loading baseline model weights from {BASELINE_MODEL_PATH}...")
        try:
            state_dict = torch.load(str(BASELINE_MODEL_PATH), map_location=DEVICE)
            model.load_state_dict(state_dict, strict=True)
            print("✓ Baseline model loaded successfully!")
        except Exception as e:
            print(f"⚠ Warning: Could not load baseline model: {e}")
            print("  Starting from random initialization...")
    else:
        print(f"⚠ Warning: Baseline model not found at {BASELINE_MODEL_PATH}")
        print("  Starting from random initialization...")
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    print(f"\nTraining Edge-Aware FL for {NUM_ROUNDS} rounds...")

    best_f1 = 0
    best_state = None

    for r in range(NUM_ROUNDS):
        print(f"\nRound {r+1}/{NUM_ROUNDS}")

        # Select eligible clients (with smart selection if enabled)
        if ENABLE_SMART_SELECTION:
            eligible = [ahu for ahu in clients.keys()
                       if EDGE_DEVICES[ahu].can_train(MIN_BATTERY, MIN_CPU)]
        else:
            # Baseline: all clients participate
            eligible = list(clients.keys())

        if not eligible:
            print("  No eligible clients! Recharging...")
            for dev in EDGE_DEVICES.values():
                dev.recharge(20)
            continue

        print(f"  Eligible: {eligible}")

        states, weights = [], []

        for ahu in eligible:
            dev = EDGE_DEVICES[ahu]
            
            # Check if should offload
            should_offload = dev.should_offload(OFFLOADING_STRATEGY)
            if should_offload:
                print(f"  {ahu}: Offloading training (strategy={OFFLOADING_STRATEGY})")
                # In real implementation, would offload to edge/cloud server
                # For now, we skip this device or train anyway
                # continue  # Uncomment to actually skip offloaded devices

            # Adapt epochs
            if ENABLE_ADAPTIVE_EPOCHS:
                epochs = dev.get_epochs(BASE_LOCAL_EPOCHS, EPOCH_ADAPTATION)
            else:
                epochs = BASE_LOCAL_EPOCHS

            # Local model - same architecture as global
            local = HybridLSTMCNNAttention(
                input_dim=input_dim,
                num_classes=model_num_classes,
                embed_dim=256,
                cnn_channels=[128, 256, 384],
                lstm_hidden_dim=256,
                lstm_num_layers=2,
                num_attention_heads=8,
                num_residual_blocks=2,
                dropout=DROPOUT
            ).to(DEVICE)
            local.load_state_dict(model.state_dict())

            # Train
            train_time = train_local(local, clients[ahu]['train'], epochs, DEVICE)
            dev.consume_battery(train_time)

            # Val
            val_preds, val_labels = evaluate(local, clients[ahu]['val'], DEVICE)
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

            print(f"  {ahu}: epochs={epochs}, time={train_time:.1f}s, battery={dev.battery:.1f}%, F1={val_f1*100:.1f}%")

            # Aggregation weight
            if ENABLE_ADAPTIVE_WEIGHTS:
                weight = dev.get_weight(clients[ahu]['size'], WEIGHT_STRATEGY)
            else:
                weight = float(clients[ahu]['size'])  # Baseline: data size only
            
            states.append(copy.deepcopy(local.state_dict()))
            weights.append(weight)
            del local

        # Aggregate
        model.load_state_dict(average_states(states, weights))

        # Test
        test_preds, test_labels = evaluate(model, test_loader, DEVICE)
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
        test_f1_m = f1_score(test_labels, test_preds, average='macro', zero_division=0)

        print(f"  Global: Acc={test_acc*100:.2f}%, F1={test_f1*100:.2f}%, F1(M)={test_f1_m*100:.2f}%")

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_state = copy.deepcopy(model.state_dict())
            print(f"  ✓ Best: {best_f1*100:.2f}%")

        # Periodic recharge
        if (r + 1) % 10 == 0:
            for dev in EDGE_DEVICES.values():
                dev.recharge(15)

    # Final
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print("="*80)

    model.load_state_dict(best_state)
    preds, labels = evaluate(model, test_loader, DEVICE)

    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average='weighted', zero_division=0)
    f1_m = f1_score(labels, preds, average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds)

    per_class = {}
    # Use actual number of classes in labels
    actual_num_classes = len(np.unique(labels))
    for i in range(actual_num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        per_class[f'class_{i}'] = {
            'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
            'precision': float(TP/(TP+FP)) if TP+FP>0 else 0.0,
            'recall': float(TP/(TP+FN)) if TP+FN>0 else 0.0,
            'f1': float(2*TP/(2*TP+FP+FN)) if 2*TP+FP+FN>0 else 0.0
        }

    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1 (Weighted): {f1_w*100:.2f}%")
    print(f"F1 (Macro): {f1_m*100:.2f}%")

    print(f"\nConfusion Matrix:")
    print(cm)

    print(f"\nPer-Class:")
    for cn, m in sorted(per_class.items()):
        print(f"  {cn}: TP={m['TP']:4d} TN={m['TN']:4d} FP={m['FP']:3d} FN={m['FN']:3d} | "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    print(f"\n{classification_report(labels, preds, zero_division=0)}")

    # Edge device stats
    print(f"\n{'='*80}")
    print("EDGE DEVICE STATS")
    print("="*80)
    for name, dev in sorted(EDGE_DEVICES.items()):
        if name in clients:
            print(f"{name}: rounds={dev.training_count}, total_time={dev.total_time:.1f}s, final_battery={dev.battery:.1f}%")

    # Save
    results = {
        'final_metrics': {
            'accuracy': float(acc),
            'f1_weighted': float(f1_w),
            'f1_macro': float(f1_m),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class
        },
        'device_stats': {
            name: {
                'training_count': dev.training_count,
                'total_time': dev.total_time,
                'final_battery': dev.battery
            }
            for name, dev in EDGE_DEVICES.items() if name in clients
        },
        'config': {
            'num_rounds': NUM_ROUNDS,
            'base_epochs': BASE_LOCAL_EPOCHS,
            'min_battery': MIN_BATTERY,
            'min_cpu': MIN_CPU,
            'threshold_mode': THRESHOLD_MODE
        },
        'ablation_config': {
            'study_1_component': ABLATION_STUDY_1,
            'enable_adaptive_epochs': ENABLE_ADAPTIVE_EPOCHS,
            'enable_adaptive_weights': ENABLE_ADAPTIVE_WEIGHTS,
            'enable_smart_selection': ENABLE_SMART_SELECTION,
            'offloading_strategy': OFFLOADING_STRATEGY,
            'weight_strategy': WEIGHT_STRATEGY,
            'threshold_mode': THRESHOLD_MODE,
            'epoch_adaptation': EPOCH_ADAPTATION
        }
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create descriptive filename based on ablation config
    config_str = f"{ABLATION_STUDY_1}_{WEIGHT_STRATEGY}_{THRESHOLD_MODE}_{EPOCH_ADAPTATION}"
    if OFFLOADING_STRATEGY != 'none':
        config_str += f"_offload_{OFFLOADING_STRATEGY}"
    
    results_file = PROJECT_ROOT / f'results_edge_aware_{config_str}_{ts}.json'
    model_file = PROJECT_ROOT / f'model_edge_aware_{config_str}_{ts}.pth'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(best_state, model_file)

    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Model saved to: {model_file}")
    print("="*80)


if __name__ == "__main__":
    # Set seed for reproducibility
    import random
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("="*80)
    print("EDGE-AWARE FL WITH ABLATION STUDY SUPPORT")
    print("="*80)
    print(f"Random seed set to {SEED} for reproducibility")
    print(f"\nAblation Configuration:")
    print(f"  Study 1 (Components): {ABLATION_STUDY_1}")
    print(f"    - Adaptive Epochs: {ENABLE_ADAPTIVE_EPOCHS} ({EPOCH_ADAPTATION})")
    print(f"    - Adaptive Weights: {ENABLE_ADAPTIVE_WEIGHTS} ({WEIGHT_STRATEGY})")
    print(f"    - Smart Selection: {ENABLE_SMART_SELECTION}")
    print(f"  Study 2 (Offloading): {OFFLOADING_STRATEGY}")
    print(f"  Study 3 (Weights): {WEIGHT_STRATEGY}")
    print(f"  Study 4 (Thresholds): {THRESHOLD_MODE} (Battery≥{MIN_BATTERY}%, CPU≥{MIN_CPU}%)")
    print(f"  Study 5 (Epochs): {EPOCH_ADAPTATION}")
    print("="*80 + "\n")
    
    main()