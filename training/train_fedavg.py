"""
FedAvg Training - CUDA Version
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
LOCAL_EPOCHS = 5
BATCH_SIZE = 128
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3


# ============================================================================
# DATASET
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


# ============================================================================
# FUNCTIONS
# ============================================================================

def train_local(model, loader, epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
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
    print("\nLoading data...")

    # Get AHUs
    ahus = sorted([d for d in os.listdir(str(SPLITS_DIR)) if os.path.isdir(SPLITS_DIR / d)])
    print(f"AHUs: {ahus}")

    # Load
    clients = {}
    input_dim = None
    num_classes = None

    for ahu in ahus:
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

    # Test
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
                # Alternative: check if it's a different layer structure
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
    print(f"\nTraining FedAvg for {NUM_ROUNDS} rounds...")

    best_f1 = 0
    best_state = None

    for r in range(NUM_ROUNDS):
        print(f"\nRound {r+1}/{NUM_ROUNDS}")

        states, weights = [], []

        for ahu, data in clients.items():
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
            train_local(local, data['train'], LOCAL_EPOCHS)

            # Val
            val_preds, val_labels = evaluate(local, data['val'])
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
            print(f"  {ahu}: F1={val_f1*100:.1f}%")

            states.append(copy.deepcopy(local.state_dict()))
            weights.append(data['size'])
            del local

        # Aggregate
        model.load_state_dict(average_states(states, weights))

        # Test
        test_preds, test_labels = evaluate(model, test_loader)
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
        test_f1_m = f1_score(test_labels, test_preds, average='macro', zero_division=0)

        print(f"  Global: Acc={test_acc*100:.2f}%, F1={test_f1*100:.2f}%, F1(M)={test_f1_m*100:.2f}%")

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_state = copy.deepcopy(model.state_dict())
            print(f"  ✓ Best: {best_f1*100:.2f}%")

    # Final
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print("="*80)

    model.load_state_dict(best_state)
    preds, labels = evaluate(model, test_loader)

    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average='weighted', zero_division=0)
    f1_m = f1_score(labels, preds, average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds)

    per_class = {}
    # Use model_num_classes which might be different from data's num_classes
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

    # Save
    results = {
        'final_metrics': {
            'accuracy': float(acc),
            'f1_weighted': float(f1_w),
            'f1_macro': float(f1_m),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class
        },
        'config': {'num_rounds': NUM_ROUNDS, 'local_epochs': LOCAL_EPOCHS}
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = PROJECT_ROOT / f'results_fedavg_{ts}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    model_file = PROJECT_ROOT / f'model_fedavg_{ts}.pth'
    torch.save(best_state, model_file)

    print(f"\n✓ Saved!")
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
    print(f"Random seed set to {SEED} for reproducibility\n")
    
    print("="*80)
    print("FEDAVG - CUDA VERSION")
    print("="*80)

    # Use CUDA if available
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    main()

