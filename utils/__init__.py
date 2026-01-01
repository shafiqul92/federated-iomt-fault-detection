"""
Utility functions for AHU Fault Detection
"""

import os
import glob
from pathlib import Path
from datetime import datetime

def find_latest_model(pattern, root_dir='.'):
    """
    Find the latest model file matching a pattern
    
    Args:
        pattern: Pattern to match (e.g., 'model_fedavg_*.pth', 'best_model_*.pth')
        root_dir: Directory to search in
        
    Returns:
        Path to latest file or None if not found
    """
    root = Path(root_dir)
    matches = list(root.glob(pattern))
    
    if not matches:
        return None
    
    # Sort by modification time (newest first)
    matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(matches[0])


def find_all_model_patterns():
    """
    Find all model files in the directory
    
    Returns:
        dict with keys: 'centralized', 'fedavg', 'edge_aware'
    """
    patterns = {
        'centralized': 'best_model_centralized_baseline.pth',
        'fedavg': 'model_fedavg_*.pth',
        'edge_aware': 'model_edge_aware_*.pth'
    }
    
    results = {}
    for key, pattern in patterns.items():
        path = find_latest_model(pattern)
        results[key] = path
    
    return results


def get_timestamp_string():
    """Get timestamp string for file naming"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

