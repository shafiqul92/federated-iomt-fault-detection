"""
Configuration management
"""

import yaml
import os
from pathlib import Path

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / config_path if not os.path.isabs(config_path) else config_path
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model_paths(pattern, root_dir='.'):
    """
    Auto-detect latest model files matching pattern
    
    Args:
        pattern: Pattern to match (e.g., 'model_fedavg_*.pth')
        root_dir: Root directory to search
        
    Returns:
        Path to latest matching file or None
    """
    import glob
    from pathlib import Path
    
    root = Path(root_dir)
    matches = list(root.glob(pattern))
    
    if not matches:
        return None
    
    # Sort by modification time, return latest
    matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(matches[0])

