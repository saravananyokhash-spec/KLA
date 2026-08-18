import os
import yaml
import random
import numpy as np
import torch

def load_config(config_path):
    """
    Loads a YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
    """
    Sets random seeds for reproducibility across packages.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Ensure deterministic PyTorch behaviors
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")
