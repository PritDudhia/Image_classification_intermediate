"""
Utility functions for configuration management.
"""

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> DictConfig:
    """
    Load Hydra configuration from file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration object
    """
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration object
        save_path: Path to save config
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_path)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
    
    Returns:
        Merged configuration
    """
    return OmegaConf.merge(base_config, override_config)


def print_config(config: DictConfig):
    """
    Pretty print configuration.
    
    Args:
        config: Configuration to print
    """
    print(OmegaConf.to_yaml(config))
