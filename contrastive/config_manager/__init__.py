"""
Configuration Manager Module for Champollion V1

This module provides a decoupled configuration management system
that separates dataset definitions from code, making the codebase
more maintainable and flexible.
"""

from .dataset_registry import DatasetRegistry, DatasetConfig
from .config_loader import ConfigLoader
from .external_config_loader import ExternalConfigLoader
from .hydra_bridge import HydraBridge, register_omegaconf_resolvers

__all__ = [
    'DatasetRegistry',
    'DatasetConfig',
    'ConfigLoader',
    'ExternalConfigLoader',
    'HydraBridge',
    'register_omegaconf_resolvers',
]
