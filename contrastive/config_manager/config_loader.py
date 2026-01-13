#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Loader

Loads and manages configuration from external YAML files,
replacing hardcoded dataset references.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from omegaconf import DictConfig, OmegaConf
import logging

from .dataset_registry import DatasetRegistry

log = logging.getLogger(__name__)


class ConfigLoader:
    """
    Load and manage configurations from external files.

    Provides a bridge between external YAML configs and Hydra/OmegaConf.
    """

    def __init__(self, config_root: Optional[str] = None):
        """
        Initialize the config loader.

        Args:
            config_root: Root directory for configuration files
        """
        self.config_root = config_root or self._get_default_config_root()
        self.dataset_registry = None

    def _get_default_config_root(self) -> str:
        """Get default config root (champollion_V1/contrastive/configs)."""
        # Assume this file is in champollion_V1/contrastive/config_manager/
        current_file = Path(__file__).resolve()
        config_root = current_file.parent.parent / "configs"
        return str(config_root)

    def get_dataset_registry(self, force_reload: bool = False) -> DatasetRegistry:
        """
        Get or create dataset registry.

        Args:
            force_reload: Force reload of datasets

        Returns:
            DatasetRegistry instance
        """
        if self.dataset_registry is None or force_reload:
            dataset_dir = os.path.join(self.config_root, "dataset")
            self.dataset_registry = DatasetRegistry(
                config_dir=dataset_dir,
                auto_discover=True
            )
            log.info(f"Loaded {len(self.dataset_registry.list_datasets())} datasets")

        return self.dataset_registry

    def load_datasets_from_names(
        self,
        dataset_names: List[str],
        datasets_root: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load dataset configurations by name.

        Args:
            dataset_names: List of dataset names (can be paths like 'gridsearch_2024/hcp_left')
            datasets_root: Optional root to prepend to dataset names

        Returns:
            Dictionary mapping dataset names to their configurations
        """
        registry = self.get_dataset_registry()

        # If datasets_root is provided, prepend it to dataset names
        if datasets_root:
            full_names = [f"{datasets_root}/{name}" for name in dataset_names]
        else:
            full_names = dataset_names

        result = {}
        for orig_name, full_name in zip(dataset_names, full_names):
            dataset = registry.get(full_name)
            if dataset:
                result[full_name] = dataset.to_dict()
                log.debug(f"Loaded dataset: {full_name}")
            else:
                log.warning(f"Dataset not found: {full_name}")

        return result

    def load_label_config(self, label_name: str) -> Dict[str, Any]:
        """
        Load label configuration from YAML.

        Args:
            label_name: Name of the label config file (without .yaml)

        Returns:
            Dictionary of label configuration
        """
        label_path = os.path.join(self.config_root, "label", f"{label_name}.yaml")

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label config not found: {label_path}")

        with open(label_path, 'r') as f:
            label_config = yaml.safe_load(f)

        return label_config

    def load_dataset_localization(self, localization_name: str) -> Dict[str, Any]:
        """
        Load dataset localization configuration.

        Args:
            localization_name: Name of the localization config (e.g., 'local', 'remote')

        Returns:
            Dictionary of localization configuration
        """
        localization_path = os.path.join(
            self.config_root,
            "dataset_localization",
            f"{localization_name}.yaml"
        )

        if not os.path.exists(localization_path):
            raise FileNotFoundError(f"Localization config not found: {localization_path}")

        with open(localization_path, 'r') as f:
            localization_config = yaml.safe_load(f)

        return localization_config

    def load_classifier_config(self, classifier_name: str) -> Dict[str, Any]:
        """
        Load classifier configuration.

        Args:
            classifier_name: Name of the classifier (e.g., 'svm', 'neural_network')

        Returns:
            Dictionary of classifier configuration
        """
        classifier_path = os.path.join(
            self.config_root,
            "classifier",
            f"{classifier_name}.yaml"
        )

        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier config not found: {classifier_path}")

        with open(classifier_path, 'r') as f:
            classifier_config = yaml.safe_load(f)

        return classifier_config

    def create_evaluation_config(
        self,
        model_path: str,
        dataset_names: List[str],
        label_name: str,
        dataset_localization: str = "local",
        classifier_name: str = "svm",
        datasets_root: Optional[str] = None,
        **kwargs
    ) -> DictConfig:
        """
        Create a complete evaluation configuration.

        Args:
            model_path: Path to trained model
            dataset_names: List of dataset names to evaluate on
            label_name: Label to use for evaluation
            dataset_localization: Dataset localization type
            classifier_name: Classifier to use
            datasets_root: Optional root for dataset names
            **kwargs: Additional config overrides

        Returns:
            OmegaConf DictConfig ready for evaluation
        """
        # Load model's original config
        model_config_path = os.path.join(model_path, '.hydra', 'config.yaml')

        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config not found: {model_config_path}")

        config = OmegaConf.load(model_config_path)

        # Replace datasets
        datasets_dict = self.load_datasets_from_names(dataset_names, datasets_root)
        config.dataset = OmegaConf.create(datasets_dict)

        # Update dataset keys
        config.datakey = list(datasets_dict.keys())

        # Load and apply label config
        label_config = self.load_label_config(label_name)
        for key, value in label_config.items():
            config[key] = value

        # Load and apply dataset localization
        localization_config = self.load_dataset_localization(dataset_localization)
        for key, value in localization_config.items():
            config[key] = value

        # Load and apply classifier config
        classifier_config = self.load_classifier_config(classifier_name)
        for key, value in classifier_config.items():
            config[key] = value

        # Apply additional overrides
        for key, value in kwargs.items():
            config[key] = value

        # Set model path
        config.model_path = model_path

        return config

    def save_config(self, config: DictConfig, output_path: str):
        """
        Save config to YAML file.

        Args:
            config: OmegaConf DictConfig to save
            output_path: Path to output file
        """
        with open(output_path, 'w') as f:
            OmegaConf.save(config, f)

        log.info(f"Saved config to {output_path}")

    def list_available_datasets(self, pattern: Optional[str] = None) -> List[str]:
        """
        List all available datasets in the registry.

        Args:
            pattern: Optional glob pattern to filter datasets

        Returns:
            List of dataset names
        """
        registry = self.get_dataset_registry()

        if pattern:
            return registry.filter_datasets(pattern)
        else:
            return registry.list_datasets()

    def list_available_labels(self) -> List[str]:
        """List all available label configurations."""
        label_dir = os.path.join(self.config_root, "label")

        if not os.path.exists(label_dir):
            return []

        labels = [
            f.stem for f in Path(label_dir).glob("*.yaml")
        ]

        return sorted(labels)

    def list_available_classifiers(self) -> List[str]:
        """List all available classifier configurations."""
        classifier_dir = os.path.join(self.config_root, "classifier")

        if not os.path.exists(classifier_dir):
            return []

        classifiers = [
            f.stem for f in Path(classifier_dir).glob("*.yaml")
        ]

        return sorted(classifiers)
