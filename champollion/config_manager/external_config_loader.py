#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External Config Loader

Extends ConfigLoader to support loading dataset configs from external paths
(e.g., data/TEST01/derivatives/champollion_V1/configs/).

This allows the pipeline to generate configs in the data directory and then
load them for embeddings generation.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from omegaconf import DictConfig, OmegaConf
import logging

from .config_loader import ConfigLoader
from .dataset_registry import DatasetRegistry, DatasetConfig

log = logging.getLogger(__name__)


class ExternalConfigLoader(ConfigLoader):
    """
    Config loader that supports loading from external dataset paths.

    Use this when dataset configs are generated in:
    champollion/data/DATASET/derivatives/champollion_V1/configs/
    """

    def __init__(
        self,
        config_root: Optional[str] = None,
        external_dataset_paths: Optional[List[str]] = None
    ):
        """
        Initialize the external config loader.

        Args:
            config_root: Root directory for built-in configuration files
            external_dataset_paths: Additional paths to search for dataset configs
                Example: ["/path/to/data/TEST01/derivatives/champollion_V1/configs"]
        """
        super().__init__(config_root)
        self.external_dataset_paths = external_dataset_paths or []

    def get_dataset_registry(
        self,
        force_reload: bool = False,
        include_external: bool = True
    ) -> DatasetRegistry:
        """
        Get or create dataset registry, optionally including external paths.

        Args:
            force_reload: Force reload of datasets
            include_external: Include datasets from external_dataset_paths

        Returns:
            DatasetRegistry instance
        """
        if self.dataset_registry is None or force_reload:
            # Load built-in datasets
            dataset_dir = os.path.join(self.config_root, "dataset")
            self.dataset_registry = DatasetRegistry(
                config_dir=dataset_dir,
                auto_discover=True
            )

            # Load external datasets
            if include_external:
                for external_path in self.external_dataset_paths:
                    if os.path.exists(external_path):
                        log.info(f"Loading external datasets from: {external_path}")
                        self._load_external_datasets(external_path)
                    else:
                        log.warning(f"External dataset path does not exist: {external_path}")

            log.info(f"Loaded {len(self.dataset_registry.list_datasets())} datasets total")

        return self.dataset_registry

    def _load_external_datasets(self, external_path: str):
        """
        Load datasets from an external path.

        Args:
            external_path: Path to directory containing dataset YAML files
        """
        external_path_obj = Path(external_path)

        if not external_path_obj.exists():
            log.warning(f"External path does not exist: {external_path}")
            return

        # Find all YAML files in the external path
        yaml_files = list(external_path_obj.glob("*.yaml"))

        log.info(f"Found {len(yaml_files)} YAML files in {external_path}")

        for yaml_file in yaml_files:
            try:
                # Load dataset config
                dataset_name = yaml_file.stem
                dataset = DatasetConfig.from_yaml(str(yaml_file), name=dataset_name)

                # Register with a prefix to indicate it's external
                # This helps avoid conflicts with built-in datasets
                external_name = f"external/{dataset_name}"
                self.dataset_registry.register(external_name, dataset)

                log.debug(f"Registered external dataset: {external_name}")
            except Exception as e:
                log.warning(f"Failed to load external dataset from {yaml_file}: {e}")

    def load_datasets_from_external_path(
        self,
        external_path: str,
        dataset_pattern: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load datasets specifically from an external path.

        Args:
            external_path: Path to directory containing dataset YAML files
            dataset_pattern: Optional glob pattern to filter datasets

        Returns:
            Dictionary mapping dataset names to their configurations
        """
        # Ensure this path is in our external paths
        if external_path not in self.external_dataset_paths:
            self.external_dataset_paths.append(external_path)

        # Reload registry to include this path
        self.get_dataset_registry(force_reload=True)

        # Get all external datasets
        all_datasets = self.dataset_registry.list_datasets()
        external_datasets = [d for d in all_datasets if d.startswith('external/')]

        # Filter by pattern if provided
        if dataset_pattern:
            from fnmatch import fnmatch
            external_datasets = [d for d in external_datasets if fnmatch(d, dataset_pattern)]

        # Get dataset configs
        result = {}
        for dataset_name in external_datasets:
            dataset = self.dataset_registry.get(dataset_name)
            if dataset:
                # Remove 'external/' prefix for the result
                clean_name = dataset_name.replace('external/', '')
                result[clean_name] = dataset.to_dict()

        return result

    def create_evaluation_config_from_external(
        self,
        model_path: str,
        external_configs_path: str,
        dataset_names: List[str],
        label_name: str,
        dataset_localization: str = "local",
        classifier_name: str = "svm",
        **kwargs
    ) -> DictConfig:
        """
        Create evaluation config using datasets from external path.

        Args:
            model_path: Path to trained model
            external_configs_path: Path where dataset configs were generated
                Example: "data/TEST01/derivatives/champollion_V1/configs"
            dataset_names: List of dataset names (without 'external/' prefix)
            label_name: Label to use for evaluation
            dataset_localization: Dataset localization type
            classifier_name: Classifier to use
            **kwargs: Additional config overrides

        Returns:
            OmegaConf DictConfig ready for evaluation
        """
        # Add external path if not already added
        if external_configs_path not in self.external_dataset_paths:
            self.external_dataset_paths.append(external_configs_path)

        # Reload registry
        self.get_dataset_registry(force_reload=True)

        # Load model's original config
        model_config_path = os.path.join(model_path, '.hydra', 'config.yaml')

        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config not found: {model_config_path}")

        config = OmegaConf.load(model_config_path)

        # Load datasets from external path
        datasets_dict = {}
        for dataset_name in dataset_names:
            # Try with 'external/' prefix
            external_name = f"external/{dataset_name}"
            dataset = self.dataset_registry.get(external_name)

            if dataset:
                datasets_dict[dataset_name] = dataset.to_dict()
            else:
                log.warning(f"Dataset not found: {dataset_name}")

        if not datasets_dict:
            raise ValueError(f"No datasets found in {external_configs_path}")

        # Replace datasets in config
        config.dataset = OmegaConf.create(datasets_dict)
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

    def list_external_datasets(self, format_output: bool = False) -> List[str]:
        """
        List datasets from external paths.

        Args:
            format_output: If True, print formatted output

        Returns:
            List of external dataset names (with 'external/' prefix)
        """
        registry = self.get_dataset_registry()
        all_datasets = registry.list_datasets()
        external_datasets = [d for d in all_datasets if d.startswith('external/')]

        if format_output:
            print(f"\n{'='*60}")
            print(f"External Datasets ({len(external_datasets)})")
            print(f"{'='*60}")

            for dataset_name in external_datasets:
                clean_name = dataset_name.replace('external/', '')
                dataset = registry.get(dataset_name)
                if dataset:
                    print(f"  - {clean_name}")
                    print(f"      Dataset name: {dataset.dataset_name}")
                    print(f"      Crop dir: {dataset.crop_dir}")

            print(f"{'='*60}\n")

        return external_datasets

    @staticmethod
    def from_dataset_directory(data_dir: str, dataset_name: str) -> 'ExternalConfigLoader':
        """
        Create ExternalConfigLoader from a dataset directory structure.

        Args:
            data_dir: Base data directory (e.g., "champollion/data")
            dataset_name: Dataset name (e.g., "TEST01")

        Returns:
            ExternalConfigLoader configured for this dataset

        Example:
            loader = ExternalConfigLoader.from_dataset_directory(
                data_dir="/path/to/champollion/data",
                dataset_name="TEST01"
            )
        """
        external_configs_path = os.path.join(
            data_dir,
            dataset_name,
            "derivatives",
            "champollion_V1",
            "configs"
        )

        return ExternalConfigLoader(external_dataset_paths=[external_configs_path])
