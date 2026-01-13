#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Registry System

Provides a registry for managing dataset configurations dynamically,
avoiding hardcoded dataset paths in the codebase.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

log = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    dataset_name: str
    numpy_all: str
    subjects_all: str
    foldlabel_all: str
    subjects_foldlabel_all: str
    crop_dir: str
    crop_file_suffix: str
    foldlabel_dir: str
    train_val_csv_file: str
    subject_labels_file: str
    subject_column_name: str = 'participant_id'
    input_size: str = "(1, 18, 41, 38)"

    # Optional fields
    test_intra_csv_file: Optional[str] = None
    test_csv_file: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str, name: Optional[str] = None) -> 'DatasetConfig':
        """Load dataset config from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if name is None:
            name = Path(yaml_path).stem

        # Extract known fields
        known_fields = {
            'name': name,
            'dataset_name': data.get('dataset_name', name),
            'numpy_all': data.get('numpy_all'),
            'subjects_all': data.get('subjects_all'),
            'foldlabel_all': data.get('foldlabel_all'),
            'subjects_foldlabel_all': data.get('subjects_foldlabel_all'),
            'crop_dir': data.get('crop_dir'),
            'crop_file_suffix': data.get('crop_file_suffix', '_cropped_skeleton.nii.gz'),
            'foldlabel_dir': data.get('foldlabel_dir'),
            'train_val_csv_file': data.get('train_val_csv_file'),
            'subject_labels_file': data.get('subject_labels_file'),
            'subject_column_name': data.get('subject_column_name', 'participant_id'),
            'input_size': data.get('input_size', "(1, 18, 41, 38)"),
            'test_intra_csv_file': data.get('test_intra_csv_file'),
            'test_csv_file': data.get('test_csv_file'),
        }

        # Extract additional params not in known fields
        additional_params = {k: v for k, v in data.items() if k not in known_fields}
        known_fields['additional_params'] = additional_params

        return cls(**known_fields)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Hydra/OmegaConf."""
        result = {
            'dataset_name': self.dataset_name,
            'numpy_all': self.numpy_all,
            'subjects_all': self.subjects_all,
            'foldlabel_all': self.foldlabel_all,
            'subjects_foldlabel_all': self.subjects_foldlabel_all,
            'crop_dir': self.crop_dir,
            'crop_file_suffix': self.crop_file_suffix,
            'foldlabel_dir': self.foldlabel_dir,
            'train_val_csv_file': self.train_val_csv_file,
            'subject_labels_file': self.subject_labels_file,
            'subject_column_name': self.subject_column_name,
            'input_size': self.input_size,
        }

        # Add optional fields
        if self.test_intra_csv_file:
            result['test_intra_csv_file'] = self.test_intra_csv_file
        if self.test_csv_file:
            result['test_csv_file'] = self.test_csv_file

        # Add additional params
        result.update(self.additional_params)

        return result

    def save_yaml(self, output_path: str):
        """Save dataset config to YAML file."""
        with open(output_path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


class DatasetRegistry:
    """
    Registry for managing dataset configurations.

    Supports multiple storage backends:
    - YAML files in a directory
    - JSON configuration
    - Runtime registration
    """

    def __init__(self, config_dir: Optional[str] = None, auto_discover: bool = True):
        """
        Initialize the dataset registry.

        Args:
            config_dir: Directory containing dataset YAML files
            auto_discover: Automatically discover datasets in config_dir
        """
        self.datasets: Dict[str, DatasetConfig] = {}
        self.config_dir = config_dir

        if config_dir and auto_discover:
            self.discover_datasets(config_dir)

    def discover_datasets(self, config_dir: str, recursive: bool = True):
        """
        Discover and register all datasets in a directory.

        Args:
            config_dir: Directory to search for dataset YAML files
            recursive: Search subdirectories recursively
        """
        config_path = Path(config_dir)

        if not config_path.exists():
            log.warning(f"Config directory does not exist: {config_dir}")
            return

        # Find all YAML files
        pattern = "**/*.yaml" if recursive else "*.yaml"
        yaml_files = list(config_path.glob(pattern))

        log.info(f"Discovering datasets in {config_dir}")
        log.info(f"Found {len(yaml_files)} YAML files")

        for yaml_file in yaml_files:
            try:
                # Generate dataset name from path relative to config_dir
                rel_path = yaml_file.relative_to(config_path)
                # Remove .yaml extension and convert path separators to dots
                dataset_name = str(rel_path.with_suffix('')).replace(os.sep, '/')

                dataset = DatasetConfig.from_yaml(str(yaml_file), name=dataset_name)
                self.register(dataset_name, dataset)
                log.debug(f"Registered dataset: {dataset_name}")
            except Exception as e:
                log.warning(f"Failed to load dataset from {yaml_file}: {e}")

    def register(self, name: str, dataset: DatasetConfig):
        """Register a dataset configuration."""
        self.datasets[name] = dataset
        log.debug(f"Registered dataset: {name}")

    def get(self, name: str) -> Optional[DatasetConfig]:
        """Get a dataset configuration by name."""
        return self.datasets.get(name)

    def get_multiple(self, names: List[str]) -> Dict[str, DatasetConfig]:
        """Get multiple dataset configurations."""
        return {name: self.get(name) for name in names if self.get(name) is not None}

    def list_datasets(self) -> List[str]:
        """List all registered dataset names."""
        return list(self.datasets.keys())

    def filter_datasets(self, pattern: str) -> List[str]:
        """
        Filter datasets by name pattern.

        Args:
            pattern: Glob-style pattern (e.g., 'hcp*', '*/left', 'gridsearch_2024/*')
        """
        from fnmatch import fnmatch
        return [name for name in self.datasets.keys() if fnmatch(name, pattern)]

    def load_from_yaml_list(self, yaml_path: str, key: str = 'datasets'):
        """
        Load dataset list from a YAML file.

        Useful for defining dataset collections for experiments.

        Args:
            yaml_path: Path to YAML file containing dataset list
            key: Key in YAML file containing the dataset list
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        dataset_names = data.get(key, [])

        for name in dataset_names:
            if name not in self.datasets:
                log.warning(f"Dataset {name} not found in registry")

    def create_hydra_dataset_dict(self, dataset_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Create a dictionary suitable for Hydra config's dataset section.

        Args:
            dataset_names: List of dataset names to include

        Returns:
            Dictionary mapping dataset names to their configurations
        """
        result = {}
        for name in dataset_names:
            dataset = self.get(name)
            if dataset:
                result[name] = dataset.to_dict()
            else:
                log.warning(f"Dataset {name} not found in registry")

        return result

    def create_dataset_from_template(
        self,
        name: str,
        template_name: str,
        overrides: Dict[str, Any]
    ) -> DatasetConfig:
        """
        Create a new dataset based on a template with overrides.

        Args:
            name: Name for the new dataset
            template_name: Name of template dataset to copy from
            overrides: Dictionary of values to override

        Returns:
            New DatasetConfig instance
        """
        template = self.get(template_name)
        if not template:
            raise ValueError(f"Template dataset {template_name} not found")

        # Convert to dict, apply overrides, create new instance
        config_dict = template.to_dict()
        config_dict.update(overrides)
        config_dict['name'] = name

        # Create new dataset config
        new_dataset = DatasetConfig(
            name=name,
            dataset_name=config_dict.get('dataset_name', name),
            **{k: v for k, v in config_dict.items() if k not in ['name', 'dataset_name']}
        )

        self.register(name, new_dataset)
        return new_dataset

    def export_dataset_list(self, output_path: str, dataset_names: Optional[List[str]] = None):
        """
        Export a list of datasets to a YAML file.

        Args:
            output_path: Path to output YAML file
            dataset_names: List of datasets to export (None = all)
        """
        if dataset_names is None:
            dataset_names = self.list_datasets()

        data = {
            'datasets': dataset_names,
            'dataset_configs': self.create_hydra_dataset_dict(dataset_names)
        }

        with open(output_path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        log.info(f"Exported {len(dataset_names)} datasets to {output_path}")
