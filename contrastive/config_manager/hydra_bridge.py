#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydra Bridge

Provides compatibility layer between external configs and Hydra,
replacing the hardcoded dataset references in utils_pipelines.py functions.
"""

import os
import yaml
import random as rd
from datetime import datetime
from typing import List, Dict, Optional, Any
from omegaconf import DictConfig, OmegaConf
import logging

from .config_loader import ConfigLoader

log = logging.getLogger(__name__)


def register_omegaconf_resolvers():
    """
    Register custom OmegaConf resolvers needed by Champollion configs.

    This function registers:
    - 'now': Formats current datetime (e.g., ${now:%Y-%m-%d})
    - 'get_train_seed': Returns random seed for training
    """
    # Check if resolvers are already registered to avoid duplicate registration
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver(
            "now",
            lambda fmt: datetime.now().strftime(fmt)
        )
        log.debug("Registered 'now' OmegaConf resolver")

    if not OmegaConf.has_resolver("get_train_seed"):
        OmegaConf.register_new_resolver(
            "get_train_seed",
            lambda: rd.randint(0, 255)
        )
        log.debug("Registered 'get_train_seed' OmegaConf resolver")


class HydraBridge:
    """
    Bridge between external configuration and Hydra/OmegaConf.

    Replaces the functions in utils_pipelines.py with versions that
    use the DatasetRegistry and ConfigLoader.
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        """
        Initialize the Hydra bridge.

        Args:
            config_loader: ConfigLoader instance (creates one if None)
        """
        self.config_loader = config_loader or ConfigLoader()

    def change_config_datasets(
        self,
        config: DictConfig,
        new_datasets: List[str],
        new_datasets_root: Optional[str] = None
    ) -> None:
        """
        Replace the 'dataset' entry of a config with new target datasets.

        This is a drop-in replacement for utils_pipelines.change_config_datasets()
        that uses the DatasetRegistry instead of hardcoded paths.

        Args:
            config: OmegaConf config object
            new_datasets: List of dataset names
            new_datasets_root: Optional root path for datasets

        Works in place on the config object.
        """
        # If datasets_root is provided, use same regions as training but with new dataset
        if new_datasets_root:
            # Get current dataset regions
            current_regions = list(config.get('dataset', {}).keys())

            # Build new dataset paths
            new_datasets = [
                f"{new_datasets_root}/{region}" for region in current_regions
            ]

        # Load datasets using registry
        datasets_dict = self.config_loader.load_datasets_from_names(new_datasets)

        # Replace config.dataset
        config.dataset = OmegaConf.create(datasets_dict)

        # Update datakey
        config.datakey = list(datasets_dict.keys())

        log.info(f"Changed config datasets to: {new_datasets}")

    def change_config_label(
        self,
        config: DictConfig,
        new_label: str
    ) -> None:
        """
        Replace the 'label' entry of a config with new target label.

        Drop-in replacement for utils_pipelines.change_config_label().

        Args:
            config: OmegaConf config object
            new_label: Name of target label

        Works in place on the config object.
        """
        # Remove old label keywords if they exist
        if 'label_names' in config.keys():
            current_label = config.label_names[0]
            try:
                old_label_config = self.config_loader.load_label_config(current_label)
                for key in old_label_config.keys():
                    if key in config:
                        del config[key]
            except FileNotFoundError:
                log.warning(f"Old label config not found: {current_label}")

        # Load and apply new label config
        label_config = self.config_loader.load_label_config(new_label)
        for key, value in label_config.items():
            config[key] = value

        log.info(f"Changed config label to: {new_label}")

    def change_config_dataset_localization(
        self,
        config: DictConfig,
        new_localization: str
    ) -> None:
        """
        Replace the 'dataset_localization' entry of a config.

        Drop-in replacement for utils_pipelines.change_config_dataset_localization().

        Args:
            config: OmegaConf config object
            new_localization: Name of target localization

        Works in place on the config object.
        """
        # Load and apply localization config
        localization_config = self.config_loader.load_dataset_localization(new_localization)

        for key, value in localization_config.items():
            config[key] = value

        log.info(f"Changed config dataset localization to: {new_localization}")

    def get_save_folder_name(self, datasets: List[str], short_name: Optional[str]) -> str:
        """
        Create a folder name from dataset names or explicit option.

        Drop-in replacement for utils_pipelines.get_save_folder_name().

        Args:
            datasets: List of dataset names
            short_name: Explicit folder name (overrides dataset names)

        Returns:
            Folder name string
        """
        if short_name is not None:
            return short_name

        folder_name = '_'.join(datasets)
        return folder_name

    def list_available_datasets(
        self,
        pattern: Optional[str] = None,
        format_output: bool = False
    ) -> List[str]:
        """
        List all available datasets.

        Args:
            pattern: Optional glob pattern to filter datasets
            format_output: If True, print formatted output

        Returns:
            List of dataset names
        """
        datasets = self.config_loader.list_available_datasets(pattern)

        if format_output:
            print(f"\n{'='*60}")
            print(f"Available Datasets ({len(datasets)})")
            print(f"{'='*60}")

            # Group by directory
            grouped: Dict[str, List[str]] = {}
            for ds in datasets:
                parts = ds.split('/')
                if len(parts) > 1:
                    group = parts[0]
                    name = '/'.join(parts[1:])
                else:
                    group = 'root'
                    name = ds

                if group not in grouped:
                    grouped[group] = []
                grouped[group].append(name)

            for group in sorted(grouped.keys()):
                print(f"\n{group}/")
                for name in sorted(grouped[group]):
                    print(f"  - {name}")

            print(f"\n{'='*60}\n")

        return datasets

    def list_available_labels(self, format_output: bool = False) -> List[str]:
        """
        List all available label configurations.

        Args:
            format_output: If True, print formatted output

        Returns:
            List of label names
        """
        labels = self.config_loader.list_available_labels()

        if format_output:
            print(f"\nAvailable Labels: {', '.join(labels)}\n")

        return labels

    def list_available_classifiers(self, format_output: bool = False) -> List[str]:
        """
        List all available classifier configurations.

        Args:
            format_output: If True, print formatted output

        Returns:
            List of classifier names
        """
        classifiers = self.config_loader.list_available_classifiers()

        if format_output:
            print(f"\nAvailable Classifiers: {', '.join(classifiers)}\n")

        return classifiers

    def create_dataset_from_directory(
        self,
        dataset_name: str,
        base_dir: str,
        subject_labels_file: str,
        train_val_csv_file: str,
        region: Optional[str] = None,
        input_size: str = "(1, 18, 41, 38)",
        **kwargs
    ) -> None:
        """
        Create a new dataset configuration from a directory structure.

        Useful for quickly registering new datasets without manually creating YAML files.

        Args:
            dataset_name: Name for the dataset
            base_dir: Base directory containing the data
            subject_labels_file: Path to subject labels CSV
            train_val_csv_file: Path to train/val split CSV
            region: Optional region name (e.g., 'left', 'right')
            input_size: Input tensor size as string
            **kwargs: Additional dataset parameters
        """
        # Build standard paths from base_dir
        if region:
            suffix = f"_{region}" if not region.startswith('_') else region
            numpy_all = os.path.join(base_dir, f"{region}skeleton.npy")
            subjects_all = os.path.join(base_dir, f"{region}skeleton_subject.csv")
            foldlabel_all = os.path.join(base_dir, f"{region}label.npy")
            subjects_foldlabel_all = os.path.join(base_dir, f"{region}label_subject.csv")
            crop_dir = os.path.join(base_dir, f"{region}crops")
            foldlabel_dir = os.path.join(base_dir, f"{region}labels")
        else:
            numpy_all = os.path.join(base_dir, "skeleton.npy")
            subjects_all = os.path.join(base_dir, "skeleton_subject.csv")
            foldlabel_all = os.path.join(base_dir, "label.npy")
            subjects_foldlabel_all = os.path.join(base_dir, "label_subject.csv")
            crop_dir = os.path.join(base_dir, "crops")
            foldlabel_dir = os.path.join(base_dir, "labels")

        # Create dataset config dict
        dataset_config = {
            'dataset_name': dataset_name,
            'numpy_all': numpy_all,
            'subjects_all': subjects_all,
            'foldlabel_all': foldlabel_all,
            'subjects_foldlabel_all': subjects_foldlabel_all,
            'crop_dir': crop_dir,
            'crop_file_suffix': '_cropped_skeleton.nii.gz',
            'foldlabel_dir': foldlabel_dir,
            'train_val_csv_file': train_val_csv_file,
            'subject_labels_file': subject_labels_file,
            'subject_column_name': 'participant_id',
            'input_size': input_size,
        }

        # Add any additional parameters
        dataset_config.update(kwargs)

        # Save to appropriate location
        config_dir = os.path.join(
            self.config_loader.config_root,
            "dataset",
            "auto_generated"
        )
        os.makedirs(config_dir, exist_ok=True)

        output_path = os.path.join(config_dir, f"{dataset_name}.yaml")

        with open(output_path, 'w') as f:
            yaml.safe_dump(dataset_config, f, default_flow_style=False, sort_keys=False)

        log.info(f"Created dataset config: {output_path}")

        # Reload registry to include new dataset
        self.config_loader.get_dataset_registry(force_reload=True)
