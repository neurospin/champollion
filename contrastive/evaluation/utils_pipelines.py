# File that contains auxiliary functions needed for either (or both) embeddings_pipeline
# and supervised_pipeline
#
# UPDATED: Now uses Config Manager for dynamic dataset loading instead of hardcoded paths

import os
import yaml
import re
import pandas as pd
from typing import List, Optional

from contrastive.utils.logs import set_root_logger_level, set_file_logger
from contrastive.config_manager import HydraBridge, ConfigLoader

log = set_file_logger(__file__)

# Create global bridge instance for config management
_bridge = None
_current_config_root = None


def _get_bridge(config_root: Optional[str] = None) -> HydraBridge:
    """Get or create HydraBridge instance.

    Args:
        config_root: Optional path to config directory. If provided and
                     different from current, creates a new bridge instance.
    """
    global _bridge, _current_config_root

    # If config_root is provided and different, create new bridge
    if config_root is not None and config_root != _current_config_root:
        _bridge = HydraBridge(ConfigLoader(config_root=config_root))
        _current_config_root = config_root
    elif _bridge is None:
        _bridge = HydraBridge()
        _current_config_root = None

    return _bridge


# ====================== Config Management Functions ======================
# These functions now use the Config Manager for dynamic dataset loading

def get_save_folder_name(datasets: List[str], short_name: Optional[str]) -> str:
    """Creates a file name from the names of the target datasets or an
    explicit option.

    Arguments:
        - datasets: list of str. Contains the names of the target datasets.
        - short_name: str or None. Default answer by the algorithm if not None.

    Returns:
        - folder_name: str. The generated folder name.
    """
    bridge = _get_bridge()
    return bridge.get_save_folder_name(datasets, short_name)


def change_config_datasets(
    config,
    new_datasets: List[str],
    new_datasets_root: Optional[str],
    config_root: Optional[str] = None
):
    """Replace the 'dataset' entry of a config with the new target datasets.

    NOW USES CONFIG MANAGER: Loads datasets dynamically from DatasetRegistry
    instead of using hardcoded paths.

    Arguments:
        - config: a config object (omegaconf).
        - new_datasets: list of str, each corresponding to the name
          of a target yaml file or dataset in the registry.
        - new_datasets_root: if not None, takes the same regions as for training
          but with the new dataset.
        - config_root: Optional path to dataset config directory.

    Works in place on the config object.
    """
    bridge = _get_bridge(config_root)
    bridge.change_config_datasets(config, new_datasets, new_datasets_root)


def change_config_label(config, new_label: str):
    """Replace the 'label' entry of a config with the new target label.

    NOW USES CONFIG MANAGER: Loads label config from ConfigLoader.

    Arguments:
        - config: a config object (omegaconf).
        - new_label: str corresponding to the name of a target yaml file.

    Works in place on the config object.
    """
    bridge = _get_bridge()
    bridge.change_config_label(config, new_label)


def change_config_dataset_localization(config, new_localization: str):
    """Replace the 'dataset_localization' entry of a config with the new
    target dataset localization.

    NOW USES CONFIG MANAGER: Loads localization config from ConfigLoader.

    Arguments:
        - config: a config object (omegaconf).
        - new_localization: str corresponding to the name of a target yaml file.

    Works in place on the config object.
    """
    bridge = _get_bridge()
    bridge.change_config_dataset_localization(config, new_localization)


def save_used_datasets(save_path, datasets):
    """Save the datasets given in order in a .txt file. Used in embeddings and supervised
    pipelines to know which datasets have been used for the results generation.
    
    Arguments:
        - save_path: str. Where the txt file is saved. Either the name 
        of the directory or directly the full path with the file name.
        - datasets: list of str. Name of the used datasets."""
    # if save path is only a directory
    if os.path.isdir(save_path):
        # add the actual file name at the end
        save_path = os.path.join(save_path, 'datasets_used.txt')
    
    # preprocess datasets
    datasets = list(datasets)

    with open(save_path, 'w') as file:
        for dataset in datasets:
            file.write(dataset)
            file.write('\n')


def save_used_label(save_path, config):
    """Save the label used for classification in a .txt file. Used both in 
    supervised and embeddings pipelines.

    Arguments:
        - save_path: str. Where the txt file is saved. Either the name 
        of the directory or directly the full path with the file name.
        - config: omegaconf object. Contains the label used for test 
        classification."""
    
    # if save path is only a directory
    if os.path.isdir(save_path):
        # add the actual file name at the end
        save_path = os.path.join(save_path, 'label_used.txt')
    
    # get label from config
    label = config.label_names[0]

    with open(save_path, 'w') as file:
        file.write(label)


def detect_collision(run_path):
    """Detects if two models have been saved in the same folder during
    a wandb grid search. Returns True if it is the case.
    
    Arguments:
        - run_path: folder associated to a model to be inspected."""
    log_path = os.path.join(run_path, 'wandb')
    try:
        files = os.listdir(log_path)
    except:
        # not a model
        return False
    count = 0
    for file in files:
        if re.match(r'run*', file) is not None:
            count+=1
    if count > 1:
        return True
    return False


def detect_collisions(sweep_path):
    """Loops detect_collision over a folder at sweep_path.
    Prints all the folder names where there is a collision (two models
    saved in the same folder)."""
    runs = os.listdir(sweep_path)
    for run in runs:
        run_path = os.path.join(sweep_path, run)
        if os.path.isdir(run_path):
            if detect_collision(run_path):
                print(run)


def save_outputs_as_csv(outputs, filenames, labels, csv_path=None, verbose=False):
    """Save and returns outputs of a model to its canonical form from a tensor. If
    the given save_path doesn't exist, creates it.
    
    Arguments:
        - outputs: the output tensor to save.
        - filenames: the ordered subjects names associated to the outputs.
        - labels: the ordered true labels associated to the outputs.
        - csv_path: the path where to save the csv. If None, only returns
        the pandas dataframe.
        - verbose: verbose."""
    columns_names = ['dim'+str(i+1) for i in range(outputs.shape[1])]
    values = pd.DataFrame(outputs.numpy(), columns=columns_names)
    labels = pd.DataFrame(labels, columns=['labels']).astype(int)
    filenames = pd.DataFrame(filenames, columns=['ID'])
    df_outputs = pd.concat([labels, values, filenames], axis=1)
    
    # remove one copy each ID
    df_outputs = df_outputs.groupby('ID').mean()
    df_outputs.labels = df_outputs.labels.astype(int)

    if verbose:
        print("outputs:", df_outputs.iloc[:10, :])
        print("nb of elements:", df_outputs.shape[0])

    # Solves the case in which index type is tensor
    if len(df_outputs.index) > 0:  # avoid cases where empty df
        if type(df_outputs.index[0]) != str:
            index = [idx.item() for idx in df_outputs.index]
            index_name = df_outputs.index.name
            df_outputs.index = index
            df_outputs.index.names = [index_name]

    if csv_path:
        df_outputs.to_csv(csv_path)

    return df_outputs

# from contextlib import contextmanager
# import subprocess
# import getpass

# @contextmanager
# def sshfs_mount(remote_path, local_mount_point, ssh_user, ssh_host):
#     """Context manager for SSHFS mounting."""
#     os.makedirs(local_mount_point, exist_ok=True)
#     password = getpass.getpass(f"Enter SSH password for {ssh_user}@{ssh_host}: ")
#     mount_cmd = f"sshpass -p '{password}' sshfs {ssh_user}@{ssh_host}:{remote_path} {local_mount_point}"
#     unmount_cmd = f"fusermount -u {local_mount_point}"

#     try:
#         subprocess.run(mount_cmd, shell=True, check=True)
#         yield
#     finally:
#         subprocess.run(unmount_cmd, shell=True, check=True)
