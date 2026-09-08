#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
"""
Tools to create datasets
"""

import pandas as pd
import numpy as np
import os
import copy

# only if foldlabel == True
try:
    from deep_folding.brainvisa.utils.save_data import quality_checks
    from deep_folding.brainvisa.utils.save_data import compare_array_aims_files
except ImportError:
    print("INFO: you cannot use deep_folding in brainvisa. Probably OK.")

from ..utils.logs import set_file_logger, set_root_logger_level

from .datasets import ContrastiveDatasetFusion

from .utils import extract_data, extract_train_and_val_subjects

import logging

log = set_file_logger(__file__)
root = logging.getLogger()


def create_sets_without_labels_without_load(config):
    """
    Create train / val / train-val / test sets when using individual directories
    and sparse matrices.
    Requires for coord_all in dataset config additionnaly to every modality.
    """

    for reg in range(len(config.data)):
        if 'coords_all' not in config.data[reg].keys():
            raise ValueError("load_sparse requires coords_all in dataset config")
        if not (
            os.path.isdir(config.data[reg].coords_all) and os.path.isdir(config.data[reg].numpy_all) \
            and os.path.isdir(config.data[reg].foldlabel_all)
            ):
            raise ValueError("load_sparse requires numpy directories to be folders, not files")
        
    sub_dirs = {'filenames': [],
                'coords_dirs': [],
                'skeleton_dirs': [],
                'foldlabel_dirs': []}
        
    dirs = {'train': copy.deepcopy(sub_dirs),
            'val': copy.deepcopy(sub_dirs),
            'train_val': copy.deepcopy(sub_dirs),
            'test': copy.deepcopy(sub_dirs)}

    for reg in range(len(config.data)):
        subjects_all = pd.read_csv(config.data[reg].subjects_all)
        # split subjects in train/val/train-val/test
        if 'train_csv_file' in config.data[reg].keys() and 'val_csv_file' in config.data[reg].keys():
            train_subjects = pd.read_csv(config.data[reg]['train_csv_file'], names=['Subject'])
            val_subjects = pd.read_csv(config.data[reg]['val_csv_file'], names=['Subject'])
            train_val_subjects = pd.concat((train_subjects, val_subjects), ignore_index=True)
        elif 'train_val_csv_file' in config.data[reg].keys():
            train_val_subjects = pd.read_csv(config.data[reg]['train_val_csv_file'], names=['Subject'])
            train_subjects, val_subjects = \
                extract_train_and_val_subjects(
                    train_val_subjects, config.partition, config.seed)
        if 'test_csv_file' in config.data[reg].keys():
            test_subjects = pd.read_csv(config.data[reg]['test_csv_file'], names=['Subject'])
        else:
            test_subjects = subjects_all.sample(1) # need not to be empty

        dirs['train']['filenames'].append(train_subjects.reset_index(drop=True))
        dirs['val']['filenames'].append(val_subjects.reset_index(drop=True))
        dirs['train_val']['filenames'].append(train_val_subjects.reset_index(drop=True))
        dirs['test']['filenames'].append(test_subjects.reset_index(drop=True))
        
        for subset in dirs.keys():
            # coords
            coords_dir = config.data[reg].coords_all
            coords_dirs = np.array([os.path.join(coords_dir,f'{sub}_coords.npy') for sub in dirs[subset]['filenames'][reg].Subject])
            #coords_dirs = np.expand_dims(coords_dirs, axis=-1)
            dirs[subset]['coords_dirs'].append(coords_dirs)
            # skels
            skels_dir = config.data[reg].numpy_all
            skeleton_dirs = np.array([os.path.join(skels_dir,f'{sub}_skeleton_values.npy') for sub in dirs[subset]['filenames'][reg].Subject])
            #skeleton_dirs = np.expand_dims(skeleton_dirs, axis=-1)
            dirs[subset]['skeleton_dirs'].append(skeleton_dirs)
            # foldlabels
            foldlabel_dir = config.data[reg].foldlabel_all
            foldlabel_dirs = np.array([os.path.join(foldlabel_dir,f'{sub}_foldlabel_values.npy') for sub in dirs[subset]['filenames'][reg].Subject])
            #foldlabel_dirs = np.expand_dims(foldlabel_dirs, axis=-1)
            dirs[subset]['foldlabel_dirs'].append(foldlabel_dirs)

    datasets = {}

    for subset_name in dirs.keys():

        datasets[subset_name] = ContrastiveDatasetFusion(
            filenames=dirs[subset_name]['filenames'], # quelle forme pd ?
            coords_arrays_dirs=dirs[subset_name]['coords_dirs'],
            skeleton_arrays_dirs=dirs[subset_name]['skeleton_dirs'],
            foldlabel_arrays_dirs=dirs[subset_name]['foldlabel_dirs'],
            config=config,
            apply_transform=config.apply_augmentations)
        
    return datasets

def create_sets_without_labels(config):
    """Creates train, validation and test sets

    Args:
        config (Omegaconf dict): contains configuration parameters
    Returns:
        train_dataset, val_dataset, test_datasetset, train_val_dataset (tuple)
    """

    skeleton_all = []
    foldlabel_all = []

    for reg in range(len(config.data)):
        # Loads and separates in train_val/test skeleton crops
        skeleton_output = extract_data(
            config.data[reg].numpy_all,
            config.data[reg].crop_dir, config, reg)
        skeleton_all.append(skeleton_output)

        # Loads and separates in train_val/test set foldlabels if requested
        if config.apply_augmentations and (config.foldlabel or config.trimdepth
                                           or config.random_choice or config.mixed):
            foldlabel_output = extract_data(config.data[reg].foldlabel_all,
                                    config.data[reg].crop_dir,
                                    config, reg)
        else:
            foldlabel_output = None
            log.info("foldlabel data NOT requested. Foldlabel data NOT loaded")
        
        foldlabel_all.append(foldlabel_output)          

    # Creates the dataset from these data by doing some preprocessing
    datasets = {}
    for subset_name in skeleton_all[0].keys():
        log.debug(subset_name)
        # Concatenates filenames
        filenames = [skeleton_output[subset_name][0]
                     for skeleton_output in skeleton_all]
        # Concatenates arrays
        arrays = [skeleton_output[subset_name][1]
                  for skeleton_output in skeleton_all]

        # Concatenates foldabel arrays
        foldlabel_arrays = []
        for foldlabel_output in foldlabel_all:
            # select the augmentation method
            if config.apply_augmentations:
                if config.trimdepth or config.random_choice or config.mixed or config.foldlabel:  # branch_clipping
                    foldlabel_array = foldlabel_output[subset_name][1]
                else:  # cutout
                    foldlabel_array = None  # no need of fold labels
            else:  # no augmentation
                foldlabel_array = None
            foldlabel_arrays.append(foldlabel_array)

        datasets[subset_name] = ContrastiveDatasetFusion(
            filenames=filenames,
            arrays=arrays,
            foldlabel_arrays=foldlabel_arrays,
            config=config,
            apply_transform=config.apply_augmentations)

    return datasets
