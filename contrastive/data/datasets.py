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
Tools to create pytorch dataloaders
"""
import torch
import numpy as np

from contrastive.utils.logs import set_file_logger

from contrastive.data.transforms import transform_only_padding, transform_mixed

from contrastive.data.utils import convert_sparse_to_numpy

from contrastive.augmentations import PaddingTensor

log = set_file_logger(__file__)


def get_sample(arr, idx, type_el):
    """Returns sub-numpy torch tensors corresponding to array of indices idx.

    First axis of arr (numpy array) corresponds to subject nbs from 0 to N-1
    type_el is 'float32' for input, 'int32' for foldlabel
    """
    log.debug(f"idx (in get_sample) = {idx}")
    log.debug(f"shape of arr (in get_sample) = {arr.shape}")
    sample = arr[idx].astype(type_el)

    return torch.from_numpy(sample)


def get_filename(filenames, idx):
    """"Returns filenames corresponding to indices idx

    filenames: dataframe with column name 'ID'
    """
    filename = filenames.Subject[idx]
    log.debug(f"filenames[:5] = {filenames[:5]}")
    log.debug(f"len(filenames) = {len(filenames)}")
    log.debug(f"idx = {idx}, filename[idx] = {filename}")
    log.debug(f"{idx} in filename = {idx in filenames.index}")

    return filename


def padd_array(sample, input_size, fill_value=0):
    """Padds array according to input_size"""
    transfo = PaddingTensor(
        input_size,
        fill_value=fill_value)
    sample = transfo(sample)
    return sample


class ContrastiveDatasetFusion():
    """Custom dataset that includes image file paths.

    Applies different transformations to data depending on the type of input.
    """

    def __init__(self, filenames, config, apply_transform=True,
                 labels=None, arrays=None, foldlabel_arrays=None,
                 coords_arrays_dirs=None, skeleton_arrays_dirs=None,
                 foldlabel_arrays_dirs=None):
        """
        Every data argument is a list over regions

        Args:
            data_tensor (tensor): contains MRIs as numpy arrays
            filenames (list of strings): list of subjects' IDs
            config (Omegaconf dict): contains configuration information
        """
        self.labels=labels
        self.arrs=arrays
        self.foldlabel_arrs=foldlabel_arrays
        self.nb_train=len(filenames[0])
        self.filenames=filenames
        self.config=config
        self.transform=apply_transform
        self.coords_arrs_dirs=coords_arrays_dirs
        self.skeleton_arrs_dirs=skeleton_arrays_dirs
        self.foldlabel_arrs_dirs=foldlabel_arrays_dirs

        log.debug(f"nb_train = {self.nb_train}")
        log.debug(f"filenames[:5] = {filenames[:5]}")
        if labels is not None and labels[0].shape[0] > 0:
            label0 = labels[0]
            log.debug(f"labels[:5] = {label0[:5]}")
            log.debug(f"There are {label0[label0[config.label_names[0]].isna()].shape[0]} NaN labels")
            log.debug(label0[label0[config.label_names[0]].isna()])

    def __len__(self):
        if self.config.multiregion_single_encoder:
            if self.arrs is not None:
                return (self.nb_train*len(self.arrs))
            else:
                return (self.nb_train*len(self.coords_arrs_dirs))
        else:
            return (self.nb_train)

    def __getitem__(self, idx):
        """Returns the two views corresponding to index idx

        The two views are generated on the fly.

        Returns:
            tuple of (views, subject ID)
        """
        if torch.is_tensor(idx):
            if self.transform:
                idx = idx.tolist()
            else:
                idx = idx.tolist(self.nb_train)
        if self.config.multiregion_single_encoder:
            idx_region, idx = idx // self.nb_train, idx%self.nb_train

        # Gets data corresponding to idx
        log.debug(f"length = {self.nb_train}")
        log.debug(f"filenames = {self.filenames[0]}")
        # get filenames
        if self.config.multiregion_single_encoder:
            filename = self.filenames[idx_region]
            filenames = [get_filename(filename, idx)]
        else:
            filenames = [get_filename(filename, idx)
                        for filename in self.filenames]
        # if arrays loaded
        if self.arrs is not None and self.arrs[0] is not None:
            if self.config.multiregion_single_encoder:
                # TODO: use same code as single region, but add condition in for loop ?
                # arr = [self.arrs[idx_region]] # 1 element list
                # apply a function which loops on the list
                arr = self.arrs[idx_region]
                samples = get_sample(arr, idx, 'float32')
                samples = [padd_array(samples,
                                    self.config.data[idx_region].input_size,
                                    fill_value=0)]
            else:
                # TODO: create a build sample function
                samples = [get_sample(arr, idx, 'float32')
                        for arr in self.arrs]
                samples = [padd_array(sample,
                                    self.config.data[reg].input_size,
                                    fill_value=0)
                            for reg, sample in enumerate(samples)]

        if self.foldlabel_arrs is not None and self.foldlabel_arrs[0] is not None:
            if self.config.multiregion_single_encoder:
                foldlabel_arr = self.foldlabel_arrs[idx_region]
                sample_foldlabels = get_sample(foldlabel_arr, idx, 'int32')
                sample_foldlabels = [padd_array(sample_foldlabels,
                                                self.config.data[idx_region].input_size,
                                                fill_value=0)]
            else:
                sample_foldlabels = [get_sample(foldlabel_arr, idx, 'int32')
                                    for foldlabel_arr in self.foldlabel_arrs]
                sample_foldlabels = [padd_array(sample_foldlabel,
                                                self.config.data[reg].input_size,
                                                fill_value=0)
                                    for reg, sample_foldlabel in enumerate(sample_foldlabels)]

        
        # if path given instead
        if self.coords_arrs_dirs is not None and self.coords_arrs_dirs[0] is not None:
            if self.config.multiregion_single_encoder:
                coords_arr_dir = self.coords_arrs_dirs[idx_region][idx]
                coords_arr = np.load(coords_arr_dir)
            else:
                coords_arr_dir = [arr[idx] for arr in self.coords_arrs_dirs]
                coords_arrs = [np.load(coords_dir) for coords_dir in coords_arr_dir]
        if self.skeleton_arrs_dirs is not None and self.skeleton_arrs_dirs[0] is not None:
            # TODO: build_sample_from_path ## nb: or from sparse ?
            # and copy the architecture from above.
            if self.config.multiregion_single_encoder:
                skeleton_arr_dir = self.skeleton_arrs_dirs[idx_region][idx]
                skeleton_arr = np.load(skeleton_arr_dir)
                samples = convert_sparse_to_numpy(skeleton_arr, coords_arr,
                                                  self.config.data[idx_region].input_size[1:], 'float32')
                samples = torch.from_numpy(samples)
                samples = [padd_array(samples,
                                    self.config.data[idx_region].input_size,
                                    fill_value=0)]
            else:
                skeleton_arr_dir = [arr[idx] for arr in self.skeleton_arrs_dirs]
                skeleton_arrs = [np.load(skeleton_dir) for skeleton_dir in skeleton_arr_dir]
                samples = [convert_sparse_to_numpy(skeleton_arr, coords_arr,
                                                  self.config.data[reg].input_size[1:], 'float32')
                                                  for reg, (skeleton_arr, coords_arr)
                                                  in enumerate(zip(skeleton_arrs, coords_arrs))]
                samples = [torch.from_numpy(sample) for sample in samples]
                samples = [padd_array(sample,
                                    self.config.data[reg].input_size,
                                    fill_value=0)
                           for reg, sample in enumerate(samples)]
        if self.foldlabel_arrs_dirs is not None and self.foldlabel_arrs_dirs[0] is not None:
            if self.config.multiregion_single_encoder:
                foldlabel_arr_dir = self.foldlabel_arrs_dirs[idx_region][idx]
                foldlabel_arr = np.load(foldlabel_arr_dir)
                sample_foldlabels = convert_sparse_to_numpy(foldlabel_arr, coords_arr,
                                                            self.config.data[idx_region].input_size[1:], 'int32')
                sample_foldlabels = torch.from_numpy(sample_foldlabels)
                sample_foldlabels = [padd_array(sample_foldlabels,
                                    self.config.data[idx_region].input_size,
                                    fill_value=0)]
            else:
                foldlabel_arr_dir = [arr[idx] for arr in self.foldlabel_arrs_dirs]
                foldlabel_arrs = [np.load(foldlabel_dir) for foldlabel_dir in foldlabel_arr_dir]
                sample_foldlabels = [convert_sparse_to_numpy(foldlabel_arr, coords_arr,
                                                  self.config.data[reg].input_size[1:], 'int32')
                                                  for reg, (foldlabel_arr, coords_arr)
                                                  in enumerate(zip(foldlabel_arrs, coords_arrs))]
                sample_foldlabels = [torch.from_numpy(sample_foldlabel) for sample_foldlabel in sample_foldlabels]
                sample_foldlabels = [padd_array(sample_foldlabel,
                                    self.config.data[reg].input_size,
                                    fill_value=0)
                           for reg, sample_foldlabel in enumerate(sample_foldlabels)]

        self.transform1 = []
        self.transform2 = []

        if self.config.multiregion_single_encoder:
            regs = [0]
            input_sizes = [self.config.data[idx_region].input_size]
            mask_paths = [self.config.data[idx_region].mask_path]
            flips = [self.config.data[idx_region].flip_dataset]
        else:
            regs = range(len(filenames))
            input_sizes = [self.config.data[reg].input_size for reg in regs]
            cutout_mask_paths = [self.config.data[reg].cutout_mask_path for reg in regs]
            cutin_mask_paths = [self.config.data[reg].cutin_mask_path for reg in regs]
            flips = [self.config.data[reg].flip_dataset for reg in regs]
        # compute the transforms
        for reg, cutout_mask_path, cutin_mask_path, input_size, flip in zip(regs, cutout_mask_paths, cutin_mask_paths, input_sizes, flips):
            if self.transform:
                transform1 = transform_mixed(
                    sample_foldlabels[reg],
                    cutin_mask_path=cutin_mask_path,
                    input_size=input_size,
                    config=self.config)
                transform2 = transform_mixed(
                    sample_foldlabels[reg],
                    cutin_mask_path=cutin_mask_path,
                    input_size=input_size,
                    config=self.config)        
            else:
                transform1 = transform_only_padding(
                    input_size, self.config)
                transform2 = transform_only_padding(
                    input_size, self.config)
            self.transform1.append(transform1)
            self.transform2.append(transform2)

        # Computes the views
        view1 = []
        view2 = []
        for reg in range(len(filenames)):
            view1.append(self.transform1[reg](samples[reg]))
            view2.append(self.transform2[reg](samples[reg]))

        # Computes the outputs as tuples
        concatenated_tuple = ()
        # loop over input datasets
        for reg in range(len(filenames)):
            views = torch.stack((view1[reg], view2[reg]), dim=0)
            if self.config.multiregion_single_encoder or \
                self.config.multiple_projection_heads:
                tuple_with_path = ((views, filenames[reg], idx_region),)
            else:
                tuple_with_path = ((views, filenames[reg]),)
            concatenated_tuple += tuple_with_path

        return concatenated_tuple
