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
# knowledge of the CeCILL license version 2 and that you ac8
import numbers
from collections import namedtuple

import os
import numpy as np
import torch
from scipy.ndimage import rotate
from sklearn.preprocessing import OneHotEncoder

from contrastive.utils import logs
from contrastive.data.utils import zero_padding, repeat_padding, pad, convert_sparse_to_numpy

log = logs.set_file_logger(__file__)


def rotate_list(l_list):
    "Rotates list by -1"
    return l_list[1:] + l_list[:1]


def mask_array_with_skeleton(array, skeleton, cval):
    array[skeleton==0]=cval
    return array


class PaddingTensor(object):
    """A class to pad a tensor"""

    def __init__(self, shape, nb_channels=1, fill_value=0):
        """ Initialize the instance.
        Parameters
        ----------
        shape: list of int
            the desired shape.
        nb_channels: int, default 1
            the number of channels.
        fill_value: int or list of int, default 0
            the value used to fill the array, if a list is given, use the
            specified value on each channel.
        """
        self.shape = rotate_list(shape)
        self.nb_channels = nb_channels
        self.fill_value = fill_value
        if self.nb_channels > 1 and not isinstance(self.fill_value, list):
            self.fill_value = [self.fill_value] * self.nb_channels
        elif isinstance(self.fill_value, list):
            assert len(self.fill_value) == self.nb_channels()

    def __call__(self, tensor):
        """ Fill a tensor to fit the desired shape.
        Parameters
        ----------
        tensor: torch.tensor
            an input tensor.
        Returns
        -------
        fill_tensor: torch.tensor
            the fill_value padded tensor.
        """
        if len(tensor.shape) - len(self.shape) == 1:
            data = []
            for _tensor, _fill_value in zip(tensor, self.fill_value):
                data.append(self._apply_padding(_tensor, _fill_value))
            return torch.from_numpy(np.asarray(data))
        elif len(tensor.shape) - len(self.shape) == 0:
            return self._apply_padding(tensor, self.fill_value)
        else:
            raise ValueError("Wrong input shape specified!")

    def _apply_padding(self, tensor, fill_value):
        """ See Padding.__call__().
        """
        arr = tensor.numpy()
        orig_shape = arr.shape
        padding = []
        # print(f"SHAPES: {orig_shape} - {self.shape}")
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append((half_shape_i, half_shape_i))
            else:
                padding.append((half_shape_i, half_shape_i + 1))
        for cnt in range(len(arr.shape) - len(padding)):
            padding.append((0, 0))

        fill_arr = np.pad(arr, padding, mode="constant",
                          constant_values=fill_value)

        # fill_arr = np.reshape(fill_arr, (1,) + fill_arr.shape[:-1])

        return torch.from_numpy(fill_arr)


class EndTensor(object):
    """Puts all internal and external values to background value 0
    """

    def __init__(self):
        None

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr = np.reshape(arr, (1,) + arr.shape[:-1])
        return torch.from_numpy(arr)


class SimplifyTensor(object):
    """Puts all internal and external values to background value 0
    """

    def __init__(self):
        None

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr[arr == 11] = 0
        return torch.from_numpy(arr)


class BinarizeTensor(object):
    """Puts non-zero values to 1
    """

    def __init__(self):
        None

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr[arr > 0] = 1
        return torch.from_numpy(arr)
    

class ReduceTensor(object):
    """Keep only the first modality (skeleton)
    """

    def __init__(self):
        None

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr_skel = arr[0]
        return torch.from_numpy(arr_skel)
    

class ConcatTensor(object):
    """Concat skeleton with foldlabel.
    """

    def __init__(self, sample_foldlabel, mask):
        self.sample_foldlabel = sample_foldlabel
        self.mask = mask

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr_foldlabel = self.sample_foldlabel.numpy()
        arr_mask = self.mask
        arr_concat = np.stack((arr, arr_foldlabel, arr_mask))

        return torch.from_numpy(arr_concat)
    

class RotateTensor(object):
    """Apply a random rotation on the images
    """

    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, tensor):
        arr = tensor.numpy()
        log.debug("Shapes before rotation", tensor.shape, arr.shape)
        rot_array = np.copy(arr)

        for axes in (1, 2), (1, 3), (2, 3):
            np.random.seed()
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            log.debug(axes, angle)
            rot_array = rotate(rot_array,
                               angle=angle,
                               axes=axes,
                               order=0,
                               reshape=False,
                               mode='constant',
                               cval=0)

        #rot_array = np.expand_dims(rot_array[..., 0], axis=0)

        log.debug("Values in the array after rotation", np.unique(rot_array))

        return torch.from_numpy(rot_array)


class PartialCutOutTensor_Roll(object):
    """Apply a rolling cutout on the images and puts only bottom value
    inside the cutout
    cf. Improved Regularization of Convolutional Neural Networks with Cutout,
    arXiv, 2017
    We assume that the rectangle to be cut is inside the image.
    """

    def __init__(self, mask_constraint=False, from_skeleton=True, input_size=None,
                 keep_extremity='bottom', keep_proba_per_branch=1., keep_proba_global=1., patch_size=None,
                 random_size=False, localization=None):
        """[summary]
        If from_skeleton==True,
            takes skeleton image, cuts it out and fills with bottom_only image
        If from_skeleton==False,
            takes bottom_only image, cuts it out and fills with skeleton image
        Args:
            mask (bool array): the mask of the ROI limits where
                the center of the cutout can be.
            mask_constraint (bool): whether mask is used or not.
            from_skeleton (bool, optional): Defaults to True.
            patch_size (either int or list of int): Defaults to None.
                if int, percentage of the volume to cutout.
                Defines a rectangle with same proportions as input.
            random_size (bool, optional): Defaults to False.
            inplace (bool, optional): Defaults to False.
            localization ([type], optional): Defaults to None.
        """

        if isinstance(patch_size, int):
            self.patch_size = patch_size
        elif len(patch_size)==2: # a range is given, select a random size in the range
            self.patch_size = np.random.randint(low=patch_size[0], high=patch_size[1])
        else: # a crop size is given
            self.patch_size = rotate_list(patch_size)
        self.input_size = input_size
        self.random_size = random_size
        self.localization = localization
        self.from_skeleton = from_skeleton
        self.mask_constraint = mask_constraint
        self.keep_proba_per_branch = keep_proba_per_branch
        self.keep_proba_global = keep_proba_global
        if keep_extremity=='random':
            np.random.seed()
            r = np.random.randint(3)
            if r == 0:
                self.keep_extremity='top'
            elif r==1:
                self.keep_extremity='bottom'
            else:
                self.keep_extremity=None
        else:
            np.random.seed()
            r = np.random.uniform()
            # don't keep bottom/top with given probability
            if r > self.keep_proba_global:
                keep_extremity=None
            self.keep_extremity = keep_extremity

    def __call__(self, tensor):

        arr_all = tensor.numpy()
        arr = arr_all[0]
        arr_foldlabel = arr_all[1]
        arr_foldlabel = mask_array_with_skeleton(arr_foldlabel, arr, cval=0)

        img_shape = np.array(arr.shape)
        if isinstance(self.patch_size, int):
            proportion = (1/100*self.patch_size)**(1/(len(img_shape)-1))
            size = rotate_list(self.input_size)
            size = proportion*np.array(size)
            size = np.round(size).astype(int)
            size[-1]=1
            
        else:
            size = np.copy(self.patch_size)
        assert len(size) == len(img_shape), f"Incorrect patch dimension : {size}"
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.random_size:
                size[ndim] = np.random.randint(0, size[ndim])
        if self.localization is not None:
            start_cutout = []
            for ndim in range(len(img_shape)):
                delta_before = max(
                    self.localization[ndim] - size[ndim] // 2, 0)
                start_cutout.append(delta_before)
        else:
            np.random.seed()
            if self.mask_constraint:
                #boolean = True
                # loop until the center of the crop is inside the mask
                #while boolean or not self.mask[tuple(middle_cutout)]:
                #    boolean = False
                #    start_cutout = []
                #    middle_cutout = []
                #    for ndim in range(len(img_shape)):
                #        delta_before = np.random.randint(0, img_shape[ndim])
                #        start_cutout.append(delta_before)
                        # define middle of cutout, taking roll into account
                #        middle_pos = int((delta_before + size[ndim] // 2)%img_shape[ndim])
                #        middle_cutout.append(middle_pos)
                # alt : use mask as proba sampling # TODO : implement properly and distinguish cutin and cutout
                # normalize the mask
                mask = arr_all[4]
                mask = mask / np.sum(mask)
                i = np.random.choice(np.arange(mask.size), p=mask.ravel())
                middle_pos = np.unravel_index(i, mask.shape)
                start_cutout = [(middle_pos[ndim] - size[ndim] // 2)%img_shape[ndim] for ndim in range(len(img_shape))]
            else:
                start_cutout = []
                for ndim in range(len(img_shape)):
                    delta_before = np.random.randint(0, img_shape[ndim])
                    start_cutout.append(delta_before)

        # Creates rolling mask cutout
        mask_roll = np.zeros(img_shape).astype('float32')

        indexes = []
        for ndim in range(len(img_shape)):
            indexes.append(slice(0, int(size[ndim])))
        mask_roll[tuple(indexes)] = 1

        for ndim in range(len(img_shape)):
            mask_roll = np.roll(mask_roll, start_cutout[ndim], axis=ndim)

        # Determines part of the array inside and outside the cutout
        arr_inside = arr * mask_roll
        arr_outside = arr * (1 - mask_roll)

        if self.keep_proba_per_branch < 1.:
            # keep bottom with proba p for each branch
            indexed_branches = np.mod(arr_foldlabel,
                            np.full(arr_foldlabel.shape, fill_value=1000))
            indexes =  np.unique(indexed_branches)
            assert (len(indexes)>1), 'No branch in foldlabel'
            indexes = indexes[1:] # remove background
            select = np.random.rand(indexes.size) < self.keep_proba_per_branch
            selected_indexes = indexes[select]
            selected_branches = np.isin(indexed_branches, selected_indexes)
            #print(np.sum(selected_branches!=0), np.sum(arr_foldlabel!=0), np.sum(arr!=0), np.sum(np.logical_and(arr!=0, selected_branches!=0)) / np.sum(arr!=0))

        # If self.from_skeleton == True:
        # This keeps the whole skeleton outside the cutout
        # and keeps only bottom value inside the cutout
        if self.from_skeleton:
            if self.keep_extremity=='top':
                arr_inside = arr_inside * (arr_inside == 35)
            elif self.keep_extremity=='bottom':
                arr_inside = arr_inside * (arr_inside == 30)
            elif self.keep_extremity=='bottom_top':
                arr_inside = arr_inside * (np.logical_or(arr_inside == 30, arr_inside == 35))
            elif self.keep_extremity=='all': # protect whole branch !
                arr_inside = arr_inside != 0
            else:
                arr_inside = arr_inside * (arr_inside == 0)
            if self.keep_proba_per_branch < 1.:
                arr_inside = arr_inside * selected_branches

        # If self.from_skeleton == False:
        # This keeps only bottom value outside the cutout
        # and keeps the whole skeleton inside the cutout
        else:
            if self.keep_extremity=='top':
                arr_outside = arr_outside * (arr_outside == 35)
            elif self.keep_extremity=='bottom':
                arr_outside = arr_outside * (arr_outside == 30)
            elif self.keep_extremity=='bottom_top':
                arr_outside = arr_outside * (np.logical_or(arr_outside == 30, arr_outside == 35))
            elif self.keep_extremity=='all': # protect whole branch !
                arr_outside = arr_outside != 0
            else:
                arr_outside = arr_outside * (arr_outside == 0)
            if self.keep_proba_per_branch < 1.:
                arr_outside = arr_outside * selected_branches
        
        trimmed_arr = arr_inside + arr_outside

        #log.info(f"{self.from_skeleton},{np.sum(arr!=0)},{np.sum(trimmed_arr!=0)},{np.sum(np.logical_and(arr!=0, arr!=30))},{np.sum(np.logical_and(trimmed_arr!=0,trimmed_arr!=30))}")
        #np.save('/volatile2/jl277509/visu_augmentations/sub_new/skel_trimdepth_extremities_cutout.npy', trimmed_arr)

        trimmed_arr = np.expand_dims(trimmed_arr, axis=0)
        arr_all = np.vstack((trimmed_arr, arr_all[1:]))

        arr_all = arr_all.astype('float32')

        return torch.from_numpy(arr_all)

    
class TransposeTensor(object):
    """
    Permute first and last dimension.
    """

    def __init__(self):
        None
    def __call__(self, tensor):
        arr = tensor.numpy()
        arr_t = np.transpose(arr, (3, 0, 1, 2))
        arr_t = arr_t.astype('float32')

        return(torch.from_numpy(arr_t))
    

class TranslateTensor(object):
    """
    Apply a random slicing of up to n_voxel in every direction and pads
    to perform translation while keeping original dimension.
    """

    def __init__(self, n_voxel):
        self.n_voxel = n_voxel
    
    def __call__(self, tensor):
        arr = tensor.numpy()
        translated_arr = arr.copy()
        if isinstance(self.n_voxel, int):
            absolute_translation_xyz = np.random.randint(self.n_voxel+1, size=3)
        else:
            absolute_translation_xyz = np.array([np.random.randint(n_vx_dim+1) for n_vx_dim in self.n_voxel])
        sign_translation = np.random.randint(2, size=3)
        slc = [slice(None) if (translation==0) else slice(translation, None) if sign else slice(-translation)
               for sign, translation in zip(sign_translation, absolute_translation_xyz)]
        translated_arr = translated_arr[tuple(slc)]
        pad_width = [(0, translation) if sign else (translation, 0) for sign, translation in zip(sign_translation, absolute_translation_xyz)] + [(0,0)]
        translated_arr = np.pad(translated_arr, pad_width, mode='constant', constant_values=0)
        #translated_arr = np.expand_dims(translated_arr[..., 0], axis=0)

        return torch.from_numpy(translated_arr)

