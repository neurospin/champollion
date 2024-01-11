# -*- coding: utf-8 -*-
# /usr/bin/env python3
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
#
# Initial code:
# https://github.com/neurospin-projects/2021_jchavas_lguillon_deepcingulate/
#                   betaVAE/load_data.py

"""
Tools in order to create pytorch dataloaders
"""
import os
import sys
import re

import pandas as pd
import numpy as np
from preprocess import *


def create_subset(config):
    """
    Creates dataset HCP_1 from HCP data
    Args:
        config: instance of class Config
    Returns:
        subset: Dataset corresponding to HCP_1
    """

    train_list = pd.read_csv(config.subject_dir)
    train_list.columns=['subjects']
    train_list['subjects'] = train_list['subjects'].astype('str')
    tmp_sub = train_list['subjects'].tolist()
    if tmp_sub[0][:4]=='sub-':
        tmp_sub = [subject[4:] for subject in tmp_sub]
        train_list['subjects']=tmp_sub


    tmp = pd.read_pickle(config.data_dir)

    subjects = tmp.columns.tolist()
    if subjects[0][:4]=='sub-': # remove sub
        subjects = [subject[4:] for subject in subjects]
        tmp.columns = subjects
    tmp = tmp.T
    tmp.index.astype('str')
    tmp['subjects'] = [tmp.index[k] for k in range(len(tmp))]
    tmp = tmp.merge(train_list, left_on = 'subjects', right_on='subjects', how='right')
    filenames = list(train_list['subjects'])

    subset = SkeletonDataset(config=config, dataframe=tmp, filenames=filenames)

    return subset


def create_test_subset(config):
    """
    Creates test dataset from ACC database
    Args:
        config: instance of class Config
    Returns:
        subset: Dataset corresponding to ACC dataset
    """

    tmp = pd.read_pickle(os.path.join(config.acc_subjects_dir, "Rskeleton.pkl")).T
    tmp.index.astype('str')

    re_expr = '/neurospin/dico/data/deep_folding/current/datasets/ACCpatterns/'\
                'crops/2mm/CINGULATE/mask/Rcrops/(.*)'

    tmp['subjects'] = [re.search(re_expr, tmp.index[k]).group(1) for k in range(
                        len(tmp))]

    filenames = list(tmp['subjects'])

    subset_test = SkeletonDataset(config=config, dataframe=tmp, filenames=filenames)

    return subset_test
