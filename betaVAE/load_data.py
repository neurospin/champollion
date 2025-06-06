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
import random
import pandas as pd
import numpy as np
from preprocess import *

def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]

def create_subset(config):
    """
    Creates dataset HCP_1 from HCP data
    Args:
        config: instance of class Config
    Returns:
        subset: Dataset corresponding to HCP_1
    """

    #We load the list of subjects with one column called 'subjects'
    train_list = pd.read_csv(config.subject_dir)
    print('---------------------------------------- Reading .csv ',config.subject_dir)
    print(train_list)

    #We get the list, ensure they are strings and remove the sub- part at the beginning of the string
    train_list.columns=['subjects']
    train_list['subjects'] = train_list['subjects'].astype('str')
    tmp_sub = train_list['subjects'].tolist()
    if tmp_sub[0][:4]=='sub-':
        tmp_sub = [subject[4:] for subject in tmp_sub]
        train_list['subjects']=tmp_sub

    print('---------------------------------------- Train list edited without sub-')
    print(train_list)

    print('Filename and file extension of subjects data')
    filename, file_extension = os.path.splitext(config.data_dir)
    print('---------------------------------------- Filename and file-extension',filename,file_extension)

    if file_extension=='.pkl':
        print('Reading pickle file')
        #Cada columna es un sujeto
        tmp = pd.read_pickle(config.data_dir)
        print('Print some pickle info',type(tmp),list(tmp.index),list(tmp.columns.values),tmp.info(),type(tmp.iloc[0]['100206']),tmp.iloc[0]['100206'].shape)

    elif file_extension=='.npy':
        print('Reading numpy file')
        #We load the numpy file and append the crop ( [numpy array] ) 
        tmp = np.load(config.data_dir)
        list_crops = []
        for crop in range(0,tmp.shape[0]):
            list_crops.append([tmp[crop,:,:,:,:]])

        #We create a dictionary containing the subject (key) and their crop (value)
        dict_sub_crop = dict(zip(train_list['subjects'].tolist(), list_crops))
        print('Size of dictionary containing subject id (key) and crop (value)', len(dict_sub_crop))

        #If we want to train with the whole dataset
        if config.remove_subjects == False:
            tmp = pd.DataFrame.from_dict(dict_sub_crop)
            print('final tmp',tmp.shape,tmp.iloc[0][0].shape,tmp.info())

        #If we want to remove some subjects
        else:
            #We load the list of subjects to be removed from train
            subjects_to_exclude = pd.read_csv(config.subjects_to_remove)
            print('Subjects to exclude:',subjects_to_exclude)

            #We get the ones wich we are sure are interrupted
            df_filtered = subjects_to_exclude[subjects_to_exclude['Note'] == 'OK']
            print('Subjects to exclude after filtering by OK:',df_filtered)

            #We remove the sub- part from the string
            df_filtered['ID'] = df_filtered['ID'].astype('str')
            tmp_excluded = df_filtered['ID'].tolist()
            print('Before removing sub-',tmp_excluded[0][:4])
            if tmp_excluded[0][:4]=='sub-':
                tmp_excluded = [subject[4:] for subject in tmp_excluded]
                df_filtered['ID']=tmp_excluded
            print('After removing sub-',df_filtered)

            #We get a list of the subjects to be removed, which are OK (realiable interrupted) and without the sub- part in the string
            subs_to_remove = df_filtered['ID'].tolist()
            print(' ° dict_sub_crop size before removing interrupted subjects', len(dict_sub_crop))
            print(' ° train_list size before removing interrupted subjects',len(train_list))

            #We removed the subjects from the dictionary and from the whole list of subjects
            for sub in subs_to_remove:
                del dict_sub_crop[sub]
            train_list = filter_rows_by_values(train_list , "subjects", subs_to_remove)

            print('°° dict_sub_crop size after removing interrupted subjects', len(dict_sub_crop))
            print('°° train_list size before after removing interrupted subjects',len(train_list))

            #We create a test dataset from the set of subject without the interrupted subjects, same size of interrupted subjects
            n = 0
            test_subjects = []
            for sub in random.sample(list(dict_sub_crop.keys()), len(subs_to_remove) ):
                test_subjects.append(sub)
                n+=1
                if n == 207:
                    break
            #We remove the test subjects from the dictionary and whole list of subjects
            for sub in test_subjects:
                del dict_sub_crop[sub]
            
            train_list = filter_rows_by_values(train_list , "subjects", test_subjects)

            print('°° dict_sub_crop size after removing test subjects', len(dict_sub_crop))
            print('°° train_list size before after removing test subjects',len(train_list))

            #Save everything to csv
            train_dataset = pd.DataFrame.from_dict({'ID':train_list['subjects'].tolist()})
            test_dataset = pd.DataFrame.from_dict({'ID':test_subjects})
            df_filtered.to_csv(config.save_dir+'/Interrupted_CS_subjects.csv')
            test_dataset.to_csv(config.save_dir+'/Test_subjects.csv')
            train_dataset.to_csv(config.save_dir+'/Train_subjects.csv')

            #Finally, we create a dataframe from the dictionary
            tmp = pd.DataFrame.from_dict(dict_sub_crop)
    
    subjects = tmp.columns.tolist()
    #print('subjects',subjects)

    #I Don't get why this is here but we don't get in the for cycle anyways
    if subjects[0][:4]=='sub-': # remove sub
        print('Yep got into if')
        subjects = [subject[4:] for subject in subjects]
        tmp.columns = subjects

    #We are almost there
    print('Last part of preprocessing the dataset')
    tmp = tmp.T
    tmp.index.astype('str')
    
    ''' Just as a reminder
    a = {'A':[123],'B':[245],'C':[678]}
    tmp = pd.DataFrame.from_dict(a)
    #print(tmp,'\n',tmp.T)
    tmp = tmp.T
    print([tmp.index[k] for k in range(len(tmp))])
    Output:
         A    B    C
        0  123  245  678 
            0
        A  123
        B  245
        C  678
        ['A', 'B', 'C']


        ** Process exited - Return Code: 0 **
        Press Enter to exit terminal
    '''
    
    #Here we get a list with the ID of the subjects
    tmp['subjects'] = [tmp.index[k] for k in range(len(tmp))]
    res = tmp['subjects'].tolist()== train_list['subjects'].tolist()
    if res == False:
        print('--- Problem --- Verify order of subjects for dataset')
        sys.exit()
    else:
        print('Same ordering',res)

    print('Final number of subject for train:',len(tmp['subjects'].tolist()))
    #We merged it
    #tmp = tmp.merge(train_list, left_on = 'subjects', right_on='subjects', how='right')
    #filenames = list(train_list['subjects'])
    #tmp = tmp.merge(train_list, left_on = 'subjects', right_on='subjects', how='right')
    #filenames = list(train_list['subjects'])
    tmp = tmp.merge(tmp['subjects'], left_on = 'subjects', right_on='subjects', how='right')
    filenames = list(tmp['subjects'])
    

    subset = SkeletonDataset(config=config, dataframe=tmp, filenames=filenames)
    print('------- Succesfully loaded dataset')
    return subset


def create_subset_eval(config):
    """
    Creates subset
    Args:
        config: instance of class Config
    Returns:
        subset: Dataset corresponding to HCP_1
    """

    #We load the list of subjects with one column called 'subjects'
    train_list = pd.read_csv(config.subject_dir)
    print('---------------------------------------- Reading .csv ',config.subject_dir)
    print(train_list)

    #We get the list, ensure they are strings and remove the sub- part at the beginning of the string
    train_list.columns=['subjects']
    train_list['subjects'] = train_list['subjects'].astype('str')
    tmp_sub = train_list['subjects'].tolist()
    if tmp_sub[0][:4]=='sub-':
        tmp_sub = [subject[4:] for subject in tmp_sub]
        train_list['subjects']=tmp_sub

    print('---------------------------------------- Train list edited without sub-')
    print(train_list)

    print('Filename and file extension of subjects data')
    filename, file_extension = os.path.splitext(config.data_dir)
    print('---------------------------------------- Filename and file-extension',filename,file_extension)

    if file_extension=='.pkl':
        print('Reading pickle file')
        #Cada columna es un sujeto
        tmp = pd.read_pickle(config.data_dir)
        print('Print some pickle info',type(tmp),list(tmp.index),list(tmp.columns.values),tmp.info(),type(tmp.iloc[0]['100206']),tmp.iloc[0]['100206'].shape)

    elif file_extension=='.npy':
        print('Reading numpy file')
        #We load the numpy file and append the crop ( [numpy array] ) 
        tmp = np.load(config.data_dir)
        list_crops = []
        for crop in range(0,tmp.shape[0]):
            list_crops.append([tmp[crop,:,:,:,:]])

        #We create a dictionary containing the subject (key) and their crop (value)
        dict_sub_crop = dict(zip(train_list['subjects'].tolist(), list_crops))
        print('Size of dictionary containing subject id (key) and crop (value)', len(dict_sub_crop))


    tmp = pd.DataFrame.from_dict(dict_sub_crop)
    subjects = tmp.columns.tolist()
    #print('subjects',subjects)

    #I Don't get why this is here but we don't get in the for cycle anyways / maybe we will remove it later on
    if subjects[0][:4]=='sub-': # remove sub
        print('Yep got into if')
        subjects = [subject[4:] for subject in subjects]
        tmp.columns = subjects

    #We are almost there
    print('Last part of preprocessing the dataset')
    tmp = tmp.T
    tmp.index.astype('str')

    #Here we get a list with the ID of the subjects
    tmp['subjects'] = [tmp.index[k] for k in range(len(tmp))]
    res = tmp['subjects'].tolist()== train_list['subjects'].tolist()

    if res == False:
        print('--- Problem --- Verify order of subjects for dataset')
        sys.exit()
    else:
        print('Same ordering',res)


    #We merged it
    #tmp = tmp.merge(train_list, left_on = 'subjects', right_on='subjects', how='right')
    #filenames = list(train_list['subjects'])
    #tmp = tmp.merge(train_list, left_on = 'subjects', right_on='subjects', how='right')
    #filenames = list(train_list['subjects'])
    tmp = tmp.merge(tmp['subjects'], left_on = 'subjects', right_on='subjects', how='right')
    filenames = list(tmp['subjects'])

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
