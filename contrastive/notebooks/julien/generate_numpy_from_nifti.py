## take the nifti dir and stack all in numpy
import numpy as np
import pandas as pd
import os
from soma import aims
from tqdm import tqdm

"""
dataset='UkBioBank40'
side='R'
region='F.I.P.'
"""

dataset='hcp'
side='R'
region='F.I.P.'

root_dir = f'/neurospin/dico/data/deep_folding/current/datasets/{dataset}/crops/2mm/{region}/mask'
ccdistmap_dir = os.path.join(root_dir, f'{side}ccdistmaps')

subjects = pd.read_csv(os.path.join(root_dir, f'{side}skeleton_subject.csv'))['Subject'].tolist()

list_arr = []

for subject in tqdm(subjects):
    ccdistmap = aims.read(os.path.join(ccdistmap_dir, f'{subject}_cropped_skeleton.nii.gz'))
    ccdistmap = ccdistmap.np
    list_arr.append(ccdistmap)

arr_skel = np.stack(list_arr)

print(f'array shape : {arr_skel.shape}')

np.save(os.path.join(root_dir, 'Rccdistmaps.npy'), arr_skel)
                          

