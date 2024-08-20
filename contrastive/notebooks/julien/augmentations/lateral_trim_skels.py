import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from soma import aims
from soma.aimsalgo.sulci import trim_extremity


## TODO: 300 ou 3 ?? valeur en mm ? 

# sshfs dir
ukb_graph_dir = '/volatile/jl277509/data/graphs_ukb/'
path_to_graph = 'ses-2/anat/t1mri/default_acquisition/default_analysis/folds/3.1'
path_to_skel = 'ses-2/anat/t1mri/default_acquisition/default_analysis/segmentation/'

ukb_raw_dir = '/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/skeletons/raw/'
trm_dir = '/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/transforms/'
save_dir = '/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/trimmed_skeletons/'


sides = ['L', 'R']

for side in sides:

    print(f'Treating side {side}')
    subjects = pd.read_csv('/volatile/jl277509/data/UkBioBank/L_subjects.csv', header=None)[0].tolist()
    subjects = [_[1:] for _ in subjects] # remove side

    scale_list = []
    sub_list = []
    for sub in tqdm(subjects):
        # get scaling with trm
        trm = pd.read_csv(os.path.join(trm_dir, f'{side}/{side}transform_to_ICBM2009c_{sub}.trm'), sep=' ', header=None)
        trm = np.array(trm)
        scale = (trm[1,0]*trm[2,1]*trm[3,2])**(1/3)
        print(f'scaling factor for tminss : {scale}')
        # get raw skeleton
        #skel = aims.read(os.path.join(ukb_raw_dir ,f'{side}/{side}skeleton_generated_{sub}.nii.gz'))
        ## TODO: WHAT SKELETON SHOULD BE USED ???
        skel = aims.read(os.path.join(ukb_graph_dir, sub, path_to_skel ,f'{side}skeleton_{sub}.nii.gz'))
        skel.np[skel.np==11]=0 # remove topological value 11
        print(f'Non zero voxels before trimming : {np.sum(skel.np!=0)}')
        # get graph
        graph = aims.read(os.path.join(ukb_graph_dir, sub, path_to_graph, f'{side}{sub}.arg'))
        tminss = 3 / scale
        ss, trimmed = trim_extremity.trim_extremities(skel, graph, tminss)
        print(f'Non zero voxels after trimming : {np.sum(trimmed.np!=0)}')
        print(f'Non zero voxels after trimming (ss) : {np.sum(ss.np!=0)}')
        aims.write(trimmed, os.path.join(save_dir, f'{side}/{side}skeleton_trimmed_edges_{sub}.nii.gz'))
        break