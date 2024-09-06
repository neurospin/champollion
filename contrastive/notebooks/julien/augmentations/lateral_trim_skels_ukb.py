import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from soma import aims
from soma.aimsalgo.sulci import trim_extremity
from multiprocessing import Pool


# sshfs dir
ukb_graph_dir = '/volatile/jl277509/data/graphs_ukb/'
path_to_graph = 'ses-2/anat/t1mri/default_acquisition/default_analysis/folds/3.1'
path_to_skel = 'ses-2/anat/t1mri/default_acquisition/default_analysis/segmentation/'

ukb_raw_dir = '/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/skeletons/raw/'
trm_dir = '/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/transforms/'
read_dir = '/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/trimmed_skeletons/'
save_dir = '/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/extremities/raw'

side = 'L'

subjects = pd.read_csv('/volatile/jl277509/data/UkBioBank/L_subjects.csv', header=None)[0].tolist()
subjects = [_[1:] for _ in subjects] # remove side
print(f'Number of subjects : {len(subjects)}')

"""
#check for already processed subjects
processed_subs = os.listdir(os.path.join(save_dir, side))
processed_subs = [_ for _ in processed_subs if _[10:23]=='trimmed_edges']
processed_subs = [_ for _ in processed_subs if _[-1]!='f']
processed_subs = [_[-18:-7] for _ in processed_subs]
subjects = list(set(subjects).symmetric_difference(processed_subs))
print(f'Number of subjects to be processed : {len(subjects)}')




def process_function(sub):
    # get scaling with trm
    trm = pd.read_csv(os.path.join(trm_dir, f'{side}/{side}transform_to_ICBM2009c_{sub}.trm'), sep=' ', header=None)
    trm = np.array(trm)
    scale = (trm[1,0]*trm[2,1]*trm[3,2])**(1/3)
    #print(f'scaling factor for tminss : {scale}')
    # get native skeleton
    skel = aims.read(os.path.join(ukb_graph_dir, sub, path_to_skel ,f'{side}skeleton_{sub}.nii.gz'))
    #skel.np[skel.np==11]=0 # remove topological value 11
    # get graph
    graph = aims.read(os.path.join(ukb_graph_dir, sub, path_to_graph, f'{side}{sub}.arg'))
    tminss = 3. / scale
    ss, trimmed = trim_extremity.trim_extremities(skel, graph, tminss, junc_dilation=1)
    ss.np[ss.np < 32500]=1
    ss.np[ss.np >= 32500]=0

    print(f'Non zero voxels ratio before/after trimming : {np.sum(ss.np!=0)} / {np.sum(trimmed.np!=0)}')
    aims.write(trimmed, os.path.join(save_dir, f'{side}/{side}skeleton_trimmed_edges_{sub}.nii.gz'))
    aims.write(ss, os.path.join(save_dir, f'{side}/{side}skeleton_ss_edges_{sub}.nii.gz'))

"""

def process_function(sub):

    trimmed = aims.read(os.path.join(read_dir, f'{side}/{side}skeleton_trimmed_edges_{sub}.nii.gz'))
    ss = aims.read(os.path.join(read_dir, f'{side}/{side}skeleton_ss_edges_{sub}.nii.gz'))

    print(np.sum(ss.np),np.sum(trimmed.np))
    ss.np[:] -= trimmed.np
    print(np.sum(ss.np!=0))
    aims.write(ss, os.path.join(save_dir, f'{side}/{side}extremities_{sub}.nii.gz'))

def process_in_parallel(subjects, process_function, num_worker):
    # Create a pool of worker processes
    with Pool(num_worker) as pool:
        # Use pool.map to apply the process_function to each image_path
        pool.map(process_function, subjects)


if __name__ == '__main__':
    process_in_parallel(subjects, process_function, num_worker=30)


