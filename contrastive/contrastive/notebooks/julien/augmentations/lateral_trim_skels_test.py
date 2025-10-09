import numpy as np
import pandas as pd
import os
from soma import aims
from soma.aimsalgo.sulci import trim_extremity

tminsss = [2,3]
junc_dilation = 1

subjects = ['107018', '141422']

for subject in subjects:

    graph_dir = f'/neurospin/dico/data/bv_databases/human/not_labeled/hcp/hcp/{subject}/t1mri/BL/default_analysis/folds/3.1/L{subject}.arg'
    skel_dir = f'/neurospin/dico/data/bv_databases/human/not_labeled/hcp/hcp/{subject}/t1mri/BL/default_analysis/segmentation/Lskeleton_{subject}.nii.gz'

    save_dir = '/volatile/jl277509/data/tmp'

    graph = aims.read(graph_dir)
    skel = aims.read(skel_dir)

    for tminss in tminsss:

        ss, trimmed = trim_extremity.trim_extremities(skel, graph, tminss=tminss, junc_dilation=junc_dilation)
        ss.np[ss.np < 32500]=1
        ss.np[ss.np >= 32500]=0

        print(np.sum(trimmed.np!=0))
        print(np.sum(ss.np!=0))

        aims.write(trimmed, os.path.join(save_dir, f'trimmed_{subject}_tminss{tminss}.nii.gz'))
        aims.write(ss, os.path.join(save_dir, f'ss_{subject}_tminss{tminss}.nii.gz'))