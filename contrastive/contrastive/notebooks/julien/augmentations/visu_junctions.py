import numpy as np
import pandas as pd
import os
from soma import aims
from skimage.morphology import ball, binary_dilation

subject = '100610'
dataset = 'hcp'
#subject = 'sub-1000021'
#dataset = 'UkBioBank'
foldlabel_dir = f'/neurospin/dico/data/deep_folding/current/datasets/{dataset}/foldlabels/raw/L/Lfoldlabel_{subject}.nii.gz'
#skel_dir = f'/neurospin/dico/data/deep_folding/current/datasets/{dataset}/skeletons/2mm/L/Lresampled_skeleton_{subject}.nii.gz'
#skel_dir = f'/neurospin/dico/data/deep_folding/current/datasets/{dataset}/skeletons/raw/L/Lskeleton_generated_{subject}.nii.gz'
skel_dir = f'/neurospin/dico/data/bv_databases/human/not_labeled/hcp/hcp/100610/t1mri/BL/default_analysis/segmentation/Lskeleton_{subject}.nii.gz'

foldlabel = aims.read(foldlabel_dir)
skel= aims.read(skel_dir)
print(np.unique(skel, return_counts=True))
save_dir = '/volatile/jl277509/data/tmp'
print(np.unique(((foldlabel.np)//1000).astype(int), return_counts=True))


## foldlabel topology

foldlabel.np[np.logical_and(foldlabel.np >= 5000, foldlabel.np<6000)]=0
foldlabel.np[foldlabel.np!=0]=1
print(np.sum(foldlabel))

aims.write(foldlabel, os.path.join(save_dir, f'binary_foldlabel_{subject}.nii.gz'))


foldlabel = aims.read(foldlabel_dir)

foldlabel.np[foldlabel.np >= 6000]=0
foldlabel.np[foldlabel.np < 5000]=0
foldlabel.np[foldlabel.np != 0]=1
print(np.sum(foldlabel))

aims.write(foldlabel, os.path.join(save_dir, f'jnctions_foldlabel_{subject}.nii.gz'))


## skeleton topology

skel.np[skel.np==80]=0
skel.np[skel.np==11]=0
skel.np[skel.np!=0]=1
print(np.sum(skel))

aims.write(skel, os.path.join(save_dir, f'binary_skeleton_{subject}.nii.gz'))


skel = aims.read(skel_dir)

skel.np[skel.np != 80]=0
skel.np[skel.np == 80]=1
print(np.sum(skel))

aims.write(skel, os.path.join(save_dir, f'jnctions_skeleton_{subject}.nii.gz'))

skel = aims.read(skel_dir)
skel.np[skel.np != 80]=0
skel.np[skel.np == 80]=1
structure = np.ones((3, 3, 3), dtype=int)  # 26-connectivity
#structure = ball(1)
dilated_80 = binary_dilation(skel.np[:,:,:,0], structure)

skel.np[dilated_80!=0]=1
aims.write(skel, os.path.join(save_dir, f'jnctions_dilated_skeleton_{subject}.nii.gz'))

