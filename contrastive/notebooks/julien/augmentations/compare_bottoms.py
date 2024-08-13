import numpy as np
import os
from soma import aims

save_dir = '/volatile/jl277509/data/test_augmentations/bottoms'

skels = np.load('/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/ORBITAL/mask/Lskeleton.npy')
folds = np.load('/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/ORBITAL/mask/Llabel.npy')

indexes = np.random.randint(0, 21000, 20)
for idx in indexes:
    skel = skels[idx]
    fold = folds[idx]

    skel_binary = skel!=0
    vol = aims.Volume(skel_binary.astype(np.int16))
    aims.write(vol, os.path.join(save_dir, f'skel_binary_{idx}.nii.gz'))

    skel_bottoms = skel==30
    vol = aims.Volume(skel_bottoms.astype(np.int16))
    aims.write(vol, os.path.join(save_dir, f'skel_bottoms_{idx}.nii.gz'))

    fold_bottoms = np.logical_and(fold>=7000, fold<8000)
    vol = aims.Volume(fold_bottoms.astype(np.int16))
    aims.write(vol, os.path.join(save_dir, f'fold_bottoms_{idx}.nii.gz'))
