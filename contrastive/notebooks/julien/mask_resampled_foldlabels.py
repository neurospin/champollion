import numpy as np
import os
from soma import aims

#dataset='UkBioBank'
dataset='ACCpatterns'


old_foldlabel_dir = f'/neurospin/dico/data/deep_folding/current/datasets/{dataset}/foldlabels/2mm_old/R/'
new_foldlabel_dir = f'/neurospin/dico/data/deep_folding/current/datasets/{dataset}/foldlabels/2mm/R/'
skels_dir = f'/neurospin/dico/data/deep_folding/current/datasets/{dataset}/skeletons/2mm/R/'
subjects = os.listdir(skels_dir)
subjects = [sub[19:] for sub in subjects if sub[-1]!='f']

for subject in subjects:
    # check if file is already computed
    if not (os.path.isfile(new_foldlabel_dir + 'Rresampled_foldlabel' + subject)
    & os.path.isfile(new_foldlabel_dir + 'Rresampled_foldlabel' + subject + '.minf')):

        skel = aims.read(skels_dir+'Rresampled_skeleton'+subject)
        old_foldlabel = aims.read(old_foldlabel_dir+'Rresampled_foldlabel'+subject)
        skel_np = skel.np
        old_foldlabel_np = old_foldlabel.np

        foldlabel = old_foldlabel_np.copy()
        foldlabel[skel_np==0]=0
        vol = aims.Volume(foldlabel)
        aims.write(vol, new_foldlabel_dir + 'Rresampled_foldlabel' + subject)

