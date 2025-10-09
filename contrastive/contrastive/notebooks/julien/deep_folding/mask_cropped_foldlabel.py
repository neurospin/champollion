import numpy as np
import os
import pandas as pd
from soma import aims
import tqdm
import warnings

def nearest_nonzero_idx(a,x,y,z):
    tmp = a[x,y,z]
    a[x,y,z] = 0
    d,e,f = np.nonzero(a)
    a[x,y,z] = tmp
    min_idx = ((d - x)**2 + (e - y)**2 + (f - z)**2).argmin()
    return(d[min_idx], e[min_idx], f[min_idx])

dataset='UkBioBank'
#dataset='ACCpatterns'
#root = '/neurospin/dico/data/deep_folding/current/datasets/'
root = '/volatile/jl277509/data/' # but I copy only the crops locally..
recompute=True

old_foldlabel_dir = f'{root}{dataset}/crops/1.5mm/S.T.s./mask/Rlabels_no_reskel/'
new_foldlabel_dir = f'{root}{dataset}/crops/1.5mm/S.T.s./mask/Rlabels/'
skels_dir = f'{root}{dataset}/crops/1.5mm/S.T.s./mask/Rcrops/'

directory = f'/volatile/jl277509/data/UkBioBank/crops'
skel_subjects = pd.read_csv(f'{root}{dataset}/crops/1.5mm/S.T.s./mask/Rskeleton_subject.csv')

foldlabel_list = []
for i, subject in enumerate(skel_subjects.Subject):
    # check if file is already computed
    #if recompute or not (os.path.isfile(new_foldlabel_dir + subject + '_cropped_foldlabel.nii.gz')
    #and os.path.isfile(new_foldlabel_dir + subject + '_cropped_foldlabel.nii.gz.minf')):

    skel = aims.read(skels_dir+subject+'_cropped_skeleton.nii.gz')
    old_foldlabel = aims.read(old_foldlabel_dir+subject+'_cropped_foldlabel.nii.gz')
    skel_np = skel.np
    old_foldlabel_np = old_foldlabel.np

    foldlabel = old_foldlabel_np.copy()
    # first mask skeleton using foldlabel because sometimes 1vx is added during skeletonization...
    foldlabel[skel_np==0]=0
    f = foldlabel!=0
    s = skel_np!=0
    diff_fs = np.sum(f!=s)
    assert (diff_fs<=5), f"subject {subject} has incompatible foldlabel and skeleton. {np.sum(s)} vx in skeleton, {np.sum(f)} vx in foldlabel"
    if diff_fs!=0:
        warnings.warn(f"subject {subject} has incompatible foldlabel and skeleton. {np.sum(s)} vx in skeleton, {np.sum(f)} vx in foldlabel")
        idxs = np.where(f!=s)
        print(idxs)
        for i in range(diff_fs):
            x,y,z = idxs[0][i], idxs[1][i], idxs[2][i]
            d,e,f = nearest_nonzero_idx(foldlabel[:,:,:,0],x,y,z)
            foldlabel[x,y,z,0]=foldlabel[d,e,f,0]
            print(f'foldlabel has a 0 at index {x,y,z}, nearest nonzero at index {d,e,f}, value {foldlabel[d,e,f,0]}')
    f = foldlabel!=0
    assert np.sum(f!=s)==0, f'subject {subject} has incompatible foldlabel and skeleton AFTER CORRECTION. {np.sum(s)} vx in skeleton, {np.sum(f)} vx in foldlabel'
    foldlabel_list.append(foldlabel)
    vol = aims.Volume(foldlabel)
    vol.header()['voxel_size'] = [1.5, 1.5, 1.5]
    aims.write(vol, new_foldlabel_dir + subject + '_cropped_foldlabel.nii.gz')

# generate Rlabel.npy
arr = np.stack(foldlabel_list)
print(f'Rlabel shape: {arr.shape}')
np.save(f'{root}{dataset}/crops/1.5mm/S.T.s./mask/Rlabel.npy', arr)