import numpy as np
import pandas as pd
import os
from soma import aims
from soma.aimsalgo.sulci import trim_extremity
from scipy.ndimage import label
from skimage.morphology import ball

subject = '100610'
graph_dir = f'/neurospin/dico/data/bv_databases/human/not_labeled/hcp/hcp/{subject}/t1mri/BL/default_analysis/folds/3.1/L{subject}.arg'
skel_dir = f'/neurospin/dico/data/bv_databases/human/not_labeled/hcp/hcp/{subject}/t1mri/BL/default_analysis/segmentation/Lskeleton_{subject}.nii.gz'

save_dir = '/volatile/jl277509/data/tmp'

graph = aims.read(graph_dir)
skel = aims.read(skel_dir)

tminss = 3


ss, trimmed = trim_extremity.trim_extremities(skel, graph, tminss=tminss, junc_dilation=1)
ss.np[ss.np < 32500]=1
ss.np[ss.np >= 32500]=0

sum_ss = np.sum(ss.np!=0)
print(sum_ss)
sum_trimmed = np.sum(trimmed.np!=0)
print(sum_trimmed)
voxels_to_trim = ss.np-trimmed.np

#structure = np.ones((3, 3, 3), dtype=int)  # 26-connectivity
structure = ball(radius=1)
print(structure.shape)
labeled_image, num_features = label(voxels_to_trim[:,:,:,0], structure=structure)

print(f"Number of objects (connected components): {num_features}")

# To extract objects, you can loop over the unique labels
for i in range(1, num_features + 1):
    object_i = labeled_image == i
    print(f'Object {i} size: {np.sum(object_i)}')
    # mask the objects with value 80 in skel (ie junction)
    subset_values = skel.np[object_i]
    if 80 in subset_values:
        print(f'Object {i} is protected (junction)')
        labeled_image[object_i]=0
        num_features-=1

trimmed_protect_junction = np.expand_dims((labeled_image!=0).astype(np.int16), -1)
print(f"shape before: {skel.np.shape}, after: {trimmed_protect_junction.shape}")
print(f"Number of objects after junction protection: {num_features}")
sum_to_trim = np.sum(trimmed_protect_junction)
print(f'Number of voxels to trim : {sum_to_trim}')
print(f"Number of voxels protected : {sum_ss-sum_trimmed-sum_to_trim}")


aims.write(trimmed, os.path.join(save_dir, f'trimmed_{subject}.nii.gz'))
aims.write(ss, os.path.join(save_dir, f'ss_{subject}.nii.gz'))

# to make the header is correct
ss.np[trimmed_protect_junction==0] = 0
ss.np[trimmed_protect_junction!=0] = 1000
aims.write(ss, os.path.join(save_dir, f'trimmed_{subject}_skel_junc_protected.nii.gz'))