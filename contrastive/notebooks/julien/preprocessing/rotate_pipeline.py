import numpy as np
from soma import aims
import os
import sparse
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from scipy.ndimage import rotate, affine_transform
import matplotlib.pyplot as plt
import random
from scipy.ndimage import map_coordinates
from deep_folding.brainvisa.utils.resample import resample


# remove zeros function
def trim_zeros(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    """
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    return arr[slices], slices


########
dataset = 'UkBioBank'
#dataset = 'ACCpatterns'
#dataset = 'hcp'
modalities = ['skeleton', 'label', 'distbottom']
#modalities = ['label', 'distbottom']
#modalities = ['skeleton', 'label']
#modalities = ['distbottom']

"""
sulcus_list = ['F.Coll.-S.Rh.', 'S.F.median-S.F.pol.tr.-S.F.sup.', 'S.F.inf.-BROCA-S.Pe.C.inf.', \
               'S.Po.C.', 'fronto-parietal_medial_face.', 'F.I.P.', 'S.T.s.-S.GSM.', 'CINGULATE.', \
               'F.C.L.p.-S.GSM.', 'S.C.-S.Po.C.', 'S.F.inter.-S.F.sup.', 'F.C.M.post.-S.p.C.', \
               'S.s.P.-S.Pa.int.', 'S.Or.-S.Olf.', 'F.P.O.-S.Cu.-Sc.Cal.', 'S.F.marginal-S.F.inf.ant.', \
               'S.F.int.-F.C.M.ant.', 'S.T.i.-S.T.s.-S.T.pol.', 'S.F.int.-S.R.', 'Lobule_parietal_sup.', \
               'S.T.i.-S.O.T.lat.', 'S.Pe.C.', 'S.T.s.br.', 'Sc.Cal.-S.Li.', 'S.T.s.', 'F.C.L.p.-subsc.-F.C.L.a.-INSULA.', \
               'S.C.-sylv.', 'S.C.-S.Pe.C.', 'OCCIPITAL', 'S.Or.']

sides = ['L', 'R']
"""

sulcus_list = ['S.C.-sylv.']
sides = ['L']
########

total_size_after=0
total_size_before=0

for sulcus in sulcus_list:
    for side in sides:

        print(f'Treating sulcus : {sulcus}, side: {side}')

        save_dir = f'/neurospin/dico/data/deep_folding/current/datasets/{dataset}/crops/2mm/{sulcus}/mask/'
        mask_dir = f'/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/{sulcus}/mask/{side}mask_cropped.nii.gz'

        mask = aims.read(mask_dir)
        print(f'Mask shape: {mask.np.shape}')
        s = sparse.COO.from_numpy(mask.np[:,:,:,0])
        print(f'Non zero voxels : {np.sum(mask.np!=0)}')

        # take density into account in the point cloud, if the mask is not binary !
        data = [coords for coords, number in zip(s.coords.T, s.data) for i in range(number)]
        data = np.vstack(data).T
        # get principal directions of the data
        #data = s.coords
        cov = np.cov(data)
        eval, evec = np.linalg.eig(cov)
        print(f'Eigenvalues : {eval}')

        # sort the eigenvalues and the eigenvectors accordingly
        idxs = np.argsort(-eval)
        evec[:] = evec[:, idxs]

        # define center of mass
        means = np.mean(data, axis=1)
        print(f'Center of mass : {means}')

        # define the rotation matrix
        rot_mat = evec
        rot_mat = np.hstack((rot_mat, np.zeros((3,1))))
        rot_mat = np.vstack((rot_mat, np.zeros(4)))
        rot_mat[3,3]=1

        # pad the mask so that its dimensions are l*l*l, l=longueur de la grande diagonale du crop
        dim = int(np.ceil(np.sqrt(mask.np.shape[0]**2 + mask.np.shape[1]**2 + mask.np.shape[2]**2)))
        # center of mass is shift by l / 2 on each dimension
        # also, the padding should be an even number on each dimension
        mask_dim = mask.np[:,:,:,0].shape
        dims_to_pad = np.array([dim-m + (dim-m)%2 for m in mask_dim]) // 2
        pad_width = [(d,d) for d in dims_to_pad]
        mask_padded = np.pad(mask.np[:,:,:,0], pad_width)
        means_padded = means+dims_to_pad

        # define transformation
        dx, dy, dz = means_padded[0], means_padded[1], means_padded[2] 

        shift = np.array(
            [
                [1, 0, 0, dx],
                [0, 1, 0, dy],
                [0, 0, 1, dz],
                [0, 0, 0,  1],
            ]
        )
        unshift = np.array(
            [
                [1, 0, 0, -dx],
                [0, 1, 0, -dy],
                [0, 0, 1, -dz],
                [0, 0, 0,   1],
            ]
        )

        trm = shift @ rot_mat @ unshift

        ## apply transformation to mask to get new shape and slices
        output_shape = np.ceil(mask_padded.shape + means_padded).astype(int)
        b = affine_transform(mask_padded, trm, output_shape=output_shape, order=0, mode='constant', cval=0.0)
        c, slices = trim_zeros(b)
        print(f'New shape: {c.shape}, slices: {slices}')
        print(f'Non zero voxel ratio: {np.sum(c!=0) / np.sum(mask.np!=0)}')
        ratio = np.prod(c.shape) / np.prod(mask.shape)
        print(f'Volume ratio: {ratio}')
        if ratio > 1:
            print("RATIO > 1, ROTATION IS DETRIMENTAL, THUS IT IS IGNORED")

        total_size_after+=min(np.prod(c.shape), np.prod(mask.np.shape))
        total_size_before+=np.prod(mask.np.shape)

        
        ## apply rotation to each crop of dataset
        for key  in modalities:
            print(f'ARRAY TYPE : {key}')
            arrs = np.load(f'/neurospin/dico/data/deep_folding/current/datasets/{dataset}/crops/2mm/{sulcus}/mask/{side}{key}.npy')

            if ratio <= 1:
                if key=='distbottom':
                    cval=32501
                else:
                    cval=0
                arr_list = []
                for arr in tqdm(arrs):
                    arr = arr[:,:,:,0]
                    # pad arr the same way as mask
                    arr = np.pad(arr, pad_width, constant_values=cval)
                    rot_arr = affine_transform(arr, trm, output_shape=output_shape, order=0, mode='constant', cval=cval)
                    rot_arr = rot_arr[slices]
                    arr_list.append(rot_arr)

                rot_arrs = np.stack(arr_list)
                rot_arrs = np.expand_dims(rot_arrs, axis=-1)
                print(f'Non zero voxels ratio after rotation : {np.sum(rot_arrs!=cval) / np.sum(arrs!=cval)}')

            else: # do not apply rotation
                rot_arrs = arrs
            
            # set smaller dim on first axis
            #smaller_dim = np.argmin(rot_arrs.shape[1:4])
            #if smaller_dim==1:
            #    rot_arrs = np.transpose(rot_arrs, (0,2,3,1,4))
            #elif smaller_dim==2:
            #    rot_arrs = np.transpose(rot_arrs, (0,3,1,2,4))
            #save rot_arrs
            #print(f'Final dimensions: {rot_arrs.shape}')
            np.save(os.path.join(save_dir, f'{side}{key}_rotated.npy'), rot_arrs)


print(f'Total voxels before: {total_size_before}')
print(f'Total_voxels after: {total_size_after}')
print(f'Ratio: {total_size_after / total_size_before}')