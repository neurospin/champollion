import numpy as np
import os
from soma import aims
import torch
import torchvision.transforms as transforms
#from contrastive.augmentations import TrimDepthTensor, SimplifyTensor, PaddingTensor, BinarizeTensor
import contrastive
from contrastive.augmentations import *

# need to pip install -e in BrainVisa to import the function from contrastive ?? Can't use venv either otherwise aims can't be imported
class TrimDepthTensor(object):
    """
    Trim depth based on distbottom.
    Set max_distance to 0 to remove bottom only.
    Set max_distance to -1 to remove nothing.
    Then the scale is 100 = 2mm.
    """

    def __init__(self, sample_distbottom, sample_foldlabel,
                 max_distance, input_size, keep_extremity, uniform, binary):
        self.max_distance = max_distance
        self.input_size = input_size
        self.sample_distbottom = sample_distbottom
        self.sample_foldlabel = sample_foldlabel
        self.uniform=uniform
        self.binary=binary
        if keep_extremity=='random':
            np.random.seed()
            r = np.random.randint(2)
            if r == 0:
                self.keep_extremity='top'
            else:
                self.keep_extremity=None
        else:
            self.keep_extremity = keep_extremity
    
    def __call__(self, tensor_skel):
        log.debug(f"Shape of tensor_skel = {tensor_skel.shape}")
        arr_skel = tensor_skel.numpy()
        arr_distbottom = self.sample_distbottom.numpy()
        arr_foldlabel = self.sample_foldlabel.numpy()

        # log.debug(f"arr_skel.shape = {arr_skel.shape}")
        # log.debug(f"arr_foldlabel.shape = {arr_foldlabel.shape}")
        assert (arr_skel.shape == arr_distbottom.shape)
        assert (self.max_distance >= -1)

        if self.uniform:
            arr_trimmed = arr_skel.copy()
            # get random threshold
            threshold = np.random.randint(-1, self.max_distance+1)
            # mask skel with thresholded distbottom
            if self.keep_extremity=='top':
                arr_trimmed[np.logical_and(arr_distbottom<=threshold, arr_skel!=35)]=0
            else:
                arr_trimmed[arr_distbottom<=threshold]=0
        else:
            # select a threshold for each branch
            arr_trimmed_branches = np.zeros(arr_skel.shape)
            indexes =  np.unique(
                                np.mod(arr_foldlabel,
                                        np.full(arr_foldlabel.shape, fill_value=1000))
                                )
            assert (len(indexes)>1), 'No branch in foldlabel'
            for index in indexes[1:]:
                arr_trimmed = arr_skel.copy()
                mask_branch = np.mod(arr_foldlabel,
                                    np.full(arr_foldlabel.shape, fill_value=1000))==index
                if self.binary:
                    r = np.random.randint(2)
                    if r == 0:
                        threshold = -1
                    else:
                        threshold = self.max_distance
                else:
                    threshold = np.random.randint(-1, self.max_distance+1)
                if self.keep_extremity=='top':
                    arr_trimmed[np.logical_and(arr_distbottom<=threshold, arr_skel!=35)]=0
                else:
                    arr_trimmed[arr_distbottom<=threshold]=0
                arr_trimmed_branches += (arr_trimmed * mask_branch)
            arr_trimmed = arr_trimmed_branches.copy()

        
        arr_trimmed = arr_trimmed.astype('float32')

        return torch.from_numpy(arr_trimmed)

root_dir = '/volatile/jl277509/data/UkBioBank/crops/1.5mm_reclassif/CINGULATE/mask/'
skel = aims.read(root_dir + 'Rcrops/sub-1000021_cropped_skeleton.nii.gz')
fold = aims.read(root_dir + 'Rlabels/sub-1000021_cropped_foldlabel.nii.gz')
dist = aims.read(root_dir + 'Rdistbottom/sub-1000021_cropped_distbottom.nii.gz')

skel_np = skel.np
fold_np = fold.np
dist_np = dist.np

skel_tensor = torch.tensor(skel_np)
fold_tensor = torch.tensor(fold_np)
dist_tensor = torch.tensor(dist_np)

transforms_list_1 = [SimplifyTensor(),
                     TrimDepthTensor(sample_distbottom=dist_tensor,
                            sample_foldlabel=fold_tensor,
                            max_distance=75,
                            input_size=(1, 29, 66, 60),
                            keep_extremity='bottom',
                            uniform=False,
                            binary=True),
                     BinarizeTensor()]
trans1 = transforms.Compose(transforms_list_1)

transforms_list_2 = [SimplifyTensor(),
                     TrimDepthTensor(sample_distbottom=dist_tensor,
                            sample_foldlabel=fold_tensor,
                            max_distance=75,
                            input_size=(1, 29, 66, 60),
                            keep_extremity='bottom',
                            uniform=False,
                            binary=True),
                     BinarizeTensor()]
trans2 = transforms.Compose(transforms_list_2)

view1 = trans1(skel_tensor)
view2 = trans2(skel_tensor)

view1=view1.detach().numpy().astype(np.int16)
view2=view2.detach().numpy().astype(np.int16)

vol1 = aims.Volume(view1)
vol1.header()['voxel_size']=[1.5,1.5,1.5]
vol2 = aims.Volume(view2)
vol2.header()['voxel_size']=[1.5,1.5,1.5]

aims.write(vol1, '/volatile/jl277509/data/distbottom_illustration/view1_trimdepth.nii.gz')
aims.write(vol2, '/volatile/jl277509/data/distbottom_illustration/view2_trimdepth.nii.gz')

