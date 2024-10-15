import numpy as np
import sparse
import os
import pandas as pd
from tqdm import tqdm
import scipy
from multiprocessing import Pool

side = 'R'
sulcus = 'F.I.P.'


target_size = (48, 48, 48)
input_size = (39, 45, 44)
#input_size = (40, 44, 42)
zoom_factor = (target_size[0] / input_size[0], target_size[1] / input_size[1], target_size[2] / input_size[2])
#zoom_factor = (112/45, 112/45, 112/45) # doesn't alter the shapes

root_read_dir =  '/home_local/jl277509/data/sparse_load/UkBioBank/crops/2mm'
#root_read_dir = '/volatile/jl277509/data/UkBioBank/crops/2mm/'
read_dir = f'{root_read_dir}/{sulcus}/mask/{side}skeleton_sparse'
root_save_dir = '/home_local/jl277509/data/sparse_load/UkBioBank/crops_upscaled/2mm'
#root_save_dir = '/volatile/jl277509/data/UkBioBank/crops_upscaled/2mm/'
save_dir = f'{root_save_dir}/{sulcus}/mask/{side}skeleton_sparse'


data_dir = f'/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/{sulcus}/mask'
subjects = pd.read_csv(os.path.join(data_dir, f'{side}skeleton_subject.csv'))
subjects = subjects['Subject'].tolist()

if not os.path.isdir(os.path.join(save_dir, 'coords')):
    os.makedirs(os.path.join(save_dir, 'coords'))
if not os.path.isdir(os.path.join(save_dir, 'skeleton')):
    os.makedirs(os.path.join(save_dir, 'skeleton'))
if not os.path.isdir(os.path.join(save_dir, 'foldlabel')):
    os.makedirs(os.path.join(save_dir, 'foldlabel'))
if not os.path.isdir(os.path.join(save_dir, 'distbottom')):
    os.makedirs(os.path.join(save_dir, 'distbottom'))
if not os.path.isdir(os.path.join(save_dir, 'extremities')):
    os.makedirs(os.path.join(save_dir, 'extremities'))



def process_function(subject):

    # sparse load
    coords = np.load(os.path.join(read_dir, f'coords/{side}{subject}_coords.npy'))
    skel = np.load(os.path.join(read_dir, f'skeleton/{side}{subject}_skeleton_values.npy'))
    fold = np.load(os.path.join(read_dir, f'foldlabel/{side}{subject}_foldlabel_values.npy'))
    distb = np.load(os.path.join(read_dir, f'distbottom/{side}{subject}_distbottom_values.npy'))
    extr = np.load(os.path.join(read_dir, f'extremities/{side}{subject}_extremities_values.npy'))

    # reconstruct np
    skel = convert_sparse_to_numpy(skel, coords, input_size)
    fold = convert_sparse_to_numpy(fold, coords, input_size)
    distb = convert_sparse_to_numpy(distb, coords, input_size)
    extr = convert_sparse_to_numpy(extr, coords, input_size)

    # zoom and save sparse
    skel = scipy.ndimage.zoom(skel, zoom_factor, order=0)
    s = sparse.COO.from_numpy(skel)
    np.save(os.path.join(save_dir, f'coords/{side}{subject}_coords.npy'), s.coords)
    np.save(os.path.join(save_dir, f'skeleton/{side}{subject}_skeleton_values.npy'), s.data)
    fold = scipy.ndimage.zoom(fold, zoom_factor, order=0)
    s = sparse.COO.from_numpy(fold)
    np.save(os.path.join(save_dir, f'foldlabel/{side}{subject}_foldlabel_values.npy'), s.data)
    distb = scipy.ndimage.zoom(distb, zoom_factor, order=0)
    s = sparse.COO.from_numpy(distb)
    np.save(os.path.join(save_dir, f'distbottom/{side}{subject}_distbottom_values.npy'), s.data)
    extr = scipy.ndimage.zoom(extr, zoom_factor, order=0)
    s = sparse.COO.from_numpy(extr)
    np.save(os.path.join(save_dir, f'extremities/{side}{subject}_extremities_values.npy'), s.data)


def process_in_parallel(subjects, process_function, num_worker):
    # Create a pool of worker processes
    with Pool(num_worker) as pool:
        # Use pool.map to apply the process_function to each image_path
        r = list(tqdm(pool.imap(process_function, subjects)))


def convert_sparse_to_numpy(data, coords, input_size):
    """
    Convert coords and associated values to numpy array
    """
    s = sparse.COO(coords, data, shape=input_size)
    arr = s.todense()

    return arr


if __name__ == '__main__':
    process_in_parallel(subjects, process_function, num_worker=30)