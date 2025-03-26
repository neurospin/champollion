import numpy as np
from scipy import ndimage
import os
import networkx as nx
from soma import aims
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

n_cpus= 46
side = 'R'
dataset = 'hcp'
sulcus = 'F.I.P.'

root_dir = f'/neurospin/dico/data/deep_folding/current/datasets/{dataset}/crops/2mm/{sulcus}/mask/'
save_dir = os.path.join(root_dir, f'{side}ccdistmaps')
root_dir = os.path.join(root_dir, f'{side}crops')
subjects_list = os.listdir(root_dir)
subjects_list = [elem for elem in subjects_list if elem[-1]!='f'] # remove minf files from list

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)


## UTILS

def find_connected_components(volume, structure):
    """Label connected components in a 3D binary volume."""
    labeled_volume, num_components = ndimage.label(volume, structure=structure)  # Label each connected component
    return labeled_volume, num_components

def voxel_graph(component, label_value):
    """Create a graph from a single connected component."""
    G = nx.Graph()
    dims = component.shape
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if component[i, j, k] == label_value:
                    G.add_node((i, j, k))
                    #for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]: ## not the right connectivity ?
                    for di in range(-1,2,1):
                        for dj in range(-1,2,1):
                            for dk in range(-1,2,1):
                                ni, nj, nk = i + di, j + dj, k + dk
                                if 0 <= ni < dims[0] and 0 <= nj < dims[1] and 0 <= nk < dims[2] and not (ni==i and nj==j and nk==k): # and not (np.linalg.norm((ni-i, nj-j, nk-k))>1.5) # use connectivity 26
                                    if component[ni, nj, nk] == label_value:
                                        G.add_edge((i, j, k), (ni, nj, nk), weight=np.linalg.norm((ni-i, nj-j, nk-k)))
    return G

def find_minimax_center(G):
    """Find the node in the graph that minimizes the maximum geodesic distance to any other node."""
    lengths = dict(nx.all_pairs_dijkstra_path_length(G))  # Compute all-pairs shortest paths
    minimax_center = None
    min_max_dist = float('inf')

    for node in G.nodes():
        max_dist = max(lengths[node].values())  # Find the worst-case distance
        if max_dist < min_max_dist:
            min_max_dist = max_dist
            minimax_center = node
    
    return minimax_center, min_max_dist, lengths[minimax_center]


## Processing

def process_element(subject_dir, root_dir, save_dir):
    skel_volume = aims.read(os.path.join(root_dir, subject_dir))
    skel_volume.np[skel_volume.np!=0]=1
    #print(np.unique(skel_volume, return_counts=True))
    skel = skel_volume.np
    volume = skel[:,:,:,0]!=0
    structure = np.ones((3,3,3)) # we could also remove the edges
    #structure = None

    # Find connected components
    labeled_volume, num_components = find_connected_components(volume, structure=structure)
    #print(f'Number of connected components : {num_components}')
    #print(np.unique(labeled_volume, return_counts=True)[1])

    # Find minimax center for each connected component
    centers = {}
    for label_value in range(1, num_components + 1):  # Labels start from 1
        G = voxel_graph(labeled_volume, label_value)
        if G.number_of_nodes() > 0:
            center, max_dist, distances = find_minimax_center(G)
            centers[label_value] = (center, max_dist, distances)

    # fill distance_map with distances
    alpha=0.5
    distance_map = np.zeros(volume.shape)
    for label, (center, max_dist, distances) in centers.items():
        if max_dist==0: # avoid division by 0
            distance_map[center]=max_dist
        else:
            for coord, value in distances.items():
                distance_map[coord]=value*(1-alpha)/max_dist+alpha

    distance_map = np.expand_dims(distance_map, axis=-1).astype(np.float32)
    distance_vol = aims.Volume(distance_map)

    aims.write(distance_vol, os.path.join(save_dir, subject_dir))
    
    return None


def process_list(root_dir, save_dir, list_sub_dirs, n_cpus):
    # Create a partial function with y and z fixed
    partial_func = partial(process_element, root_dir=root_dir, save_dir=save_dir)
    
    # Create a pool of workers with n_cpus
    with mp.Pool(processes=n_cpus) as pool:
        # Map the partial function to the elements list
        results = list(tqdm(pool.imap(partial_func, list_sub_dirs), total=len(list_sub_dirs)))
    return results


process_list(root_dir, save_dir, subjects_list, n_cpus)
