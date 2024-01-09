import anatomist.api as anatomist
from soma import aims
import numpy as np
import os
import json
import pandas as pd

ana = anatomist.Anatomist()

dir_dHCP = '/neurospin/grip/external_databases/dHCP_CR_JD_2018/Projects/denis/release3_morpho_bids/'

"""
# 2023-12-18/16-56-49
clusters_beta_vae = {'cluster 0': ['CC00549XX22', 'CC00594XX18', 'CC00073XX08', 'CC00115XX08', 'CC00184XX12', 'CC00477XX16'],
                      'cluster 1': ['CC00403XX07', 'CC00189XX17', 'CC00933XX18', 'CC00155XX07', 'CC00101XX02', 'CC00305XX08'],
                      'cluster 2': ['CC00150AN02', 'CC00530XX11', 'CC00566XX14', 'CC00500XX05', 'CC00618XX16', 'CC00542XX15'],
                      'cluster 3': ['CC00907XX16', 'CC00284AN13', 'CC00947XX24', 'CC00418BN14', 'CC00337XX16', 'CC00425XX13']}
"""
#2023-12-11/14-10-52
clusters_beta_vae = {'cluster 0': ['CC00955XX15',
  'CC00353XX07',
  'CC00946XX23',
  'CC00538XX19',
  'CC00536XX17',
  'CC00395XX17'],
 'cluster 1': ['CC00446XX18',
  'CC01105XX08',
  'CC00314XX09',
  'CC00352XX06',
  'CC00508XX13',
  'CC00554XX10'],
 'cluster 2': ['CC00891XX18',
  'CC00838XX22',
  'CC00105XX06',
  'CC00551XX07',
  'CC00661XX10',
  'CC00845AN21'],
 'cluster 3': ['CC00589XX21',
  'CC00928XX21',
  'CC00164XX08',
  'CC00475XX14',
  'CC00082XX09',
  'CC00088XX15']}

cluster = 'cluster 0'

info_dHCP = pd.read_csv('/neurospin/dico/jlaval/data/info_dHCP.csv')
info_dHCP.drop(info_dHCP[~(info_dHCP['participant_id'].isin(clusters_beta_vae[cluster]))].index, inplace = True)
info_dHCP.reset_index(drop=True, inplace=True)

bck_list_1 = []
bck_list_2 = []
window_list = []

ana.loadObject('/casa/host/build/share/brainvisa-share-5.2/nomenclature/hierarchy/sulcal_root_colors.hie')

for idx, (id, session, _, _) in info_dHCP.iterrows():
    dir_folds = dir_dHCP + f'sub-{id}/ses-{session}/anat/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto/R{id}_default_session_auto.arg'
    dir_mesh = dir_dHCP + f'sub-{id}/ses-{session}/anat/t1mri/default_acquisition/default_analysis/segmentation/mesh/{id}_Rwhite.gii'
    print(dir_folds)

    bck_list_1.append(ana.loadObject(dir_folds))
    bck_list_2.append(ana.loadObject(dir_mesh))
    window_list.append(ana.createWindow('3D'))

print(len(bck_list_1), len(bck_list_2), len(window_list))
for window, bck1, bck2 in zip(window_list, bck_list_1, bck_list_2):
    #bck1.setPalette(palette)
    window.addObjects(bck1)
    window.addObjects(bck2)

input('Press a key to continue')