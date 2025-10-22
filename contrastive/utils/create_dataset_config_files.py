import pandas as pd
import os
import glob
import json

# %%
os.getcwd()

# %%
path = f"{os.getcwd()}/../configs/dataset/julien/TESTXX"
ref_file = f"{path}/reference.yaml"
crop_path = "/neurospin/dico/data/deep_folding/current/datasets/TESTXX/crops/2mm"
crop_dirs = glob.glob(f"{crop_path}/*")
crop_drops = []

crop_dirs = [f for f in crop_dirs if not os.path.basename(f) in crop_drops]

print(ref_file)
print('\n'.join(crop_dirs))

# Read in the reference file
with open(ref_file, 'r') as file:
  ref = file.read()

print(ref)


def replace_reference_yaml(crop_dir, side, ref):
  """For each crop name, it builds the yaml from the reference yaml"""
  crop_name = os.path.basename(crop_dir)
  dataset_name = crop_name.replace('.', '')
  mask_file = f"{crop_dir}/mask/{side}mask_cropped.nii.gz.minf"
  with open(mask_file, 'r') as file:
    mask = file.read()
  mask = mask.replace("attributes = ", "")
  mask = mask.replace("\'", "\"")
  # print(mask)
  mask_json = json.loads(mask)
  # print(mask_json)
  side = side
  side_long = "left" if side=='L' else "right"
  dataset_name = f"{dataset_name}_{side_long}"
  filedata = ref.replace('REPLACE_CROP_NAME', crop_name)
  filedata = filedata.replace('REPLACE_DATASET', dataset_name)
  filedata = filedata.replace('REPLACE_SIDE', side)
  filedata = filedata.replace('REPLACE_SIZEX', str(mask_json['sizeX']))
  filedata = filedata.replace('REPLACE_SIZEY', str(mask_json['sizeY']))
  filedata = filedata.replace('REPLACE_SIZEZ', str(mask_json['sizeZ']))

  result_file = f"{path}/{dataset_name}.yaml"

  return filedata, result_file

# Replace the target string
for crop_dir in crop_dirs:
    for side in ['L', 'R']:
        filedata, result_file = replace_reference_yaml(crop_dir, side, ref)
        print(result_file)
        with open(result_file, 'w') as file:
          file.write(filedata)



