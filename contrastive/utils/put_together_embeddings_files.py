
import os
import shutil
import logging
import tarfile
import pandas as pd

path_champollion = "/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation"
embeddings_subpath = "testxx_random_embeddings/full_embeddings.csv"
output_path = "/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation/embeddings/TESTXX_embeddings"

# %%
def is_it_a_file(sub_dir):
    if os.path.isdir(sub_dir):
        return False
    else:
        logging.debug(f"{sub_dir} is a file. Continue.")
        return True
    

def is_folder_a_model(sub_dir):
    if os.path.exists(sub_dir+'/.hydra/config.yaml'):
        return True
    else:
        logging.debug(f"\n{sub_dir} not associated to a model. Continue")
        return False

def get_model_paths(dir_path, result = None):
    """Recursively gets all models included in dir_path"""
    if result is None:  # create a new result if no intermediate was given
        result = [] 
    for name in os.listdir(dir_path):
        sub_dir = dir_path + '/' + name
        # checks if directory
        if is_it_a_file(sub_dir):
            pass
        elif not is_folder_a_model(sub_dir):
            result.extend(get_model_paths(sub_dir))
        else:
            result.append(sub_dir)
    return result

# %%
model_paths = get_model_paths(path_champollion)

# %%
model_paths

# %% [markdown]
# 

# %%
if not os.path.exists(output_path):
    os.mkdir(output_path)

# %%
for model_path in model_paths:
    file_input_name = f"{model_path}/{embeddings_subpath}"
    region = model_path.split('Champollion_V1_after_ablation/')[1].split('/')[0]
    model = model_path.split(region+'/')[1].replace("/", "--").replace("_", "--")
    file_output_name = f"{output_path}/{region}_{model}_embeddings.csv"
    try:
        shutil.copyfile(file_input_name, file_output_name)
    except OSError as e:
        msg = str(e)
        if "] " in msg:
            msg = msg.split("] ", 1)[1]
        print(f"The following warning can be normal if you have not generated this region in your dataset: {msg}")

# %%
# f = tarfile.open("/neurospin/dico/data/deep_folding/current/models/Champollion_V0/embeddings/ukb40_embeddings.tar", 'r')
# f.extractall("/neurospin/dico/data/deep_folding/current/models/Champollion_V0/embeddings/tmp")

# %%
# f

# %%



