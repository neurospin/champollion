# import os
# import yaml
# import json
# import omegaconf
# import inspect

# from generate_embeddings import compute_embeddings
# from train_multiple_classifiers import train_classifiers
# from utils_pipelines import get_save_folder_name, change_config_datasets,\
#                             change_config_label, change_config_dataset_localization

# from sklearn.utils._testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning


# # Auxilary function used to process the config linked to the model.
# # For instance, change the embeddings save path to being next to the model.
# def preprocess_config(sub_dir, dataset_localization,
#                       datasets_root, datasets, idx_region_evaluation,
#                       label, folder_name, classifier_name='svm',
#                       epoch=None, split=None, cv=5,
#                       splits_basedir=None, verbose=False):
#     """Loads the associated config of the given model and changes what has to be done,
#     mainly the datasets, the classifier type and a few other keywords.
    
#     Arguments:
#         - sub_dir: str. Path to the directory containing the saved model.
#         - datasets: list of str. List of the datasets to be used for the results generation.
#         - label: str. Name of the label to be used for evaluation.
#         - folder_name: str. Name of the directory where to store both embeddings and aucs.
#         - classifier_name: str. Should correspond to a classifier yaml file's name 
#         (currently either 'svm' or 'neural_network').
#         - epoch: int. Specifies the epoch used for inference. Set to None to use the last epoch.
#         - verbose: bool. Verbose.
        
#     Output:
#         - cfg: the config as an omegaconf object."""

#     if verbose:
#         print(os.getcwd())
#     cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config.yaml')

#     # replace the datasets
#     change_config_datasets(cfg, datasets, datasets_root)
#     # replace the label
#     change_config_label(cfg, label)
#     # replace the dataset localizatyion
#     change_config_dataset_localization(cfg, dataset_localization)

#     # get the right classifiers parameters
#     with open(os.getcwd() + f'/configs/classifier/{classifier_name}.yaml', 'r') as file:
#         dataset_yaml = yaml.load(file, yaml.FullLoader)
#     for key in dataset_yaml:
#         cfg[key] = dataset_yaml[key]

#     # replace the possibly incorrect config parameters
#     cfg.model_path = sub_dir
#     cfg.embeddings_save_path = \
#         sub_dir + f"/{folder_name}_embeddings"
#     cfg.training_embeddings = \
#         sub_dir + f"/{folder_name}_embeddings"
#     cfg.apply_transformations = False
#     cfg.multiregion_single_encoder = False
#     cfg.load_sparse = False

#     # add epoch to config if specified
#     if epoch is not None:
#         cfg.epoch = epoch
#     # add splitting strategy to config
#     cfg.split = split
#     if split=='custom':
#         cfg.splits_basedir=splits_basedir
#     elif split=='random':
#         cfg.cv=cv

#     # in multi head case
#     if idx_region_evaluation is not None:
#         cfg.idx_region_evaluation=idx_region_evaluation

#     # change config partition to avoid errors
#     cfg.partition = [0.9,0.1]

#     return cfg


# def is_it_a_file(sub_dir):
#     if os.path.isdir(sub_dir):
#         return False
#     else:
#         print(f"{sub_dir} is a file. Continue.")
#         return True
    

# def is_folder_a_model(sub_dir):
#     if os.path.exists(sub_dir+'/.hydra/config.yaml'):
#         return True
#     else:
#         print(f"\n{sub_dir} not associated to a model. Continue")
#         return False
    

# def is_folder_accepted_model(sub_dir):
#     if '#' in sub_dir:
#         print(
#             "Model with an incompatible structure "
#             "with the current one, because there is # in the name."
#             "Pass."
#             )
#         return False
#     else:
#         return True


# def get_model_folder_name(epoch, folder_name):
#     if epoch is not None:
#         f_name = folder_name + f'_epoch{epoch}'
#     else:
#         f_name = folder_name
#     return f_name


# def print_config(cfg, verbose):
#     if verbose:
#         print("CONFIG FILE", type(cfg))
#         print(json.dumps(omegaconf.OmegaConf.to_container(
#             cfg, resolve=True), indent=4, sort_keys=True))


# def save_classifier_config(cfg, sub_dir):
#     # save the modified classifier config next to the real one
#     with open(sub_dir+'/.hydra/config_classifiers.yaml', 'w') \
#             as file:
#         yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)


# def reload_classifier_config(sub_dir):
#     # reload config for train_classifiers to work properly
#     cfg = omegaconf.OmegaConf.load(
#         sub_dir+'/.hydra/config_classifiers.yaml')
#     return cfg


# def check_if_compute_embedding(sub_dir, f_name, overwrite, embeddings, idx):
#     if (
#         os.path.exists(sub_dir + f"/{f_name}_embeddings")
#         and (not overwrite)
#     ):
#         print(f"Model {f_name} already treated "
#             "(existing folder with embeddings). "
#             "Set overwrite to True if you still want "
#             "to compute them.")
#         do_we_compute_embeddings = False
#         valid_path=True # assume that the embeddings exist
#     else:
#         # apply the functions
#         if embeddings and idx==0:
#             do_we_compute_embeddings = True
#             valid_path = False # will be set during embedding computation
#         elif not embeddings:
#             do_we_compute_embeddings = False
#             valid_path=True # assume that the embeddings exist 
#     return do_we_compute_embeddings, valid_path


# def do_we_classify(valid_path, embeddings_only):
#     if valid_path and not embeddings_only:
#         return True
#     elif not valid_path:
#         print('Invalid epoch number, skipped')
#         return False
#     else:
#         return False 


# # main function
# # creates embeddings and train classifiers for all models contained in folder
# @ignore_warnings(category=ConvergenceWarning)
# def embeddings_pipeline(dir_path, dataset_localization,
#                         datasets_root, datasets, idx_region_evaluation, labels,
#                         short_name=None, classifier_name='svm',
#                         overwrite=False, embeddings=True, embeddings_only=False,
#                         use_best_model=False, subsets=['full'],
#                         epochs=None, split='random', cv=5, splits_basedir=None, verbose=False):
#     """Pipeline to generate automatically the embeddings and compute the associated AUCs 
#     for all the models contained in a given directory. All the AUCs are computed with 
#     5-folds cross validation .

#     Arguments:
#         - dir_path: str. Path where the models are stored and where is applied 
#         recursively the process.
#         - dataset_localization: gives position of dataset
#         - datasets: list of str. Datasets the embeddings are generated from.
#         - labels: str list. Names of the labels to be used for evaluation.
#         - short_name: str or None. Name of the directory where to store both embeddings 
#         and aucs. If None, use datasets to generate the folder name.
#         - classifier_name: str. Parameter to select the desired classifer type
#         (currently neural_network or svm).
#         - overwrite: bool. Redo the process on models where embeddings already exist.
#         - embeddings: bool. Compute the embeddings, or use the ones previously computed.
#         - use_best_model: bool. Use the best model saved during to generate embeddings. 
#         The 'normal' model is always used, the best is only added.
#         - subsets: list of subsets you want the SVM to learn on. Set to ['full'] if you
#         want to learn on all subjects in one go.
#         - epoch: int. Specifies the epoch used for inference. Set to None to use the last epoch.
#         - verbose: bool. Verbose.
#     """

#     print("/!\\ Convergence warnings are disabled")

#     # Gets function parameters to call it recursively with same parameters
#     frame = inspect.currentframe()
#     args, _, _, values = inspect.getargvalues(frame)
#     args_function = {i: values[i] for i in args}     

#     # walks recursively through the subfolders
#     for name in os.listdir(dir_path):
#         sub_dir = dir_path + '/' + name
#         # checks if directory
#         if is_it_a_file(sub_dir):
#             pass
#         elif not is_folder_a_model(sub_dir):
#             args_function["dir_path"] = sub_dir
#             embeddings_pipeline(**args_function)
#         elif not is_folder_accepted_model(sub_dir):
#             pass
#         else:
#             print("\nTreating", sub_dir)

#             folder_name = get_save_folder_name(datasets=datasets,
#                                                short_name=short_name+'_'+split)

#             print("Start computing")

#             # Loops over labels
#             for idx, label in enumerate(labels):

#                 # Loops over epochs if requested
#                 for epoch in epochs:
#                     f_name = get_model_folder_name(epoch, folder_name)

#                     try:
#                         # Takes the model configuration
#                         # And updates it with input parameters
#                         cfg = preprocess_config(
#                             sub_dir,
#                             dataset_localization=dataset_localization,
#                             datasets_root=datasets_root,
#                             datasets=datasets,
#                             idx_region_evaluation=idx_region_evaluation,
#                             label=label,
#                             folder_name=f_name,
#                             classifier_name=classifier_name,
#                             epoch=epoch, split=split, cv=cv,
#                             splits_basedir=splits_basedir)
                        
#                         print_config(cfg, verbose)
#                         save_classifier_config(cfg, sub_dir)

#                         ####################
#                         # Compute embeddings
#                         ####################
#                         do_we_compute_embeddings, valid_path =\
#                             check_if_compute_embedding(sub_dir, f_name, overwrite,
#                                                     embeddings, idx)
#                         if do_we_compute_embeddings == True:
#                             valid_path = compute_embeddings(cfg, subsets=subsets)
                        
#                         ####################
#                         # Compute Classifier
#                         ####################
#                         cfg = reload_classifier_config(sub_dir)
#                         if do_we_classify(valid_path, embeddings_only):
#                             train_classifiers(cfg, subsets=subsets)


#                         #######################################
#                         # compute embeddings for the best model
#                         #######################################
#                         if (use_best_model and os.path.exists(sub_dir+'/logs/best_model_weights.pt')):
#                             print("\nCOMPUTE AGAIN WITH THE BEST MODEL\n")
#                             # apply the functions
#                             cfg = omegaconf.OmegaConf.load(
#                                 sub_dir+'/.hydra/config_classifiers.yaml')
#                             cfg.use_best_model = True
#                             if embeddings and idx==0:
#                                 _ = compute_embeddings(cfg, subsets=subsets)
#                             # reload config for train_classifiers to work properly
#                             cfg = omegaconf.OmegaConf.load(
#                                 sub_dir+'/.hydra/config_classifiers.yaml')
#                             cfg.use_best_model = True
#                             cfg.training_embeddings = cfg.embeddings_save_path + \
#                                 '_best_model'
#                             cfg.embeddings_save_path = \
#                                 cfg.embeddings_save_path + '_best_model'
#                             train_classifiers(cfg, subsets=subsets)
#                     except OSError as e:
#                         msg = str(e)
#                         if "] " in msg:
#                             msg = msg.split("] ", 1)[1]
#                         print("The following warning can be normal "
#                               f"if you have not generated this region in your dataset: {msg}")



# if __name__ == "__main__":

#     embeddings_pipeline("/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation",
#         dataset_localization="local",
#         datasets_root="TEST_DRABCZUK",
#         short_name='test_drabczuk',
#         overwrite=True,
#         datasets=["toto"],
#         idx_region_evaluation=None,
#         labels=["Sex"],
#         classifier_name='logistic',
#         embeddings=True, embeddings_only=True, use_best_model=False,
#         subsets=['full'], epochs=[None], split='random', cv=1,
#         splits_basedir='',
#         verbose=False) 


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Refactored embeddings_pipeline.py with a main function that parses arguments.
# """

# import os
# import yaml
# import json
# import omegaconf
# import inspect
# import argparse
# from generate_embeddings import compute_embeddings
# from train_multiple_classifiers import train_classifiers
# from utils_pipelines import get_save_folder_name, change_config_datasets, \
#                             change_config_label, change_config_dataset_localization
# from sklearn.utils._testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning

# # Auxiliary functions remain unchanged
# def preprocess_config(sub_dir, dataset_localization, datasets_root, datasets, idx_region_evaluation,
#                       label, folder_name, classifier_name='svm', epoch=None, split=None, cv=5,
#                       splits_basedir=None, verbose=False):
#     """Loads the associated config of the given model and changes what has to be done."""
#     if verbose:
#         print(os.getcwd())
#     cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config.yaml')
#     # replace the datasets
#     change_config_datasets(cfg, datasets, datasets_root)
#     # replace the label
#     change_config_label(cfg, label)
#     # replace the dataset localization
#     change_config_dataset_localization(cfg, dataset_localization)
#     # get the right classifiers parameters
#     with open(os.getcwd() + f'/configs/classifier/{classifier_name}.yaml', 'r') as file:
#         dataset_yaml = yaml.load(file, yaml.FullLoader)
#     for key in dataset_yaml:
#         cfg[key] = dataset_yaml[key]
#     # replace the possibly incorrect config parameters
#     cfg.model_path = sub_dir
#     cfg.embeddings_save_path = sub_dir + f"/{folder_name}_embeddings"
#     cfg.training_embeddings = sub_dir + f"/{folder_name}_embeddings"
#     cfg.apply_transformations = False
#     cfg.multiregion_single_encoder = False
#     cfg.load_sparse = False
#     # add epoch to config if specified
#     if epoch is not None:
#         cfg.epoch = epoch
#     # add splitting strategy to config
#     cfg.split = split
#     if split=='custom':
#         cfg.splits_basedir=splits_basedir
#     elif split=='random':
#         cfg.cv=cv
#     # in multi head case
#     if idx_region_evaluation is not None:
#         cfg.idx_region_evaluation=idx_region_evaluation
#     # change config partition to avoid errors
#     cfg.partition = [0.9,0.1]
#     return cfg

# # Other helper functions remain unchanged
# def is_it_a_file(sub_dir):
#     if os.path.isdir(sub_dir):
#         return False
#     else:
#         print(f"{sub_dir} is a file. Continue.")
#         return True

# def is_folder_a_model(sub_dir):
#     if os.path.exists(sub_dir+'/.hydra/config.yaml'):
#         return True
#     else:
#         print(f"\n{sub_dir} not associated to a model. Continue")
#         return False

# def is_folder_accepted_model(sub_dir):
#     if '#' in sub_dir:
#         print("Model with an incompatible structure with the current one, because there is # in the name. Pass.")
#         return False
#     else:
#         return True

# def get_model_folder_name(epoch, folder_name):
#     if epoch is not None:
#         f_name = folder_name + f'_epoch{epoch}'
#     else:
#         f_name = folder_name
#     return f_name

# def print_config(cfg, verbose):
#     if verbose:
#         print("CONFIG FILE", type(cfg))
#         print(json.dumps(omegaconf.OmegaConf.to_container(cfg, resolve=True), indent=4, sort_keys=True))

# def save_classifier_config(cfg, sub_dir):
#     # save the modified classifier config next to the real one
#     with open(sub_dir+'/.hydra/config_classifiers.yaml', 'w') as file:
#         yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)

# def reload_classifier_config(sub_dir):
#     # reload config for train_classifiers to work properly
#     cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config_classifiers.yaml')
#     return cfg

# def check_if_compute_embedding(sub_dir, f_name, overwrite, embeddings, idx):
#     if os.path.exists(sub_dir + f"/{f_name}_embeddings") and (not overwrite):
#         print(f"Model {f_name} already treated (existing folder with embeddings). Set overwrite to True if you still want to compute them.")
#         return False, True
#     else:
#         if embeddings and idx==0:
#             return True, False
#         elif not embeddings:
#             return False, True
#     return False, True


# def do_we_classify(valid_path, embeddings_only):
#     if valid_path and not embeddings_only:
#         return True
#     elif not valid_path:
#         print('Invalid epoch number, skipped')
#         return False
#     else:
#         return False


# def prompt_create_directory(directory_path):
#     """
#     Prompts the user to create a directory if it doesn’t exist.
#     Returns:
#         bool: True if the directory exists or was created, False otherwise.
#     """
#     if os.path.exists(directory_path):
#         return True

#     print(f"\n  {directory_path} does not exist.")
#     while True:
#         user_input = input("Would you like to create it? [y/n]: ").strip().lower()
#         if user_input == 'y':
#             try:
#                 os.makedirs(directory_path, exist_ok=True)
#                 print(f"Directory created: {directory_path}")
#                 return True
#             except OSError as e:
#                 print(f"Failed to create directory: {e}")
#                 return False
#         elif user_input == 'n':
#             print("Skipping directory creation.")
#             return False
#         else:
#             print("Please enter 'y' or 'n'.")


# @ignore_warnings(category=ConvergenceWarning)
# def embeddings_pipeline(models_path, dataset_localization, datasets_root, short_name,
#                         datasets=["toto"], idx_region_evaluation=None, labels=["Sex"],
#                         classifier_name='svm', overwrite=False, embeddings=True,
#                         embeddings_only=False, use_best_model=False, subsets=['full'],
#                         epochs=[None], split='random', cv=5, splits_basedir=None, verbose=False, ssh=False):
#     """Pipeline to generate automatically the embeddings and compute the associated AUCs."""
#     print("/!\\ Convergence warnings are disabled")
#     # Gets function parameters to call it recursively with same parameters
#     frame = inspect.currentframe()
#     args, _, _, values = inspect.getargvalues(frame)
#     args_function = {i: values[i] for i in args}
#     if not prompt_create_directory(models_path):
#         print("Aborting: directory was not created.")
#         exit(1)

#     # walks recursively through the subfolders
#     for name in os.listdir(models_path):
#         sub_dir = models_path + '/' + name
#         # checks if directory
#         if is_it_a_file(sub_dir):
#             pass
#         elif not is_folder_a_model(sub_dir):
#             args_function["models_path"] = sub_dir
#             embeddings_pipeline(**args_function)
#         elif not is_folder_accepted_model(sub_dir):
#             pass
#         else:
#             print("\nTreating", sub_dir)
#             folder_name = get_save_folder_name(datasets=datasets, short_name=short_name+'_'+split)
#             print("Start computing")
#             # Loops over labels
#             for idx, label in enumerate(labels):
#                 # Loops over epochs if requested
#                 for epoch in epochs:
#                     f_name = get_model_folder_name(epoch, folder_name)
#                     try:
#                         # Takes the model configuration
#                         # And updates it with input parameters
#                         cfg = preprocess_config(
#                             sub_dir,
#                             dataset_localization=dataset_localization,
#                             datasets_root=datasets_root,
#                             datasets=datasets,
#                             idx_region_evaluation=idx_region_evaluation,
#                             label=label,
#                             folder_name=f_name,
#                             classifier_name=classifier_name,
#                             epoch=epoch, split=split, cv=cv,
#                             splits_basedir=splits_basedir, verbose=verbose
#                         )

#                         print_config(cfg, verbose)
#                         save_classifier_config(cfg, sub_dir)
#                         ####################
#                         # Compute embeddings
#                         ####################
#                         do_we_compute_embeddings, valid_path = check_if_compute_embedding(sub_dir, f_name, overwrite, embeddings, idx)
#                         if do_we_compute_embeddings:
#                             valid_path = compute_embeddings(cfg, subsets=subsets)

#                         ####################
#                         # Compute Classifier
#                         ####################
#                         cfg = reload_classifier_config(sub_dir)
#                         if do_we_classify(valid_path, embeddings_only):
#                             train_classifiers(cfg, subsets=subsets)

#                         #######################################
#                         # compute embeddings for the best model
#                         #######################################
#                         if use_best_model and os.path.exists(sub_dir+'/logs/best_model_weights.pt'):
#                             print("\nCOMPUTE AGAIN WITH THE BEST MODEL\n")
#                             cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config_classifiers.yaml')
#                             cfg.use_best_model = True
#                             if embeddings and idx==0:
#                                 _ = compute_embeddings(cfg, subsets=subsets)
#                             cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config_classifiers.yaml')
#                             cfg.use_best_model = True
#                             cfg.training_embeddings = cfg.embeddings_save_path + '_best_model'
#                             cfg.embeddings_save_path = cfg.embeddings_save_path + '_best_model'
#                             train_classifiers(cfg, subsets=subsets)
#                     except OSError as e:
#                         msg = str(e)
#                         if "] " in msg:
#                             msg = msg.split("] ", 1)[1]
#                         print("The following warning can be normal if you have not generated this region in your dataset: {msg}")

# def main():
#     """Main function to parse command line arguments and call embeddings_pipeline."""
#     parser = argparse.ArgumentParser(
#         description="Generate embeddings and train classifiers for deep learning models."
#     )

#     # Required arguments
#     parser.add_argument(
#         "models_path",
#         type=str,
#         help="Path to the directory containing model folders."
#     )
#     parser.add_argument(
#         "dataset_localization",
#         type=str,
#         help="Key for dataset localization."
#     )
#     parser.add_argument(
#         "datasets_root",
#         type=str,
#         help="Root path to the dataset YAML configs."
#     )
#     parser.add_argument(
#         "short_name",
#         type=str,
#         help="Name of the directory where to store both embeddings and aucs."
#     )

#     # Optional arguments with defaults
#     parser.add_argument(
#         "--datasets",
#         type=str,
#         nargs="+",
#         default=["toto"],
#         help="List of dataset names (default: ['toto'])."
#     )
#     parser.add_argument(
#         "--labels",
#         type=str,
#         nargs="+",
#         default=["Sex"],
#         help="List of labels (default: ['Sex'])."
#     )
#     parser.add_argument(
#         "--classifier_name",
#         type=str,
#         default="svm",
#         help="Classifier name (default: 'svm')."
#     )
#     parser.add_argument(
#         "--overwrite",
#         action="store_true",
#         help="Overwrite existing embeddings (default: False)."
#     )
#     parser.add_argument(
#         "--embeddings_only",
#         action="store_true",
#         help="Only compute embeddings (skip classifiers, default: False)."
#     )
#     parser.add_argument(
#         "--use_best_model",
#         action="store_true",
#         help="Use the best model saved during training (default: False)."
#     )
#     parser.add_argument(
#         "--subsets",
#         type=str,
#         nargs="+",
#         default=["full"],
#         help="Subsets of data to train on (default: ['full'])."
#     )
#     parser.add_argument(
#         "--epochs",
#         type=str,
#         nargs="+",
#         default=["None"],
#         help="List of epochs to evaluate (default: [None], uses last epoch). Use 'None' for last epoch."
#     )
#     parser.add_argument(
#         "--split",
#         type=str,
#         default="random",
#         help="Splitting strategy ('random' or 'custom', default: 'random')."
#     )
#     parser.add_argument(
#         "--cv",
#         type=int,
#         default=5,
#         help="Number of cross-validation folds (default: 5)."
#     )
#     parser.add_argument(
#         "--splits_basedir",
#         type=str,
#         default="",
#         help="Directory for custom splits (default: None)."
#     )
#     parser.add_argument(
#         "--idx_region_evaluation",
#         type=int,
#         default=None,
#         help="Index of the region to evaluate (for multi-head models, default: None)."
#     )
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Enable verbose output (default: False)."
#     )

#     args = parser.parse_args()

#     # Convert epochs to proper format
#     epochs = []
#     for epoch in args.epochs:
#         if epoch.lower() == "none":
#             epochs.append(None)
#         else:
#             epochs.append(int(epoch))

#     # Call the embeddings_pipeline function
#     embeddings_pipeline(
#         models_path=args.models_path,
#         dataset_localization=args.dataset_localization,
#         datasets_root=args.datasets_root,
#         short_name=args.short_name,
#         datasets=args.datasets,
#         idx_region_evaluation=args.idx_region_evaluation,
#         labels=args.labels,
#         classifier_name=args.classifier_name,
#         overwrite=args.overwrite,
#         embeddings=True,  # Hardcoded as in the original function call
#         embeddings_only=args.embeddings_only,
#         use_best_model=args.use_best_model,
#         subsets=args.subsets,
#         epochs=epochs,
#         split=args.split,
#         cv=args.cv,
#         splits_basedir=args.splits_basedir if args.splits_basedir else None,
#         verbose=args.verbose,
#     )

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate embeddings and train classifiers for deep learning models.
"""

import os
import sys
import yaml
import json
import omegaconf
import inspect
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from os.path import abspath, dirname, join, exists
from champollion_utils.script_builder import ScriptBuilder
from generate_embeddings import compute_embeddings
from train_multiple_classifiers import train_classifiers
from utils_pipelines import get_save_folder_name, change_config_datasets, \
                          change_config_label, change_config_dataset_localization

# Add parent directory to path to import config_manager
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from config_manager import register_omegaconf_resolvers

# ====================== Population Strategy Pattern ======================

class PopulationStrategy(ABC):
    """Abstract base class for directory population strategies."""

    @abstractmethod
    def populate(self, source_path: str, target_path: str, **kwargs) -> bool:
        """Populate the target directory from the source."""
        pass

class LocalTarStrategy(PopulationStrategy):
    """Strategy to extract a local tar file."""

    def populate(self, source_path: str, target_path: str, **kwargs) -> bool:
        import tarfile
        if not exists(source_path):
            print(f"❌ Tar file not found: {source_path}")
            return False
        try:
            with tarfile.open(source_path, "r:*") as tar:
                tar.extractall(path=target_path)
            print(f"✅ Extracted {source_path} to {target_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to extract tar: {e}")
            return False

class HuggingFaceStrategy(PopulationStrategy):
    """Strategy to download from a Hugging Face repository."""

    def populate(self, source_path: str, target_path: str, **kwargs) -> bool:
        try:
            from huggingface_hub import snapshot_download
            token = kwargs.get("token", None)
            snapshot_download(
                repo_id=source_path,
                local_dir=target_path,
                local_dir_use_symlinks=False,
                token=token
            )
            print(f"✅ Downloaded {source_path} to {target_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to download from Hugging Face: {e}")
            return False

class DirectoryPopulator:
    """Context to apply a population strategy."""

    def __init__(self, strategy: PopulationStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: PopulationStrategy):
        """Change the strategy at runtime."""
        self._strategy = strategy

    def populate(self, source_path: str, target_path: str, **kwargs) -> bool:
        """Ensure the directory exists and apply the strategy."""
        if not self._ensure_directory(target_path):
            return False
        return self._strategy.populate(source_path, target_path, **kwargs)

    @staticmethod
    def _ensure_directory(path: str) -> bool:
        """Create the directory if it doesn't exist."""
        try:
            os.makedirs(path, exist_ok=True)
            print(f"✅ Directory ready: {path}")
            return True
        except OSError as e:
            print(f"❌ Failed to create directory: {e}")
            return False

# ====================== Helper Functions ======================

def is_it_a_file(sub_dir: str) -> bool:
    """Check if path is a file."""
    if os.path.isdir(sub_dir):
        return False
    print(f"{sub_dir} is a file. Continue.")
    return True

def is_folder_a_model(sub_dir: str) -> bool:
    """Check if directory contains a model."""
    if exists(join(sub_dir, '.hydra', 'config.yaml')):
        return True
    print(f"\n{sub_dir} not associated to a model. Continue")
    return False

def is_folder_accepted_model(sub_dir: str) -> bool:
    """Check if model folder has an accepted structure."""
    if '#' in sub_dir:
        print("Model with an incompatible structure with the current one, because there is # in the name. Pass.")
        return False
    return True

def get_model_folder_name(epoch: Optional[int], folder_name: str) -> str:
    """Generate model folder name with epoch if specified."""
    if epoch is not None:
        return f"{folder_name}_epoch{epoch}"
    return folder_name

def print_config(cfg: omegaconf.OmegaConf, verbose: bool) -> None:
    """Print configuration if verbose mode is enabled."""
    if verbose:
        print("CONFIG FILE", type(cfg))
        print(json.dumps(omegaconf.OmegaConf.to_container(cfg, resolve=True), indent=4, sort_keys=True))

def save_classifier_config(cfg: omegaconf.OmegaConf, sub_dir: str) -> None:
    """Save classifier configuration to file."""
    with open(join(sub_dir, '.hydra', 'config_classifiers.yaml'), 'w') as file:
        yaml.safe_dump(omegaconf.OmegaConf.to_container(cfg, resolve=True), file)

def reload_classifier_config(sub_dir: str) -> omegaconf.OmegaConf:
    """Reload classifier configuration."""
    return omegaconf.OmegaConf.load(join(sub_dir, '.hydra', 'config_classifiers.yaml'))

def check_if_compute_embedding(sub_dir: str, f_name: str, overwrite: bool, embeddings: bool, idx: int) -> tuple:
    """Check if embeddings need to be computed."""
    embeddings_path = join(sub_dir, f"{f_name}_embeddings")
    if exists(embeddings_path) and not overwrite:
        print(f"Model {f_name} already treated (existing folder with embeddings). Set overwrite to True if you still want to compute them.")
        return False, True

    if embeddings and idx == 0:
        return True, False
    return False, True

def do_we_classify(valid_path: bool, embeddings_only: bool) -> bool:
    """Determine if classification should be performed."""
    if valid_path and not embeddings_only:
        return True
    elif not valid_path:
        print('Invalid epoch number, skipped')
        return False
    return False

def preprocess_config(
    sub_dir: str,
    dataset_localization: str,
    datasets_root: str,
    datasets: List[str],
    idx_region_evaluation: Optional[int],
    label: str,
    folder_name: str,
    classifier_name: str = 'svm',
    epoch: Optional[int] = None,
    split: str = 'random',
    cv: int = 5,
    splits_basedir: Optional[str] = None,
    verbose: bool = False,
    config_path: Optional[str] = None,
    cpu: bool = False,
    nb_jobs: Optional[int] = None
) -> omegaconf.OmegaConf:
    """Load and update model configuration."""
    if verbose:
        print(os.getcwd())

    cfg = omegaconf.OmegaConf.load(join(sub_dir, '.hydra', 'config.yaml'))

    # Update configuration
    change_config_datasets(cfg, datasets, datasets_root, config_root=config_path)
    change_config_label(cfg, label)
    change_config_dataset_localization(cfg, dataset_localization)

    # Load classifier config
    with open(join(os.getcwd(), f'configs/classifier/{classifier_name}.yaml'), 'r') as file:
        dataset_yaml = yaml.safe_load(file)

    for key in dataset_yaml:
        cfg[key] = dataset_yaml[key]

    # Update paths and parameters
    cfg.model_path = sub_dir
    cfg.embeddings_save_path = join(sub_dir, f"{folder_name}_embeddings")
    cfg.training_embeddings = join(sub_dir, f"{folder_name}_embeddings")
    cfg.apply_transformations = False
    cfg.multiregion_single_encoder = False
    cfg.load_sparse = False

    if epoch is not None:
        cfg.epoch = epoch

    cfg.split = split
    if split == 'custom':
        cfg.splits_basedir = splits_basedir
    elif split == 'random':
        cfg.cv = cv

    if idx_region_evaluation is not None:
        cfg.idx_region_evaluation = idx_region_evaluation

    cfg.partition = [0.9, 0.1]

    # Override device if CPU mode is requested
    if cpu:
        cfg.device = 'cpu'

    # Override num_cpu_workers if specified via CLI
    if nb_jobs is not None:
        cfg.num_cpu_workers = nb_jobs

    # Safety cap: never exceed available CPUs
    try:
        available = len(os.sched_getaffinity(0))
        if cfg.num_cpu_workers > available:
            print(
                f"  Capping num_cpu_workers from "
                f"{cfg.num_cpu_workers} to {available}"
            )
            cfg.num_cpu_workers = available
    except (AttributeError, OSError):
        available = None

    # Container-only optimisations: reduce overhead for single-subject
    # inference on constrained hardware (e.g. 2-CPU HuggingFace Spaces).
    if os.environ.get("CHAMPOLLION_CONTAINER"):
        ncpus = available if available is not None else 2
        torch.set_num_threads(ncpus)
        torch.set_num_interop_threads(1)
        os.environ["OMP_NUM_THREADS"] = str(ncpus)
        os.environ["MKL_NUM_THREADS"] = str(ncpus)

        cfg.num_cpu_workers = 0   # in-process loading, no fork overhead
        cfg.pin_mem = False       # pin_memory is useless on CPU

        print(
            f"  Container mode: torch threads={ncpus}, "
            f"num_cpu_workers=0, pin_mem=False"
        )

    return cfg

def process_model(sub_dir: str, **kwargs: Dict[str, Any]) -> None:
    """Process a single model directory."""
    print("\nTreating", sub_dir)

    # Extract parameters from kwargs
    dataset_localization = kwargs['dataset_localization']
    datasets_root = kwargs['datasets_root']
    datasets = kwargs['datasets']
    short_name = kwargs['short_name']
    split = kwargs['split']
    labels = kwargs['labels']
    classifier_name = kwargs['classifier_name']
    overwrite = kwargs['overwrite']
    embeddings = kwargs['embeddings']
    embeddings_only = kwargs['embeddings_only']
    use_best_model = kwargs['use_best_model']
    subsets = kwargs['subsets']
    epochs = kwargs['epochs']
    cv = kwargs['cv']
    splits_basedir = kwargs['splits_basedir']
    verbose = kwargs['verbose']
    idx_region_evaluation = kwargs['idx_region_evaluation']
    config_path = kwargs.get('config_path')
    cpu = kwargs.get('cpu', False)
    nb_jobs = kwargs.get('nb_jobs')

    folder_name = get_save_folder_name(datasets=datasets, short_name=f"{short_name}_{split}")
    print("Start computing")

    for idx, label in enumerate(labels):
        for epoch in epochs:
            f_name = get_model_folder_name(epoch, folder_name)
            try:
                # Preprocess config
                cfg = preprocess_config(
                    sub_dir=sub_dir,
                    dataset_localization=dataset_localization,
                    datasets_root=datasets_root,
                    datasets=datasets,
                    idx_region_evaluation=idx_region_evaluation,
                    label=label,
                    folder_name=f_name,
                    classifier_name=classifier_name,
                    epoch=epoch,
                    split=split,
                    cv=cv,
                    splits_basedir=splits_basedir,
                    verbose=verbose,
                    config_path=config_path,
                    cpu=cpu,
                    nb_jobs=nb_jobs
                )

                print_config(cfg, verbose)
                save_classifier_config(cfg, sub_dir)

                # Compute embeddings
                do_we_compute_embeddings, valid_path = check_if_compute_embedding(
                    sub_dir, f_name, overwrite, embeddings, idx
                )
                if do_we_compute_embeddings:
                    valid_path = compute_embeddings(cfg, subsets=subsets)

                # Train classifiers
                cfg = reload_classifier_config(sub_dir)
                if do_we_classify(valid_path, embeddings_only):
                    train_classifiers(cfg, subsets=subsets)

                # Process best model if exists
                best_model_path = join(sub_dir, 'logs', 'best_model_weights.pt')
                if use_best_model and exists(best_model_path):
                    print("\nCOMPUTE AGAIN WITH THE BEST MODEL\n")
                    cfg = reload_classifier_config(sub_dir)
                    cfg.use_best_model = True

                    if embeddings and idx == 0:
                        _ = compute_embeddings(cfg, subsets=subsets)

                    cfg.training_embeddings = f"{cfg.embeddings_save_path}_best_model"
                    cfg.embeddings_save_path = f"{cfg.embeddings_save_path}_best_model"
                    if do_we_classify(valid_path, embeddings_only):
                        train_classifiers(cfg, subsets=subsets)

            except OSError as e:
                msg = str(e)
                if "] " in msg:
                    msg = msg.split("] ", 1)[1]
                print(f"The following warning can be normal if you have not generated this region in your dataset: {msg}")

def walk_models(models_path: str, callback: callable, **kwargs: Dict[str, Any]) -> None:
    """Recursively walk through models_path and apply callback to each valid model folder."""
    for name in os.listdir(models_path):
        sub_dir = join(models_path, name)

        if is_it_a_file(sub_dir):
            continue
        elif not is_folder_a_model(sub_dir):
            walk_models(sub_dir, callback, **kwargs)  # Recursive call
        elif not is_folder_accepted_model(sub_dir):
            continue
        else:
            callback(sub_dir, **kwargs)

# ====================== Main Script Class ======================

class RunEmbeddingsPipeline(ScriptBuilder):
    """Script for running embeddings pipeline to generate embeddings and train classifiers."""

    def __init__(self):
        super().__init__(
            script_name="run_embeddings_pipeline",
            description="Generating embeddings and training classifiers for deep learning models."
        )

        # Configure arguments using method chaining
        self.add_required_argument("--models_path", "Path to the directory containing model folders.")
        self.add_required_argument("--dataset_localization", "Key for dataset localization.")
        self.add_required_argument("--datasets_root", "Root path to the dataset YAML configs.")
        self.add_required_argument("--short_name", "Name of the directory where to store both embeddings and aucs.")

        # Use parser.add_argument directly for list arguments with nargs
        self.parser.add_argument("--datasets", nargs='+', default=["toto"], type=str, help="List of dataset names")
        self.parser.add_argument("--labels", nargs='+', default=["Sex"], type=str, help="List of labels")
        self.parser.add_argument("--subsets", nargs='+', default=["full"], type=str, help="Subsets of data to train on")
        self.parser.add_argument("--epochs", nargs='+', default=["None"], type=str, help="List of epochs to evaluate")

        # Continue with method chaining for non-list arguments
        (self.add_optional_argument("--classifier_name", "Classifier name", default="svm", type_=str)
         .add_flag("--overwrite", "Overwrite existing embeddings")
         .add_flag("--embeddings_only", "Only compute embeddings (skip classifiers)")
         .add_flag("--use_best_model", "Use the best model saved during training")
         .add_optional_argument("--split", "Splitting strategy ('random' or 'custom')", default="random", type_=str)
         .add_optional_argument("--cv", "Number of cross-validation folds", default=5, type_=int)
         .add_optional_argument("--splits_basedir", "Directory for custom splits", default="", type_=str)
         .add_optional_argument("--idx_region_evaluation", "Index of the region to evaluate (for multi-head models)", default=None, type_=int)
         .add_flag("--verbose", "Enable verbose output")
         .add_optional_argument("--population_source", "Source for directory population ('local' or 'huggingface')", default=None, type_=str)
         .add_optional_argument("--population_source_path", "Path to the source for population (tar file or HF repo)", default=None, type_=str)
         .add_optional_argument("--hf_token", "Hugging Face token for private repositories", default=None, type_=str)
         .add_optional_argument("--config_path", "Path to dataset config directory", default=None, type_=str)
         .add_flag("--cpu", "Force CPU usage (disable CUDA)")
         .add_optional_argument("--nb_jobs", "Number of CPU workers for DataLoader", default=None, type_=int))

    @ignore_warnings(category=ConvergenceWarning)
    def run(self) -> int:
        """Execute the embeddings pipeline script."""
        print("/!\\ Convergence warnings are disabled")

        # Convert epochs to proper format
        epochs = []
        for epoch in self.args.epochs:
            if epoch.lower() == "none":
                epochs.append(None)
            else:
                epochs.append(int(epoch))

        # Population step BEFORE directory validation
        if self.args.population_source:
            populator = DirectoryPopulator(LocalTarStrategy())  # Default to local tar

            if self.args.population_source == "huggingface":
                populator.set_strategy(HuggingFaceStrategy())

            success = populator.populate(
                source_path=self.args.population_source_path,
                target_path=self.args.models_path,
                token=self.args.hf_token
            )

            if not success:
                print("❌ Failed to populate directory. Continuing with existing content.")

        # Validate paths after population
        if not exists(self.args.models_path):
            raise ValueError(f"Models path does not exist: {self.args.models_path}")

        # Prepare kwargs for the callback
        callback_kwargs = {
            'dataset_localization': self.args.dataset_localization,
            'datasets_root': self.args.datasets_root,
            'datasets': self.args.datasets,
            'short_name': self.args.short_name,
            'split': self.args.split,
            'labels': self.args.labels,
            'classifier_name': self.args.classifier_name,
            'overwrite': self.args.overwrite,
            'embeddings': True,
            'embeddings_only': self.args.embeddings_only,
            'use_best_model': self.args.use_best_model,
            'subsets': self.args.subsets,
            'epochs': epochs,
            'cv': self.args.cv,
            'splits_basedir': self.args.splits_basedir if self.args.splits_basedir else None,
            'verbose': self.args.verbose,
            'idx_region_evaluation': self.args.idx_region_evaluation,
            'config_path': self.args.config_path,
            'cpu': self.args.cpu,
            'nb_jobs': self.args.nb_jobs
        }

        # Traverse directory and process models
        walk_models(self.args.models_path, process_model, **callback_kwargs)

        return 0

def main() -> int:
    """Main entry point."""
    script = RunEmbeddingsPipeline()
    return script.build().print_args().run()

if __name__ == "__main__":
    # Register custom OmegaConf resolvers
    register_omegaconf_resolvers()
    exit(main())
