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
#         dataset_localization="neurospin",
#         datasets_root="julien/TESTXX",
#         short_name='testxx',
#         overwrite=True,
#         datasets=["toto"],
#         idx_region_evaluation=None,
#         labels=["Sex"],
#         classifier_name='logistic',
#         embeddings=True, embeddings_only=True, use_best_model=False,
#         subsets=['full'], epochs=[None], split='random', cv=1,
#         splits_basedir='',
#         verbose=False) 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline to generate embeddings and train classifiers for all models in a directory.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline to generate embeddings and train classifiers for all models in a directory.
"""

import os
import yaml
import json
import argparse
import omegaconf
import inspect
from generate_embeddings import compute_embeddings
from train_multiple_classifiers import train_classifiers
from utils_pipelines import (
    get_save_folder_name,
    change_config_datasets,
    change_config_label,
    change_config_dataset_localization,
)
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# --- Helper Functions ---
def preprocess_config(
    sub_dir,
    dataset_localization,
    datasets_root,
    datasets,
    idx_region_evaluation,
    label,
    folder_name,
    classifier_name="svm",
    epoch=None,
    split=None,
    cv=5,
    splits_basedir=None,
    verbose=False,
    config_path=None,
):
    """Load and update the model config for embedding generation and classifier training."""
    cfg = omegaconf.OmegaConf.load(os.path.join(sub_dir, ".hydra", "config.yaml"))

    # Update datasets, labels, and localization
    change_config_datasets(cfg, datasets, datasets_root)
    change_config_label(cfg, label)
    change_config_dataset_localization(cfg, dataset_localization)

    # Load classifier config
    with open(os.path.join(config_path, f"{classifier_name}.yaml"), "r") as file:
        classifier_yaml = yaml.load(file, yaml.FullLoader)
    for key in classifier_yaml:
        cfg[key] = classifier_yaml[key]

    # Update paths and settings
    cfg.model_path = sub_dir
    cfg.embeddings_save_path = os.path.join(sub_dir, f"{folder_name}_embeddings")
    cfg.training_embeddings = os.path.join(sub_dir, f"{folder_name}_embeddings")
    cfg.apply_transformations = False
    cfg.multiregion_single_encoder = False
    cfg.load_sparse = False

    if epoch is not None:
        cfg.epoch = epoch
    cfg.split = split
    if split == "custom":
        cfg.splits_basedir = splits_basedir
    elif split == "random":
        cfg.cv = cv
    if idx_region_evaluation is not None:
        cfg.idx_region_evaluation = idx_region_evaluation
    cfg.partition = [0.9, 0.1]

    return cfg

def is_it_a_file(sub_dir):
    """Check if the path is a file (not a directory)."""
    if os.path.isdir(sub_dir):
        return False
    print(f"{sub_dir} is a file. Continue.")
    return True

def is_folder_a_model(sub_dir):
    """Check if the folder contains a Hydra config (i.e., is a model folder)."""
    if os.path.exists(os.path.join(sub_dir, ".hydra", "config.yaml")):
        return True
    print(f"\n{sub_dir} not associated with a model. Continue.")
    return False

def is_folder_accepted_model(sub_dir):
    """Check if the model folder is compatible (no '#' in the name)."""
    if "#" in sub_dir:
        print("Model with an incompatible structure. Pass.")
        return False
    return True

def get_model_folder_name(epoch, folder_name):
    """Generate the model folder name, optionally including the epoch."""
    return f"{folder_name}_epoch{epoch}" if epoch is not None else folder_name

def print_config(cfg, verbose):
    """Print the config in a readable format."""
    if verbose:
        print("CONFIG FILE", type(cfg))
        print(json.dumps(omegaconf.OmegaConf.to_container(cfg, resolve=True), indent=4, sort_keys=True))

def save_classifier_config(cfg, sub_dir):
    """Save the modified classifier config."""
    os.makedirs(os.path.join(sub_dir, ".hydra"), exist_ok=True)
    with open(os.path.join(sub_dir, ".hydra", "config_classifiers.yaml"), "w") as file:
        yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)

def reload_classifier_config(sub_dir):
    """Reload the classifier config for training."""
    return omegaconf.OmegaConf.load(os.path.join(sub_dir, ".hydra", "config_classifiers.yaml"))

def check_if_compute_embedding(sub_dir, f_name, overwrite, embeddings, idx):
    """Check if embeddings need to be computed."""
    if os.path.exists(os.path.join(sub_dir, f"{f_name}_embeddings")) and not overwrite:
        print(f"Model {f_name} already treated. Set overwrite=True to recompute.")
        return False, True
    if embeddings and idx == 0:
        return True, False
    return False, True

def do_we_classify(valid_path, embeddings_only):
    """Check if classifiers should be trained."""
    return valid_path and not embeddings_only

# --- Main Pipeline ---
@ignore_warnings(category=ConvergenceWarning)
def embeddings_pipeline(
    dir_path,
    dataset_localization,
    datasets_root,
    datasets,
    idx_region_evaluation,
    labels,
    short_name=None,
    classifier_name="svm",
    overwrite=False,
    embeddings=True,
    embeddings_only=False,
    use_best_model=False,
    subsets=["full"],
    epochs=[None],
    split="random",
    cv=5,
    splits_basedir=None,
    verbose=False,
    config_path=None,
):
    """Generate embeddings and train classifiers for all models in `dir_path`."""
    print("/!\\ Convergence warnings are disabled")

    # Recursively process subfolders
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    args_function = {i: values[i] for i in args}

    for name in os.listdir(dir_path):
        sub_dir = os.path.join(dir_path, name)
        if is_it_a_file(sub_dir):
            continue
        if not is_folder_a_model(sub_dir):
            args_function["dir_path"] = sub_dir
            embeddings_pipeline(**args_function)
            continue
        if not is_folder_accepted_model(sub_dir):
            continue

        print(f"\nTreating {sub_dir}")
        folder_name = get_save_folder_name(datasets=datasets, short_name=f"{short_name}_{split}")

        for idx, label in enumerate(labels):
            for epoch in epochs:
                f_name = get_model_folder_name(epoch, folder_name)
                try:
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

                    # Optionally use the best model
                    if use_best_model and os.path.exists(os.path.join(sub_dir, "logs", "best_model_weights.pt")):
                        print("\nCOMPUTE AGAIN WITH THE BEST MODEL\n")
                        cfg = reload_classifier_config(sub_dir)
                        cfg.use_best_model = True
                        if embeddings and idx == 0:
                            compute_embeddings(cfg, subsets=subsets)
                        cfg.training_embeddings = f"{cfg.embeddings_save_path}_best_model"
                        cfg.embeddings_save_path = f"{cfg.embeddings_save_path}_best_model"
                        train_classifiers(cfg, subsets=subsets)

                except OSError as e:
                    msg = str(e).split("] ", 1)[-1] if "] " in str(e) else str(e)
                    print(f"Warning (can be normal): {msg}")

# --- Main Function ---
def main():
    """Entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Generate embeddings and train classifiers.")

    # Required arguments
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to classifier configs (e.g., '/path/to/classifier/configs').",
    )
    parser.add_argument(
        "--dir_path",
        type=str,
        required=True,
        help="Path to model folders (e.g., '/path/to/models').",
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        required=True,
        help="Root path to dataset YAML configs (e.g., '/path/to/champollion_config_data').",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--dataset_localization",
        type=str,
        default="neurospin",
        help="Key for dataset localization (default: 'neurospin').",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["toto"],
        help="List of dataset names (default: ['toto']).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=["Sex"],
        help="List of labels (default: ['Sex']).",
    )
    parser.add_argument(
        "--short_name",
        type=str,
        default=None,
        help="Custom output folder name (default: derived from datasets).",
    )
    parser.add_argument(
        "--classifier_name",
        type=str,
        default="svm",
        help="Classifier name (default: 'svm').",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embeddings (default: False).",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Compute embeddings (default: True).",
    )
    parser.add_argument(
        "--embeddings_only",
        action="store_true",
        help="Only compute embeddings (skip classifiers, default: False).",
    )
    parser.add_argument(
        "--use_best_model",
        action="store_true",
        help="Use the best model saved during training (default: False).",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["full"],
        help="Subsets of data to train on (default: ['full']).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[None],
        help="List of epochs to evaluate (default: [None], uses last epoch).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="random",
        help="Splitting strategy ('random' or 'custom', default: 'random').",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--splits_basedir",
        type=str,
        default=None,
        help="Directory for custom splits (default: None).",
    )
    parser.add_argument(
        "--idx_region_evaluation",
        type=int,
        default=None,
        help="Index of the region to evaluate (for multi-head models, default: None).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (default: False).",
    )

    args = parser.parse_args()

    # Convert epochs to list of ints or None
    epochs = [None if epoch == "None" else epoch for epoch in args.epochs] if args.epochs else [None]

    embeddings_pipeline(
        dir_path=args.dir_path,
        dataset_localization=args.dataset_localization,
        datasets_root=args.datasets_root,
        datasets=args.datasets,
        idx_region_evaluation=args.idx_region_evaluation,
        labels=args.labels,
        short_name=args.short_name,
        classifier_name=args.classifier_name,
        overwrite=args.overwrite,
        embeddings=args.embeddings,
        embeddings_only=args.embeddings_only,
        use_best_model=args.use_best_model,
        subsets=args.subsets,
        epochs=epochs,
        split=args.split,
        cv=args.cv,
        splits_basedir=args.splits_basedir,
        verbose=args.verbose,
        config_path=args.config_path,
    )

if __name__ == "__main__":
    main()
