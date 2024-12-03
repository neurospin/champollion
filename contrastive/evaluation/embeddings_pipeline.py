import os
import yaml
import json
import omegaconf

from generate_embeddings import compute_embeddings
from train_multiple_classifiers import train_classifiers
from utils_pipelines import get_save_folder_name, change_config_datasets,\
                            change_config_label, change_config_dataset_localization

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


# Auxilary function used to process the config linked to the model.
# For instance, change the embeddings save path to being next to the model.
def preprocess_config(sub_dir, dataset_localization, datasets, label, folder_name, classifier_name='svm',
                      epoch=None, split=None, cv=5, splits_basedir=None, verbose=False):
    """Loads the associated config of the given model and changes what has to be done,
    mainly the datasets, the classifier type and a few other keywords.
    
    Arguments:
        - sub_dir: str. Path to the directory containing the saved model.
        - datasets: list of str. List of the datasets to be used for the results generation.
        - label: str. Name of the label to be used for evaluation.
        - folder_name: str. Name of the directory where to store both embeddings and aucs.
        - classifier_name: str. Should correspond to a classifier yaml file's name 
        (currently either 'svm' or 'neural_network').
        - epoch: int. Specifies the epoch used for inference. Set to None to use the last epoch.
        - verbose: bool. Verbose.
        
    Output:
        - cfg: the config as an omegaconf object."""

    if verbose:
        print(os.getcwd())
    cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config.yaml')

    # replace the datasets
    change_config_datasets(cfg, datasets)
    # replace the label
    change_config_label(cfg, label)
    # replace the dataset localizatyion
    change_config_dataset_localization(cfg, dataset_localization)

    # get the right classifiers parameters
    with open(os.getcwd() + f'/configs/classifier/{classifier_name}.yaml', 'r') as file:
        dataset_yaml = yaml.load(file, yaml.FullLoader)
    for key in dataset_yaml:
        cfg[key] = dataset_yaml[key]

    # replace the possibly incorrect config parameters
    cfg.model_path = sub_dir
    cfg.embeddings_save_path = \
        sub_dir + f"/{folder_name}_embeddings"
    cfg.training_embeddings = \
        sub_dir + f"/{folder_name}_embeddings"
    cfg.apply_transformations = False

    # add epoch to config if specified
    if epoch is not None:
        cfg.epoch = epoch
    # add splitting strategy to config
    cfg.split = split
    if split=='custom':
        cfg.splits_basedir=splits_basedir
    elif split=='random':
        cfg.cv=cv


    return cfg


# main function
# creates embeddings and train classifiers for all models contained in folder
@ignore_warnings(category=ConvergenceWarning)
def embeddings_pipeline(dir_path, dataset_localization, datasets, labels,
                        short_name=None, classifier_name='svm',
                        overwrite=False, embeddings=True, embeddings_only=False,
                        use_best_model=False, subsets=['full'],
                        epochs=None, split='random', cv=5, splits_basedir=None, verbose=False):
    """Pipeline to generate automatically the embeddings and compute the associated AUCs 
    for all the models contained in a given directory. All the AUCs are computed with 
    5-folds cross validation .

    Arguments:
        - dir_path: str. Path where the models are stored and where is applied 
        recursively the process.
        - dataset_localization: gives position of dataset
        - datasets: list of str. Datasets the embeddings are generated from.
        - labels: str list. Names of the labels to be used for evaluation.
        - short_name: str or None. Name of the directory where to store both embeddings 
        and aucs. If None, use datasets to generate the folder name.
        - classifier_name: str. Parameter to select the desired classifer type
        (currently neural_network or svm).
        - overwrite: bool. Redo the process on models where embeddings already exist.
        - embeddings: bool. Compute the embeddings, or use the ones previously computed.
        - use_best_model: bool. Use the best model saved during to generate embeddings. 
        The 'normal' model is always used, the best is only added.
        - subsets: list of subsets you want the SVM to learn on. Set to ['full'] if you
        want to learn on all subjects in one go.
        - epoch: int. Specifies the epoch used for inference. Set to None to use the last epoch.
        - verbose: bool. Verbose.
    """

    print("/!\\ Convergence warnings are disabled")
    # walks recursively through the subfolders
    for name in os.listdir(dir_path):
        sub_dir = dir_path + '/' + name
        # checks if directory
        if os.path.isdir(sub_dir):
            # check if directory associated to a model
            if os.path.exists(sub_dir+'/.hydra/config.yaml'):
                print("\nTreating", sub_dir)

                # check if embeddings and ROC already computed
                # if already computed and don't want to overwrite, then pass
                # else apply the normal process
                folder_name = get_save_folder_name(datasets=datasets, short_name=short_name+'_'+split)
                if (
                    os.path.exists(sub_dir + f"/{folder_name}_embeddings")
                    and (not overwrite)
                ):
                    print("Model already treated "
                          "(existing folder with embeddings). "
                          "Set overwrite to True if you still want "
                          "to compute them.")

                elif '#' in sub_dir:
                    print(
                        "Model with an incompatible structure "
                        "with the current one. Pass.")

                else:
                    print("Start post processing")
                    # get the config and correct it to suit
                    # what is needed for classifiers
                    for idx, label in enumerate(labels):
                        for epoch in epochs:
                            if epoch is not None:
                                f_name = folder_name + f'_epoch{epoch}'
                            else:
                                f_name = folder_name
                            cfg = preprocess_config(sub_dir,
                                                    dataset_localization=dataset_localization,
                                                    datasets=datasets,
                                                    label=label,
                                                    folder_name=f_name,
                                                    classifier_name=classifier_name,
                                                    epoch=epoch, split=split, cv=cv,
                                                    splits_basedir=splits_basedir)
                            if verbose:
                                print("CONFIG FILE", type(cfg))
                                print(json.dumps(omegaconf.OmegaConf.to_container(
                                    cfg, resolve=True), indent=4, sort_keys=True))
                            # save the modified config next to the real one
                            with open(sub_dir+'/.hydra/config_classifiers.yaml', 'w') \
                                    as file:
                                yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)

                            # apply the functions
                            if embeddings and idx==0:
                                valid_path = compute_embeddings(cfg)
                            elif not embeddings:
                                valid_path=True # assume that the embeddings exist
                            # reload config for train_classifiers to work properly
                            cfg = omegaconf.OmegaConf.load(
                                sub_dir+'/.hydra/config_classifiers.yaml')
                            if valid_path and not embeddings_only:
                                train_classifiers(cfg, subsets=subsets)
                            elif not valid_path:
                                print('Invalid epoch number, skipped')

                            # compute embeddings for the best model if saved
                            if (use_best_model and os.path.exists(sub_dir+'/logs/best_model_weights.pt')):
                                print("\nCOMPUTE AGAIN WITH THE BEST MODEL\n")
                                # apply the functions
                                cfg = omegaconf.OmegaConf.load(
                                    sub_dir+'/.hydra/config_classifiers.yaml')
                                cfg.use_best_model = True
                                if embeddings and idx==0:
                                    _ = compute_embeddings(cfg)
                                # reload config for train_classifiers to work properly
                                cfg = omegaconf.OmegaConf.load(
                                    sub_dir+'/.hydra/config_classifiers.yaml')
                                cfg.use_best_model = True
                                cfg.training_embeddings = cfg.embeddings_save_path + \
                                    '_best_model'
                                cfg.embeddings_save_path = \
                                    cfg.embeddings_save_path + '_best_model'
                                train_classifiers(cfg, subsets=subsets)

            else:
                print(f"\n{sub_dir} not associated to a model. Continue")
                embeddings_pipeline(sub_dir,
                                    dataset_localization,
                                    datasets=datasets,
                                    labels=labels,
                                    short_name=short_name,
                                    classifier_name=classifier_name,
                                    overwrite=overwrite,
                                    embeddings=embeddings,
                                    embeddings_only=embeddings_only,
                                    use_best_model=use_best_model,
                                    subsets=subsets,
                                    epochs=epochs,
                                    split=split,
                                    cv=cv,
                                    splits_basedir=splits_basedir,
                                    verbose=verbose)
        else:
            print(f"{sub_dir} is a file. Continue.")

if __name__ == "__main__":
    # embeddings_pipeline("/neurospin/dico/jchavas/Runs/70_self-supervised_two-regions/Output/2024-06-21",
    #                     dataset_localization="neurospin",
    #                     datasets=["julien/MICCAI_2024/evaluation/orbital_left_hcp_custom"],
    #                     labels=['Left_OFC'],
    #                     short_name='troiani', overwrite=True, embeddings=True,
    #                     embeddings_only=False, use_best_model=False,
    #                     subsets=['full'], epochs=[None], split='custom', cv=3,
    #                     splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
    #                     verbose=False)

    embeddings_pipeline("/neurospin/dico/data/deep_folding/current/models/Champollion_V0/FIP_right",
                        dataset_localization="neurospin",
                        datasets=["with_reskel_distbottom/2mm/UKB40/FIP_right"],
                        labels=['Left_PCS'],
                        classifier_name='logistic',
                        short_name='three_datasets_UKB', overwrite=False, embeddings=True,
                        embeddings_only=True, use_best_model=False,
                        subsets=['full'], epochs=[0,10,20,30,40,50,60,70,80,90,100], split='random', cv=3,
                        splits_basedir='',
                        verbose=False)

    # embeddings_pipeline("/neurospin/dico/jchavas/Runs/70_self-supervised_two-regions/Output/ORBITAL_BT",
    #                     dataset_localization="neurospin",
    #                     datasets=["with_reskel_distbottom/2mm/schiz_extended/ORBITAL_left"],
    #                     labels=['diagnosis'],
    #                     short_name='schiz_extended', overwrite=True, embeddings=True,
    #                     embeddings_only=True, use_best_model=False,
    #                     subsets=['full'], epochs=[None], split='random', cv=3,
    #                     splits_basedir='',
    #                     verbose=False)
    
    # embeddings_pipeline("/neurospin/dico/jchavas/Runs/70_self-supervised_two-regions/Output/ORBITAL_BT",
    #                     dataset_localization="neurospin",
    #                     datasets=["with_reskel_distbottom/2mm/schiz_extended/ORBITAL_left"],
    #                     labels=['diagnosis'],
    #                     short_name='schiz_extended', overwrite=True, embeddings=True,
    #                     embeddings_only=True, use_best_model=True,
    #                     subsets=['full'], epochs=[None], split='random', cv=3,
    #                     splits_basedir='',
    #                     verbose=False)

    # embeddings_pipeline("/neurospin/dico/jchavas/Runs/70_self-supervised_two-regions/Output/2024-06-06_pretraining",
    #                     dataset_localization="neurospin",
    #                     datasets=["with_reskel_distbottom/2mm/schiz_extended/SC_SPeC_left_female",
    #                               "with_reskel_distbottom/2mm/schiz_extended/SC_SPeC_right_female"],
    #                     labels=['diagnosis'],
    #                     short_name='schiz_extended', overwrite=True, embeddings=True,
    #                     embeddings_only=False, use_best_model=False,
    #                     subsets=['full'], epochs=[None], split='random', cv=3,
    #                     splits_basedir='',
    #                     verbose=False)
    
    # embeddings_pipeline("/neurospin/dico/jchavas/Runs/70_self-supervised_two-regions/Output/2024-06-06_pretraining",
    #                     dataset_localization="neurospin",
    #                     datasets=["with_reskel_distbottom/2mm/UKB/SC_SPeC_left",
    #                               "with_reskel_distbottom/2mm/UKB/SC_SPeC_right"],
    #                     labels=['diagnosis'],
    #                     short_name='ukb', overwrite=True, embeddings=True,
    #                     embeddings_only=True, use_best_model=False,
    #                     subsets=['full'], epochs=[None], split='random', cv=3,
    #                     splits_basedir='',
    #                     verbose=False)



""" PCS
        datasets=["local_julien/cingulate_ACCpatterns_custom_cv_2mm"],
        labels=['Right_PCS'],
        short_name='ACC', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['test'], epochs=[None], split='custom', cv=3,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/ACCpatterns/ACCpatterns_subjects_train_split_',
        verbose=False)
"""

"""OFC
        datasets=["local_julien/orbital_left_hcp_custom_cv_2mm"],
        labels=['Left_OFC'],
        short_name='troiani', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['full'], epochs=[None], split='custom', cv=3,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
        verbose=False)
"""

"""STS Preterm
        datasets=["local_julien/1-5mm/STs_babies_dHCP_374_subjects_right_1-5mm"],
        labels=['Preterm_23-28_vs_fullterm'],
        short_name='dHCP', overwrite=True, embeddings=True, embeddings_only=False, use_best_model=False,
        subsets=['full'], epochs=range(0,250,10), split='random', cv=3,
        splits_basedir='/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/train_val_split_',
        verbose=False)
"""


#datasets=["local_julien/old/STs_dHCP_374_subjects"]
#split='random', 'custom'
#epochs=[None], range(0, 250, 10)
#subset=['full'], ['train_val']
#labels=['Preterm_28', 'Preterm_32', 'Preterm_37']
#labels=['Preterm_23-28_vs_fullterm']
#short_name='UKB_5percent'
#datasets=["local_julien/1-5mm/STs_babies_UKB_right_5percent_1-5mm"]
#datasets=["local_julien/1-5mm/STs_babies_dHCP_374_subjects_right_1-5mm"]

#    embeddings_pipeline("/volatile/jl277509/Runs/02_STS_babies/Program/Output/old_best/",
#        datasets=["local_julien/cingulate_UKB_right_5percent"],
#        labels=['Age', 'Age_64', 'Sex'],
#        short_name='UKB_5percent', overwrite=True, embeddings=True, use_best_model=False,
#        subsets=['train_val'], epochs=[None], verbose=False)


#datasets=["local_julien/cingulate_UKB_right"]
#datasets=["local_julien/cingulate_ACCpatterns_1
#label='Age_64'
#label='Right_PCS'
#short_name='UKB_1'
