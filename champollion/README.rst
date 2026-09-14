Description of each folder / file
=================================

configs
-------
yaml hydra-like configuration files.

backbones
---------
Definition of neural network backbones, currently a 3D 12-layer lightweight convnet, and a heavier 3D resnet18.

data
----
Definition of datasets and data modules.

augmentations.py
----------------
File containing the augmentation classes used for augmentation-based SSL (Barlow Twins, SimCLR).

losses.py
---------
File containing loss functions (Barlow Twins, SimCLR).

ssl_model.py
------------
File containing the model class.

train.py
--------
Script to launch the training.

evaluate.py
----------
Script to generate the embeddings of the models.


Tutorial: train a Champollion-like SSL model
============================================

Run the command line (here the number of epochs is given, overwritting the config default) :

.. code-block:: shell

    python3 train.py +dataset=config_dataset +dataset_localization=local --max_epochs=81


Tutorial: generate embeddings
=============================

.. code-block:: shell

    python3 evaluate.py --model_path ${DIR_MODEL} --skels_path ${DIR_DATA_NPY} --subjects_path ${DIR_DATA_ID} --saving_path ${DIR_OUTPUT}


Tutorial: CKA coherence testing
================================

To measure the similarity between representations learned by different models,
CKA (Centered Kernel Alignment) can be used. The module ``metrics/cka_coherence.py``
computes pairwise linear CKA scores between embedding matrices.

Given a directory containing multiple models with ``full_embeddings.csv`` files,
the convenience function discovers all embeddings, aligns subjects across models,
and produces a pairwise CKA similarity matrix.

.. code-block:: python

    from champollion.metrics.cka_coherence import test_models_coherence_from_directory

    cka_matrix, stats = test_models_coherence_from_directory(
        models_dir='path/to/models',
        embedding_filename='full_embeddings.csv'
    )
    print(f"Mean coherence: {stats['mean_cka']:.4f}")