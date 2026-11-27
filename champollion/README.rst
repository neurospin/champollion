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

Run the command line (here the number of epochs is given, overriding the config default) :

.. code-block:: shell

    python3 train.py +dataset=config_dataset +dataset_localization=local max_epochs=81


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

Changes and configuration notes
===============================

CUDA platform
-------------

``configs/platform/cuda.yaml`` sets ``num_cpu_workers: 24`` (it was 20).

ResNet backbone
---------------

``configs/backbone/ResNet.yaml`` has a ``block`` key. ``basicblock`` (the
default) selects ``BasicBlock``; ``bottleneck`` selects ``Bottleneck``. Both
``ssl_model.py`` and ``evaluate.py`` read this key.

Resuming training
-----------------

``checkpoint_path`` (``configs/trainer/default.yaml``, empty by default) is
passed to ``trainer.fit`` as ``ckpt_path``, so a run can resume from a
checkpoint.

Sparse loading (``load_sparse``)
--------------------------------

With ``load_sparse: True`` (``configs/trainer/default.yaml``, default
``False``), the data module reads one file per subject from three folders of
each ``data`` entry: ``coords_all``, ``numpy_all`` (skeleton values) and
``foldlabel_all`` (fold label values). Each folder holds ``<sub>.npy``, where
``<sub>`` is the subject id. The old suffixed names ``_coords``,
``_skeleton_values`` and ``_foldlabel_values`` are no longer used. Every
``data`` entry must define ``coords_all``, and all three paths must be
folders, not files; otherwise a ``ValueError`` is raised. The mask is read
from ``data[0].mask_path``.

The whole-brain sparse path has no producer yet: nothing in this repository
or in the pipeline writes these per-subject ``<sub>.npy`` files.

Translation augmentation
------------------------

Three keys in ``configs/augmentations/mixed.yaml`` control translation of the
crop mask:

``offset_proba_tr`` (default 0.5)
    Threshold on a uniform random draw. A random translation is applied when
    the draw is above it.

``max_tr`` (default 0)
    Maximum radius, in voxels, of the random translation. A radius is drawn
    between 0 and ``max_tr`` and applied along a random direction. With 0
    no random translation happens.

``deterministic_translations`` (default null)
    Path to a CSV with the columns ``ID``, ``shift_x``, ``shift_y`` and
    ``shift_z``. The mask of subject ``ID`` is shifted by the negated values
    before any random translation. It applies only with ``load_sparse``.
