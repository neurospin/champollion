
Champollion V1
==============

Self-supervised SSL models for generating embeddings of cortical folding patterns from T1 MRI brain scans.

This repository is used as a **git submodule** inside `champollion_pipeline <https://github.com/neurospin/champollion_pipeline>`_.

Pre-trained models are published on Hugging Face: `neurospin/Champollion_V1 <https://huggingface.co/neurospin/Champollion_V1>`_.


What it does
------------

Given preprocessed 3D brain crops of sulcal regions (ROIs), each model fold produces a fixed-size embedding vector per subject.
The repository covers 56 sulcal regions (28 regions × 2 hemispheres), matching the regions defined in
``champollion_pipeline/sulci_regions_champollion_V1.json``.

- **Architecture**: SSL (Barlow Twins) with a CNN backbone
- **Input**: 3D numpy crops of sulcal regions
- **Output**: Fixed-size embedding vectors (one per subject per region)
- **Training data**: UKBioBank (42,433 subjects)


Installation
------------

This submodule is installed automatically by ``champollion_pipeline``:

.. code-block:: shell

    git clone https://github.com/neurospin/champollion_pipeline.git
    cd champollion_pipeline
    pixi run install-all

It can also be installed as a standalone deep learning framework :

.. code-block:: shell

    git clone https://github.com/neurospin/champollion.git
    cd champollion
    virtualenv --python=python3 --system-site-packages venv
    . venv/bin/activate
    pip install -e .


Mask versions
-------------

Models are organised by ROI mask version inside the Hugging Face repo:

+-----------------------------+----------------------------------------------------------+
| Version                     | Description                                              |
+=============================+==========================================================+
| ``canonical_25``            | Original mask version used for the first training run.   |
+-----------------------------+----------------------------------------------------------+
| ``canonical_corrected_26_1``| Updated labelling with reduced region boundary artefacts.|
|                             | **Recommended for new datasets.**                        |
+-----------------------------+----------------------------------------------------------+

The mask version to use is selected via the ``--masks`` flag in ``run_cortical_tiles.py`` (step 3)
and must match the model version downloaded in step 5.


Configuration system
--------------------

The submodule uses Hydra-style YAML configs located in ``champollion/configs/``.


Training
--------

This repository can be used to train models on already preprocessed data.

An end-to-end preprocessing from MRI, training, and evaluation is performed by the pipeline, with ``train_champollion.py`` from ``champollion_pipeline``:

.. code-block:: shell

    pixi run python3 src/train_champollion.py \
        /path/to/crops/2mm \
        --dataset dataset_name \
        --region SC-sylv_left

See ``champollion_pipeline`` documentation for the full training workflow.

For details on the training architecture, augmentations, and evaluation scripts,
see `champollion/README.rst <champollion/README.rst>`_.


Repository structure
--------------------

.. code-block:: text

    champollion/
        champollion/
            configs/                    # Hydra-style YAML configs
            backbones/                  # CNN backbone definitions
            data/                       # Dataset and DataModule classes
            ssl_model.py                # SSL model
            evaluate.py                 # Embedding generation scripts
            train.py                    # Training entry point
        setup.cfg
        LICENSE


License
-------

Released under the `CeCILL-B <LICENSE>`_ license.
