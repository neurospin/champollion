
Champollion V1
==============

Self-supervised Barlow Twins models for generating embeddings of cortical folding patterns from T1 MRI brain scans.

This repository is used as a **git submodule** inside `champollion_pipeline <https://github.com/neurospin/champollion_pipeline>`_.

Pre-trained models are published on Hugging Face: `neurospin/Champollion_V1 <https://huggingface.co/neurospin/Champollion_V1>`_.


What it does
------------

Given preprocessed 3D brain crops of sulcal regions, each model fold produces a fixed-size embedding vector per subject.
The repository covers 56 sulcal regions (28 regions × 2 hemispheres), matching the regions defined in
``champollion_pipeline/sulci_regions_champollion_V1.json``.

- **Architecture**: Barlow Twins (self-supervised learning) with a CNN backbone
- **Input**: 3D numpy crops of sulcal regions
- **Output**: Fixed-size embedding vectors (one per subject per region)
- **Training data**: UKBioBank (42,433 subjects)


Installation
------------

This submodule is installed automatically by ``champollion_pipeline``:

.. code-block:: shell

    git clone https://github.com/neurospin/champollion.git
    cd champollion
    

Configuration system
--------------------

The submodule uses Hydra-style YAML configs located in ``contrastive/configs/``.

Two files are updated at runtime by ``generate_champollion_config.py`` (step 4 of the pipeline):

- ``contrastive/configs/local.yaml`` — sets ``dataset_folder`` to the crop directory on disk
- ``contrastive/configs/dataset_localization/local.yaml`` — selects the ``local`` localization preset

Pass ``--external-config`` in read-only environments (Apptainer/Docker) to write these files
to a writable path instead.


Mask versions
-------------

Models are organised by mask version inside the Hugging Face repo:

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


Training
--------

To retrain models on a new dataset, use ``train_champollion.py`` from ``champollion_pipeline``:

.. code-block:: shell

    pixi run python3 src/train_champollion.py \
        /path/to/crops/2mm \
        --dataset dataset_name \
        --region SC-sylv_left

See ``champollion_pipeline`` documentation for the full training workflow.

For details on the training architecture, augmentations, and evaluation scripts,
see `contrastive/README.rst <contrastive/README.rst>`_.


Repository structure
--------------------

.. code-block:: text

    champollion_V1/
        contrastive/
            configs/
                dataset_localization/
                    local.yaml          # Updated by generate_champollion_config.py
                local.yaml              # Updated by generate_champollion_config.py
            backbones/                  # CNN backbone definitions
            data/                       # Dataset and DataModule classes
            models/                     # Barlow Twins model definitions
            evaluation/                 # Embedding evaluation scripts
            train.py                    # Training entry point
        setup.cfg
        LICENSE


License
-------

Released under the `CeCILL-B <LICENSE>`_ license.
