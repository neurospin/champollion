
Self-supervised learning for preterm-specific variability of cortical folding in the right superior temporal sulcus region
###########################################################################

This repository is based on https://github.com/neurospin-projects/2023_agaudin_jchavas_folding_supervised. It aims to apply the self-supervised deep learning pipepline to preterm-specific folding pattern analysis and explore explainability methods.
Official Pytorch implementation for Unsupervised Learning and Cortical Folding (`paper <https://openreview.net/forum?id=ueRZzvQ_K6u>`_).


Dependencies
------------
- python >= 3.6
- pytorch >= 1.4.0
- numpy >= 1.16.6
- pandas >= 0.23.3


Set up the work environment
---------------------------
First, the repository can be cloned thanks to:

.. code-block:: shell

    git clone https://github.com/neurospin-projects/2023_jlaval_STSbabies/
    cd 2023_jlaval_STSbabies

Then, install a virtual environment through the following command lines:

.. code-block:: shell

    python3 -m venv venv
    . venv/bin/activate
    pip3 install --upgrade pip
    pip3 install -e .

Note that you might need a `BrainVISA <https://brainvisa.info>`_ environment to run
some of the functions or notebooks.

Preterm analysis requires training on UkBioBank, using SimCLR. A comprehensive description is given in contrastive/README.rst.

.. code-block:: shell

    cd contrastive
    python3 train.py mode=encoder

Once the model is trained, the model performances can be assessed using SVC running:

.. code-block:: shell

    cd contrastive
    python3 evaluation/embeddings_pipeline.py

To compute Grad-CAM heatmaps, we first train a linear classifier with the frozen self-supervised backbone.
It can be done on multiple iterrations of the UkBioBank-trained model serially using the command : 

.. code-block:: shell

    cd contrastive
    python3 train.py --multirun mode=classifier augmentations=no_augmentation label=Preterm_28 drop_rate=0.0 load_encoder_only=True freeze_encoders=True fusioned_latent_space_size=-1 projection_head=linear max_epochs=50 lr=0.01 early_stopping_patience=25 pretrained_model_path=\"/neurospin/dico/jlaval/Runs/02_STS_babies/Program/Output/2023-11-29/09-59-38_188/logs/lightning_logs/version_0/checkpoints/epoch=249-step=296250.ckpt\",\"/neurospin/dico/jlaval/Runs/02_STS_babies/Program/Output/2023-11-29/15-49-36_0/logs/lightning_logs/version_0/checkpoints/epoch=249-step=296250.ckpt\",\"/neurospin/dico/jlaval/Runs/02_STS_babies/Program/Output/2023-11-29/15-49-36_1/logs/lightning_logs/version_0/checkpoints/epoch=249-step=296250.ckpt\",\"/neurospin/dico/jlaval/Runs/02_STS_babies/Program/Output/2023-11-29/15-49-36_2/logs/lightning_logs/version_0/checkpoints/epoch=249-step=296250.ckpt\"

In the current version, this needs to be run twice by switching the train and test sets manually to perform cross-validation.

Then, the Grad-CAM heatmaps are obtained running:

.. code-block:: shell

    cd contrastive
    python3 evaluation/supervised_pipeline.py

And can be visualised in : contrastive/notebooks/julien/plot_grad_cam.ipynb
