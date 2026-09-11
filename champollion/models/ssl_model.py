#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
"""
Some helper functions are taken from:
https://learnopencv.com/tensorboard-with-pytorch-lightning

"""
import os
import json
import numpy as np
import torch
import pytorch_lightning as pl
from collections import OrderedDict

from champollion.backbones.convnet import ConvNet
from champollion.backbones.resnet import ResNet, BasicBlock
from champollion.backbones.projection_heads import *
from champollion.data.utils import change_list_device
from champollion.losses import *

from champollion.utils.logs import set_file_logger
log = set_file_logger(__file__)



class SSLModel(pl.LightningModule):

    def __init__(self, config, sample_data):
        super(SSLModel, self).__init__()

        n_datasets = len(config.data)
        log.info(f"n_datasets {n_datasets}")

        # define the encoder structure
        self.backbones = nn.ModuleList()
        if config.backbone_name == "convnet":
            for i in range(n_datasets):
                self.backbones.append(ConvNet(
                    in_channels=1,
                    encoder_depth=config.encoder_depth,
                    filters=config.filters,
                    block_depth=config.block_depth,
                    initial_kernel_size=config.initial_kernel_size,
                    initial_stride=config.initial_stride,
                    max_pool=config.max_pool,
                    num_representation_features=config.backbone_output_size,
                    adaptive_pooling=config.adaptive_pooling,
                    drop_rate=config.drop_rate,
                    in_shape=config.data[i].input_size))
        elif config.backbone_name == 'resnet':
            for i in range(n_datasets):
                self.backbones.append(ResNet(
                    in_channels=1,
                    block=BasicBlock,
                    out_block=None,
                    layers=config.layers,
                    channels=config.channels,
                    num_classes=config.backbone_output_size,
                    zero_init_residual=config.zero_init_residual,
                    dropout_rate=config.drop_rate,
                    initial_kernel_size=config.initial_kernel_size,
                    initial_stride=config.initial_stride,
                    adaptive_pooling=config.adaptive_pooling))
        else:
            raise ValueError(f"No underlying backbone with backbone name {config.backbone_name}")
        
        # freeze the backbone weights if required
        if 'freeze_encoders' in config.keys() and config.freeze_encoders:
            for backbone in self.backbones:
                backbone.freeze()
            log.info("The model's encoders weights are frozen. Set 'freeze_encoders' \
                      in the config to False to unfreeze them.")

        # rename variables
        num_representation_features = config.backbone_output_size * n_datasets

        # build converter (if required) and set the latent space size according to it
        self.converter = nn.Sequential() # TODO : remove once sure it can be removed

        # set up the projection head layers shapes
        layers_shapes = config.proj_layers_shapes
        output_shape = layers_shapes[-1]

        # set projection head activation
        activation = config.projection_head_name
        log.debug(f"activation = {activation}")

        self.projection_head = ProjectionHead(
            num_representation_features=num_representation_features,
            layers_shapes=layers_shapes,
            activation=activation,
            drop_rate=config.ph_drop_rate)

        # set up class keywords
        self.config = config
        self.n_datasets = n_datasets
        self.sample_data = sample_data
        self.sample_i = np.array([])
        self.sample_j = np.array([])
        self.sample_filenames = []
        self.num_representation_features = num_representation_features
        self.output_shape = output_shape
        self.lr = self.config.lr

        # Keeps track of losses
        self.training_step_outputs = []
        self.validation_step_outputs = []
        if self.config.ssl_loss=='BarlowTwins':
            self.training_step_loss_inv = []
            self.training_step_loss_redund = []
            self.validation_step_loss_inv = []
            self.validation_step_loss_redund = []

        # Output of intermediate layer of ProjectionHead
        self.activation={}

    def forward(self, x, idx_region=None):
        # log.info(f"x shape: {x.shape}")
        embeddings = []
        for i in range(self.n_datasets):
            embedding = self.backbones[i].forward(x[i])
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=1)
        embeddings = self.converter.forward(embeddings)
        if idx_region is not None:
            out = self.projection_head[idx_region].forward(embeddings)
        else:
            out = self.projection_head.forward(embeddings)
        return out


    def get_full_inputs_from_batch(self, batch):
        full_inputs = []
        for (inputs, filenames) in batch:  # loop over datasets
            full_inputs.append(inputs)
        
        inputs = full_inputs
        return (inputs, filenames)


    def load_pretrained_model(self, pretrained_model_path,
                              encoder_only=False, freeze_loaded_layers=False):
        """Load weights stored in a state_dict at pretrained_model_path
        """

        pretrained_state_dict = torch.load(pretrained_model_path)['state_dict']
        if encoder_only:
            pretrained_state_dict = OrderedDict(
                {k: v for k, v in pretrained_state_dict.items()
                 if 'encoder' in k})

        model_dict = self.state_dict()

        loaded_layers = []
        for n, p in pretrained_state_dict.items():
            if n in model_dict:
                loaded_layers.append(n)
                model_dict[n] = p

        self.load_state_dict(model_dict)

        not_loaded_layers = [
            key for key in model_dict.keys() if key not in loaded_layers]
        # print(f"Loaded layers = {loaded_layers}")
        log.info(f"Layers not loaded = {not_loaded_layers}")

        # freeze loaded layers
        if freeze_loaded_layers:
            for name, para in self.named_parameters():
                if name in loaded_layers:
                    para.requires_grad = False


    def configure_optimizers(self):
        """Adam optimizer"""
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay)
        return_dict = {"optimizer": optimizer}

        return return_dict
    
    
    def barlow_twins_loss(self, z_i, z_j):
        "Loss function for SSL (BarlowTwins)"
        loss = BarlowTwinsLoss(lambda_param=self.config.lambda_BT,
                               correlation=self.config.BT_correlation,
                               device=self.config.device)
        return loss.forward(z_i, z_j)
    

    def nt_xen_loss(self, z_i, z_j):
        """Loss function for contrastive (SimCLR)"""
        loss = NTXenLoss(temperature=self.config.temperature,
                         return_logits=True)
        return loss.forward(z_i, z_j)


    def training_step(self, train_batch, batch_idx):
        """Training step.
        """
        inputs, filenames = self.get_full_inputs_from_batch(train_batch)

        # print("TRAINING STEP", inputs.shape)
        input_i = [inputs[i][:, 0, ...] for i in range(self.n_datasets)]
        input_j = [inputs[i][:, 1, ...] for i in range(self.n_datasets)]
        z_i = self.forward(input_i)
        z_j = self.forward(input_j)

        # compute the right loss
        if self.config.ssl_loss=='SimCLR':
            batch_loss, sim_zij, sim_zii, sim_zjj = self.nt_xen_loss(z_i, z_j)
        elif self.config.ssl_loss=='BarlowTwins':
            batch_loss, loss_invariance, loss_redundancy = self.barlow_twins_loss(z_i,z_j)

        if batch_idx == 0:
            self.sample_i = change_list_device(input_i, 'cpu')
            self.sample_j = change_list_device(input_j, 'cpu')
            self.sample_filenames = filenames
            if self.config.ssl_loss=='SimCLR':
                self.sim_zij = sim_zij * self.config.temperature
                self.sim_zii = sim_zii * self.config.temperature
                self.sim_zjj = sim_zjj * self.config.temperature
        
        # logs - a dictionary
        #self.log('Loss/Train', float(batch_loss), on_epoch=True)
        logs = {"train_loss": float(batch_loss)}
        if self.config.ssl_loss=='BarlowTwins':
            logs["train_loss_inv"] = float(loss_invariance)
            logs["train_loss_redund"] = float(loss_redundancy)

        self.training_step_outputs.append(batch_loss)
        if self.config.ssl_loss=='BarlowTwins':
            # decompose loss in invariance and redundancy term
            self.training_step_loss_inv.append(loss_invariance)
            self.training_step_loss_redund.append(loss_redundancy)

        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "loss": batch_loss,
            # optional for batch logging purposes
            "log": logs}

        return batch_dictionary
    
        
    def is_epoch_to_save(self):
        """Tells if it is the right epoch to save model weights."""
        return (self.config.nb_epochs_per_weight_save > 0) and (self.current_epoch % self.config.nb_epochs_per_weight_save == 0 \
                or self.current_epoch >= self.config.max_epochs)
        

    def on_train_epoch_end(self):
        """Computation done at the end of the epoch"""
                
        if self.is_epoch_to_save():
            print('saving model weights')
            dir_to_save = './logs/model_weights_evolution/'
            if not os.path.isdir(dir_to_save):
                os.mkdir(dir_to_save)
            torch.save({'state_dict': self.state_dict()},
                        dir_to_save + f'model_weights_epoch{self.current_epoch}.pt')
            
        # calculates average loss
        avg_loss = torch.stack([x for x in self.training_step_outputs]).mean()

        # logging using tensorboard logger
        self.loggers[0].experiment.add_scalar(
            "Loss/Train",
            avg_loss,
            self.current_epoch)
        
        if self.config.ssl_loss=='BarlowTwins':
            # visu the two loss components on tensorboard
            avg_loss_inv = torch.stack([x for x in self.training_step_loss_inv]).mean()
            avg_loss_redund = torch.stack([x for x in self.training_step_loss_redund]).mean()
            self.loggers[0].experiment.add_scalar(
                "LossInv/Train",
                avg_loss_inv,
                self.current_epoch)
            self.loggers[0].experiment.add_scalar(
                "LossRedund/Train",
                avg_loss_redund,
                self.current_epoch)

        if self.config.ssl_loss=='BarlowTwins':
            avg_loss_inv = avg_loss_inv.detach().cpu().item()
            avg_loss_redund = avg_loss_redund.detach().cpu().item()

        self.training_step_outputs.clear()  # free memory
        if self.config.ssl_loss=='BarlowTwins':
            self.training_step_loss_inv.clear()
            self.training_step_loss_redund.clear()


    def validation_step(self, val_batch, batch_idx):
        """Validation step"""
        inputs, _ = self.get_full_inputs_from_batch(val_batch)
        
        input_i = [inputs[i][:, 0, ...] for i in range(self.n_datasets)]
        input_j = [inputs[i][:, 1, ...] for i in range(self.n_datasets)]
        z_i = self.forward(input_i)
        z_j = self.forward(input_j)

        if self.config.ssl_loss=='SimCLR':
            batch_loss, sim_zij, sim_zii, sim_zjj = self.nt_xen_loss(z_i, z_j)
        elif self.config.ssl_loss=='BarlowTwins':
            batch_loss, loss_invariance, loss_redundancy = self.barlow_twins_loss(z_i,z_j)
        
        # values useful for early stoppings
        self.log('val_loss', float(batch_loss), on_epoch=True)
        # logs- a dictionary
        logs = {"val_loss": float(batch_loss)}
        if self.config.ssl_loss=='BarlowTwins':
            logs["val_loss_inv"] = float(loss_invariance)
            logs["val_loss_redund"] = float(loss_redundancy)
        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "val_loss": batch_loss,
            # optional for batch logging purposes
            "log": logs}
        self.validation_step_outputs.append(batch_loss)
        if self.config.ssl_loss=='BarlowTwins':
            # decompose loss in invariance and redundancy term
            self.validation_step_loss_inv.append(loss_invariance)
            self.validation_step_loss_redund.append(loss_redundancy)

        return batch_dictionary


    def on_validation_epoch_end(self):
        """Computation done at the end of each validation epoch"""        

        # calculates average loss
        avg_loss = torch.stack([x for x in self.validation_step_outputs]).mean()

        # logs losses using tensorboard logger
        self.loggers[0].experiment.add_scalar(
            "Loss/Val",
            avg_loss,
            self.current_epoch)
        
        if self.config.ssl_loss=='BarlowTwins':
            # visu the two loss components on tensorboard
            avg_loss_inv = torch.stack([x for x in self.validation_step_loss_inv]).mean()
            avg_loss_redund = torch.stack([x for x in self.validation_step_loss_redund]).mean()
            self.loggers[0].experiment.add_scalar(
                "LossInv/Val",
                avg_loss_inv,
                self.current_epoch)
            self.loggers[0].experiment.add_scalar(
                "LossRedund/Val",
                avg_loss_redund,
                self.current_epoch)


        # save model if best validation loss
        save_path = './logs/'
        if self.current_epoch == 0:
            best_loss = np.inf
        elif self.current_epoch > 0:
            # load the current best loss
            with open(save_path+"best_model_params.json", 'r') as file:
                best_model_params = json.load(file)
                best_loss = best_model_params['best_loss']

        # compare to the current loss and replace the best if necessary
        avg_loss = avg_loss.cpu().item()
        if avg_loss < best_loss:
            torch.save({'state_dict': self.state_dict()},
                    save_path+'best_model_weights.pt')
            best_model_params = {
                'epoch': self.current_epoch, 'best_loss': avg_loss}
            with open(save_path+"best_model_params.json", 'w') as file:
                json.dump(best_model_params, file)

        if self.config.ssl_loss=='BarlowTwins':
            avg_loss_inv = avg_loss_inv.detach().cpu().item()
            avg_loss_redund = avg_loss_redund.detach().cpu().item()

        self.validation_step_outputs.clear()  # free memory
        if self.config.ssl_loss=='BarlowTwins':
            self.validation_step_loss_inv.clear()
            self.validation_step_loss_redund.clear()
