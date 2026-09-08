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

from contrastive.backbones.convnet import ConvNet
from contrastive.backbones.resnet import ResNet, BasicBlock
from contrastive.backbones.projection_heads import *
from contrastive.data.utils import change_list_device
from contrastive.models.models_utils import *
from contrastive.losses import *

from contrastive.utils.logs import set_file_logger
log = set_file_logger(__file__)



class ContrastiveLearnerFusion(pl.LightningModule):

    def __init__(self, config, sample_data):
        super(ContrastiveLearnerFusion, self).__init__()

        if config.multiregion_single_encoder:
            n_datasets = 1
            n_regions = len(config.data)
            log.info("n_datasets 1 because a single encoder is used for multiple regions")
        else:
            n_datasets = len(config.data)
            log.info(f"n_datasets {n_datasets}")

        # define the encoder structure
        self.backbones = nn.ModuleList()
        if config.backbone_name == "convnet":
            for i in range(n_datasets):
                self.backbones.append(ConvNet(
                    encoder_depth=config.encoder_depth,
                    filters=config.filters,
                    block_depth=config.block_depth,
                    initial_kernel_size=config.initial_kernel_size,
                    initial_stride=config.initial_stride,
                    max_pool=config.max_pool,
                    num_representation_features=config.backbone_output_size,
                    linear = config.linear_in_backbone,
                    adaptive_pooling=config.adaptive_pooling,
                    drop_rate=config.drop_rate,
                    in_shape=config.data[i].input_size))
        elif config.backbone_name == 'resnet':
            for i in range(n_datasets):
                self.backbones.append(ResNet(
                    block=BasicBlock,
                    layers=config.layers,
                    channels=config.channels,
                    in_channels=1,
                    num_classes=config.backbone_output_size,
                    zero_init_residual=config.zero_init_residual,
                    dropout_rate=config.drop_rate,
                    out_block=None,
                    prediction_bias=False,
                    initial_kernel_size=config.initial_kernel_size,
                    initial_stride=config.initial_stride,
                    adaptive_pooling=config.adaptive_pooling,
                    linear_in_backbone=config.linear_in_backbone))
        else:
            raise ValueError(f"No underlying backbone with backbone name {config.backbone_name}")
        
        # freeze the backbone weights if required
        if 'freeze_encoders' in config.keys() and config.freeze_encoders:
            for backbone in self.backbones:
                backbone.freeze()
            log.info("The model's encoders weights are frozen. Set 'freeze_encoders' \
                      in the config to False to unfreeze them.")

        # rename variables
        concat_latent_spaces_size = config.backbone_output_size * n_datasets

        # build converter (if required) and set the latent space size according to it
        converter, num_representation_features = build_converter(config, concat_latent_spaces_size)
        self.converter = converter

        # set up the projection head layers shapes
        layers_shapes = get_projection_head_shape(config, num_representation_features)
        output_shape = layers_shapes[-1]

        # set projection head activation
        activation = config.projection_head_name
        log.debug(f"activation = {activation}")

        if config.multiple_projection_heads:
            # Evaluation: need to initialize the right number of projection heads for weight mapping
            n_regions = len(config.data)
            self.projection_head = nn.ModuleList()
            for reg in range(n_regions):
                if config.linear_in_backbone:
                    self.projection_head.append(ProjectionHead(
                    num_representation_features=num_representation_features,
                    layers_shapes=layers_shapes,
                    activation=activation,
                    drop_rate=config.ph_drop_rate))
                else:
                    # add a variable size linear layer to each projection head
                    layers_shapes_including_variable = layers_shapes.copy()
                    # TODO: make it compatible with ResNet !
                    backbone_output_shape = [config.data[reg].input_size[1] // 2**config.encoder_depth,
                                            config.data[reg].input_size[2] // 2**config.encoder_depth,
                                            config.data[reg].input_size[3] // 2**config.encoder_depth]
                    backbone_output_shape = config.filters[-1]*np.prod(backbone_output_shape)
                    layers_shapes_including_variable = [backbone_output_shape] + layers_shapes_including_variable
                    self.projection_head.append(ProjectionHead(
                        num_representation_features=num_representation_features,
                        layers_shapes=layers_shapes_including_variable,
                        activation=activation,
                        drop_rate=config.ph_drop_rate))
        else:
            self.projection_head = ProjectionHead(
                num_representation_features=num_representation_features,
                layers_shapes=layers_shapes,
                activation=activation,
                drop_rate=config.ph_drop_rate)

        # set up class keywords
        self.config = config
        self.n_datasets = n_datasets
        if self.config.multiregion_single_encoder:
            self.n_regions = n_regions
        self.sample_data = sample_data
        self.sample_i = np.array([])
        self.sample_j = np.array([])
        self.sample_k = np.array([])
        self.sample_filenames = []
        self.num_representation_features = num_representation_features
        self.output_shape = output_shape
        self.lr = self.config.lr

        # Keeps track of losses
        self.training_step_outputs = []
        self.validation_step_outputs = []
        if self.config.multiple_projection_heads or self.config.multiregion_single_encoder:
            self.training_step_idxs_region = [] 
            self.validation_step_idxs_region = []
        if self.config.contrastive_model=='BarlowTwins':
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
    
    def get_full_inputs_from_batch_with_region_idx(self, batch):
        full_inputs = []
        for (inputs, filenames, idx_region) in batch:  # loop over datasets
            full_inputs.append(inputs)
        
        inputs = full_inputs
        idx_region = idx_region.detach().cpu().numpy()[0]
        return (inputs, filenames, idx_region)


    def get_full_inputs_from_batch(self, batch):
        full_inputs = []
        for (inputs, filenames) in batch:  # loop over datasets
            full_inputs.append(inputs)
        
        inputs = full_inputs
        return (inputs, filenames)


    def load_pretrained_model(self, pretrained_model_path, encoder_only=False,
                              convolutions_only=False, freeze_loaded_layers=False,
                              freeze_bias=False):
        """Load weights stored in a state_dict at pretrained_model_path
        """

        pretrained_state_dict = torch.load(pretrained_model_path)['state_dict']
        if convolutions_only:
            pretrained_state_dict = OrderedDict(
                {k: v for k, v in pretrained_state_dict.items()
                 if 'encoder' in k and
                 ('conv' in k or 'norm' in k)})
        elif encoder_only:
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
                if 'bias' in name and not freeze_bias:
                    para.requires_grad = True


    def configure_optimizers(self):
        """Adam optimizer"""
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay)
        return_dict = {"optimizer": optimizer}

        if 'scheduler' in self.config.keys() and self.config.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma)
            return_dict["lr_scheduler"] = {"scheduler": scheduler,
                                           "interval": "epoch"}


        """
        if 'scheduler' in self.config.keys() and self.config.scheduler:  
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',           # We want to minimize the loss
                factor=self.config.factor, # Divide the learning rate by 3
                patience=self.config.step_size, # Wait for 10 epochs without improvement
                threshold=self.config.threshold_plateau, # Minimum loss reduction of 10% to be considered an improvement
                threshold_mode='rel', # Relative threshold, i.e., 10% relative decrease in loss
            )
            return_dict["lr_scheduler"] = {"scheduler": scheduler,
                                        "monitor": 'val_loss',
                                        "interval": "epoch",
                                        "frequency": 1}
        """
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
        if self.config.multiple_projection_heads or self.config.multiregion_single_encoder:
            inputs, filenames, idx_region = self.get_full_inputs_from_batch_with_region_idx(train_batch)
        else:
            inputs, filenames = self.get_full_inputs_from_batch(train_batch)

        # print("TRAINING STEP", inputs.shape)
        input_i = [inputs[i][:, 0, ...] for i in range(self.n_datasets)]
        input_j = [inputs[i][:, 1, ...] for i in range(self.n_datasets)]
        if self.config.multiple_projection_heads:
            z_i = self.forward(input_i, idx_region=idx_region)
            z_j = self.forward(input_j, idx_region=idx_region)
        else:
            z_i = self.forward(input_i)
            z_j = self.forward(input_j)

        # compute the right loss
        if self.config.contrastive_model=='SimCLR':
            batch_loss, sim_zij, sim_zii, sim_zjj = self.nt_xen_loss(z_i, z_j)
        elif self.config.contrastive_model=='BarlowTwins':
            batch_loss, loss_invariance, loss_redundancy = self.barlow_twins_loss(z_i,z_j)

        if batch_idx == 0:
            self.sample_i = change_list_device(input_i, 'cpu')
            self.sample_j = change_list_device(input_j, 'cpu')
            self.sample_filenames = filenames
            if self.config.contrastive_model=='SimCLR':
                self.sim_zij = sim_zij * self.config.temperature
                self.sim_zii = sim_zii * self.config.temperature
                self.sim_zjj = sim_zjj * self.config.temperature
        
        # logs - a dictionary
        #self.log('Loss/Train', float(batch_loss), on_epoch=True)
        logs = {"train_loss": float(batch_loss)}
        if self.config.contrastive_model=='BarlowTwins':
            logs["train_loss_inv"] = float(loss_invariance)
            logs["train_loss_redund"] = float(loss_redundancy)

        self.training_step_outputs.append(batch_loss)
        if self.config.contrastive_model=='BarlowTwins':
            # decompose loss in invariance and redundancy term
            self.training_step_loss_inv.append(loss_invariance)
            self.training_step_loss_redund.append(loss_redundancy)
        if self.config.multiple_projection_heads or self.config.multiregion_single_encoder:
            self.training_step_idxs_region.append(idx_region)

        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "loss": batch_loss,
            # optional for batch logging purposes
            "log": logs}

        if self.config.scheduler:
            batch_dictionary['learning_rate'] = self.optimizers().param_groups[0]['lr']

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
        
        if self.config.contrastive_model=='BarlowTwins':
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

        # if multiregion, train loss for each region
        if self.config.multiregion_single_encoder:
            for region in range(self.n_regions):
                regional_loss = [x for x, idx in zip(self.training_step_outputs, self.training_step_idxs_region)
                                if idx==region]
                if len(regional_loss) > 0:
                    regional_loss = torch.stack(regional_loss).mean()
                    self.loggers[0].experiment.add_scalar(
                    f"LossRegion{region}/Train",
                    regional_loss,
                    self.current_epoch)

        if self.config.scheduler:
            self.loggers[0].experiment.add_scalar(
                "Learning rate",
                self.optimizers().param_groups[0]['lr'],
                self.current_epoch)

        if self.config.mode == "encoder" and self.config.contrastive_model=='BarlowTwins':
            avg_loss_inv = avg_loss_inv.detach().cpu().item()
            avg_loss_redund = avg_loss_redund.detach().cpu().item()

        self.training_step_outputs.clear()  # free memory
        if self.config.multiple_projection_heads or self.config.multiregion_single_encoder:
            self.training_step_idxs_region.clear()
        if self.config.mode == "encoder" and self.config.contrastive_model=='BarlowTwins':
            self.training_step_loss_inv.clear()
            self.training_step_loss_redund.clear()


    def validation_step(self, val_batch, batch_idx):
        """Validation step"""
        if self.config.multiple_projection_heads or self.config.multiregion_single_encoder:
            (inputs, _, idx_region) = self.get_full_inputs_from_batch_with_region_idx(val_batch)
        else:
            inputs, _ = self.get_full_inputs_from_batch(val_batch)
        
        input_i = [inputs[i][:, 0, ...] for i in range(self.n_datasets)]
        input_j = [inputs[i][:, 1, ...] for i in range(self.n_datasets)]
        if self.config.multiple_projection_heads:
            z_i = self.forward(input_i, idx_region=idx_region)
            z_j = self.forward(input_j, idx_region=idx_region)
        else:
            z_i = self.forward(input_i)
            z_j = self.forward(input_j)

        if self.config.contrastive_model=='SimCLR':
            batch_loss, sim_zij, sim_zii, sim_zjj = self.nt_xen_loss(z_i, z_j)
        elif self.config.contrastive_model=='BarlowTwins':
            batch_loss, loss_invariance, loss_redundancy = self.barlow_twins_loss(z_i,z_j)
        
        # values useful for early stoppings
        self.log('val_loss', float(batch_loss), on_epoch=True)
        # logs- a dictionary
        logs = {"val_loss": float(batch_loss)}
        if self.config.contrastive_model=='BarlowTwins':
            logs["val_loss_inv"] = float(loss_invariance)
            logs["val_loss_redund"] = float(loss_redundancy)
        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "val_loss": batch_loss,
            # optional for batch logging purposes
            "log": logs}
        self.validation_step_outputs.append(batch_loss)
        if self.config.contrastive_model=='BarlowTwins':
            # decompose loss in invariance and redundancy term
            self.validation_step_loss_inv.append(loss_invariance)
            self.validation_step_loss_redund.append(loss_redundancy)
        if self.config.multiple_projection_heads or self.config.multiregion_single_encoder:
            self.validation_step_idxs_region.append(idx_region)

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
        
        if self.config.contrastive_model=='BarlowTwins':
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
        
        # if multiregion, val loss for each region
        if self.config.multiregion_single_encoder:
            for region in range(self.n_regions):
                regional_loss = [x for x, idx in zip(self.validation_step_outputs, self.validation_step_idxs_region)
                                if idx==region]
                if len(regional_loss) > 0:
                    regional_loss = torch.stack(regional_loss).mean()
                    self.loggers[0].experiment.add_scalar(
                    f"LossRegion{region}/Val",
                    regional_loss,
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

        if self.config.contrastive_model=='BarlowTwins':
            avg_loss_inv = avg_loss_inv.detach().cpu().item()
            avg_loss_redund = avg_loss_redund.detach().cpu().item()

        self.validation_step_outputs.clear()  # free memory
        if self.config.multiple_projection_heads or self.config.multiregion_single_encoder:
            self.validation_step_idxs_region.clear()
        if self.config.mode == "encoder" and self.config.contrastive_model=='BarlowTwins':
            self.validation_step_loss_inv.clear()
            self.validation_step_loss_redund.clear()
