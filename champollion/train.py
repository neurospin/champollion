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

import os
import hydra
import numpy.random as rd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profilers import PyTorchProfiler, SimpleProfiler
import omegaconf
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from champollion.data.datamodule import DataModule_Learning
from champollion.models.ssl_model import SSLModel

from champollion.utils.config import process_config
from champollion.utils.logs import set_root_logger_level, \
    set_file_log_handler, set_file_logger

tb_logger = pl_loggers.TensorBoardLogger('logs')
writer = SummaryWriter()
log = set_file_logger(__file__)


def get_train_seed():
    """Get a random seed for training."""
    train_seed = rd.randint(256)
    return train_seed


@hydra.main(config_name='config', version_base="1.1", config_path="configs")
def train(config):
    config = process_config(config)

    # set the number of working cpus
    available_cpus = len(os.sched_getaffinity(0))
    log.debug('Available working cpus:', available_cpus)
    n_cpus = min(available_cpus, config.num_cpu_workers)
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cpus)
    log.debug('NUMEXPR_MAX_THREADS', n_cpus)

    set_root_logger_level(config.verbose)
    # Sets handler for logger
    set_file_log_handler(file_dir=os.getcwd(),
                         suffix='output')
    log.debug(f"current directory = {os.getcwd()}")

    data_module = DataModule_Learning(config)
    
    model = SSLModel(config,
                                     sample_data=data_module)

    # load pretrained model's weights if in config
    if config.pretrained_model_path is not None:
        log.info(f"Load weigths stored at {config.pretrained_model_path}")
        model.load_pretrained_model(config.pretrained_model_path,
                                    encoder_only=config.load_encoder_only,
                                    freeze_loaded_layers=config.freeze_loaded_layers)

    input_size = tuple([1] + list(config.data[0].input_size))
    if (len(config.dataset.keys()) == 1): # if one region
        print(config.data[0].input_size)
        summary(model, input_data=input_size, batch_dim=0, device=config.device, depth=6)
    else:
        summary(model, device=config.device, depth=6) # TODO : why 16 ?

    # choose the logger
    loggers = [tb_logger]

    # Configure device (CPU or GPU)
    if config.device == 'cuda':
        accelerator = 'gpu'
        devices = 1
    else:
        accelerator = 'cpu'
        devices = 'auto'

    # Configure profiler if enabled
    profiler = None
    if config.get('enable_profiling', False):
        profiler_type = config.get('profiler_type', 'pytorch')
        output_dir = config.get('profiler_output_dir', './profiler_logs')

        if profiler_type == 'simple':
            profiler = SimpleProfiler(dirpath=output_dir, filename="simple_profiler")
        else:
            profiler = PyTorchProfiler(
                dirpath=output_dir,
                filename="pytorch_profiler",
                export_to_chrome=config.get('export_chrome_trace', True),
                row_limit=20
            )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=config.max_epochs,
        logger=loggers,
        log_every_n_steps=config.log_every_n_steps,
        accumulate_grad_batches=config.accumulate_grad_batches,
        profiler=profiler
        )

    # start training
    trainer.fit(model, data_module, ckpt_path=config.checkpoint_path)
    log.info("Fitting is done")

    print(f"End of training for model {os.path.abspath('./')}")


if __name__ == "__main__":
    omegaconf.OmegaConf.register_new_resolver("get_train_seed", get_train_seed)
    train()
