"""Smoke tests for REQ-CHAMPTEST-05 and REQ-CHAMPTEST-06 (champollion_pipeline ledger, TASK-101).

These are characterization tests of the real single-region training path:
the Hydra config is composed from ``champollion/configs/config.yaml`` the way
the pipeline wrapper (``train_champollion.py``) does it -- ``+dataset/<name>=
<region>`` from an extra search path, ``++dataset_folder=...``,
``platform=cuda`` -- plus ``device=cpu`` so it runs on a CPU-only machine.
The dataset is a tiny synthetic region (skeleton, foldlabel and cut-in mask
``.npy`` files plus subject CSVs) written to ``tmp_path`` by the shared
``synthetic_region`` / ``compose_training_config`` fixtures (conftest.py); the default
augmentations (``mixed``: rotation, cutout/cutin, translation), the default
``ConvNet`` backbone, the ``relu`` projection head and the BarlowTwins loss
all run for real.

They pin two failures that reached main untested:
- TASK-098: ``print_model_summary`` crashed on the real ``SSLModel``
  (torchinfo called with the old torch-summary API).
- TASK-096: the pipeline composes ``platform=cuda``; a removed/renamed
  platform option makes composition fail before training starts.

The conftest chdirs each test into ``tmp_path``, so ``logs/`` and ``runs/``
written by training stay out of the checkout. Offline, CPU only. Run from the
champollion_pipeline root:
    pixi run test-champollion
"""

import importlib
import math

import pytest
import pytorch_lightning as pl
from champollion.data.datamodule import DataModule_Learning
from champollion.ssl_model import SSLModel
from pytorch_lightning import loggers as pl_loggers


@pytest.fixture
def config(synthetic_region, compose_training_config):
    return compose_training_config(*synthetic_region)


@pytest.fixture
def train_module():
    # Imported lazily: champollion.train has cwd side effects at import time
    # (the conftest keeps them in tmp_path).
    return importlib.import_module("champollion.train")


class _LossRecorder(pl.Callback):
    def __init__(self):
        self.train_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_losses.append(float(outputs["loss"]))


def test_one_training_and_validation_step_run_on_cpu(config):
    """REQ-CHAMPTEST-05: one train batch + one val batch of the real SSLModel, finite loss."""
    data_module = DataModule_Learning(config)
    model = SSLModel(config, sample_data=data_module)
    recorder = _LossRecorder()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=pl_loggers.TensorBoardLogger("logs"),
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[recorder],
    )

    trainer.fit(model, data_module)

    assert trainer.global_step == 1
    assert len(recorder.train_losses) == 1
    assert math.isfinite(recorder.train_losses[0])
    # Only that validation ran: after a single step, eval-mode BatchNorm (running
    # stats barely updated) shrinks the gap between two samples below float32
    # resolution, so BarlowTwins' per-batch std is 0 and val_loss is NaN here.
    assert "val_loss" in trainer.callback_metrics, "validation step did not run"


def test_model_summary_runs_forward_pass_on_real_model(config, train_module):
    """REQ-CHAMPTEST-06: print_model_summary completes on the real single-region SSLModel."""
    model = SSLModel(config, sample_data=DataModule_Learning(config))

    stats = train_module.print_model_summary(model, config)

    assert stats.total_params == sum(p.numel() for p in model.parameters())
    assert stats.total_mult_adds > 0, "summary did not run a forward pass"
