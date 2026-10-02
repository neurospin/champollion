"""Smoke tests for REQ-V1INTEG-01..04 (champollion_pipeline ledger, TASK-159).

Integration of neurospin/champollion PR #3 (``refactor-update-2``): the ResNet
backbone is brought in line with the Champollion preprint. The shipped
``configs/backbone/ResNet.yaml`` sets ``adaptive_pooling: null`` and
``initial_stride: 1``; the ResNet must then derive its linear head from the
configured input shape, and a ``block`` option selects ``BasicBlock`` or
``Bottleneck`` residual blocks.

The config is composed the way the pipeline wrapper does it (conftest.py's
``compose_training_config`` on the tiny synthetic single-region dataset), plus
``backbone=ResNet``. The default ConvNet path stays pinned by
test_smoke_train_step.py (REQ-CHAMPTEST-05/06).

Offline, CPU only. Run from the champollion_pipeline root:
    pixi run test-champollion
"""

import importlib
import math

import pytest
import pytorch_lightning as pl
import torch
from champollion.data.datamodule import DataModule_Learning
from champollion.ssl_model import SSLModel
from pytorch_lightning import loggers as pl_loggers

BATCH = 2


@pytest.fixture
def resnet_config(synthetic_region, compose_training_config):
    return compose_training_config(*synthetic_region, "backbone=ResNet")


def _bottleneck_class():
    resnet = importlib.import_module("champollion.backbones.resnet")
    cls = getattr(resnet, "Bottleneck", None)
    if cls is None:
        pytest.fail("champollion.backbones.resnet.Bottleneck is not defined (REQ-V1INTEG-02)")
    return cls


def _residual_blocks(backbone):
    return [block for name in ("layer1", "layer2", "layer3", "layer4") for block in getattr(backbone, name).children()]


def test_resnet_backbone_maps_input_to_backbone_output_size(resnet_config):
    """REQ-V1INTEG-01: (2, *input_size) -> (2, backbone_output_size) with the shipped ResNet config."""
    model = SSLModel(resnet_config, sample_data=DataModule_Learning(resnet_config))
    backbone = model.backbones[0].eval()
    x = torch.rand(BATCH, *resnet_config.data[0].input_size)

    with torch.no_grad():
        out = backbone(x)

    assert tuple(out.shape) == (BATCH, resnet_config.backbone_output_size)


def test_bottleneck_block_option_builds_only_bottleneck_residual_blocks(synthetic_region, compose_training_config):
    """REQ-V1INTEG-02: block=bottleneck -> every residual block is a Bottleneck."""
    bottleneck = _bottleneck_class()
    bottleneck_config = compose_training_config(*synthetic_region, "backbone=ResNet", "block=bottleneck")
    model = SSLModel(bottleneck_config, sample_data=DataModule_Learning(bottleneck_config))

    blocks = _residual_blocks(model.backbones[0])

    assert blocks, "ResNet backbone has no residual blocks"
    offenders = sorted({type(b).__name__ for b in blocks if not isinstance(b, bottleneck)})
    assert not offenders, f"non-Bottleneck residual blocks: {offenders}"


def test_model_summary_runs_resnet_backbone_forward(resnet_config):
    """REQ-V1INTEG-03: print_model_summary runs the real ResNet SSLModel's backbone forward."""
    train_module = importlib.import_module("champollion.train")
    model = SSLModel(resnet_config, sample_data=DataModule_Learning(resnet_config))
    block_types = {type(b) for b in _residual_blocks(model.backbones[0])}

    stats = train_module.print_model_summary(model, resnet_config)

    # SSLModel calls ``backbone.forward(x)`` directly, which skips the backbone's own
    # module hooks; its residual blocks are called normally, so torchinfo records
    # their output shapes only if the backbone's forward really ran.
    assert stats is not None, "print_model_summary returned no torchinfo statistics"
    assert stats.total_mult_adds > 0, "summary did not run a forward pass"
    ran = [layer for layer in stats.summary_list if layer.module is not None and type(layer.module) in block_types]
    assert ran, "no ResNet residual block in the torchinfo summary"
    assert all(layer.output_size for layer in ran), "ResNet residual blocks recorded no output (forward not run)"


class _LossRecorder(pl.Callback):
    def __init__(self):
        self.train_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_losses.append(float(outputs["loss"]))


def test_one_resnet_training_step_has_finite_loss(resnet_config):
    """REQ-V1INTEG-04: one CPU training step with backbone=ResNet, finite training loss."""
    # BarlowTwins divides each embedding dim by its std over the batch with no epsilon,
    # so a degenerate draw on this 2-subject batch can give a NaN loss whatever the
    # backbone (seen once, then 0 in 80 unseeded and 80 seeded reruns); pin the RNG.
    pl.seed_everything(0)
    data_module = DataModule_Learning(resnet_config)
    model = SSLModel(resnet_config, sample_data=data_module)
    recorder = _LossRecorder()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
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
