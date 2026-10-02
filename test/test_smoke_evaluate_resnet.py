"""Smoke test for REQ-V1INTEG-05 (champollion_pipeline ledger, TASK-159).

neurospin/champollion PR #3 makes ``evaluate.load_model`` handle ResNet runs:
the ResNet backbone's weights live under ``backbones.0.<layer>`` in the
Lightning checkpoint (not ``backbones.0.encoder.`` like ConvNet), and the
ResNet is rebuilt from the run's saved config, including its input shape.

The run directory is laid out the way ``evaluate.load_model`` reads it (and
test_smoke_evaluate.py builds it for ConvNet): ``.hydra/config.yaml`` holds the
composed, unprocessed config; the checkpoint sits under
``logs/lightning_logs/version_0/checkpoints/``. Instead of a fit, the backbone's
parameters and BatchNorm buffers are randomized so freshly-initialized weights
cannot match by accident. The ConvNet evaluate path stays pinned by
test_smoke_evaluate.py (REQ-CHAMPTEST-07).

Offline, CPU only. Run from the champollion_pipeline root:
    pixi run test-champollion
"""

import importlib

import torch
from champollion.data.datamodule import DataModule_Learning
from champollion.ssl_model import SSLModel
from omegaconf import OmegaConf


def _randomize(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for tensor in [*module.parameters(), *module.buffers()]:
            if tensor.is_floating_point():
                tensor.copy_(torch.rand(tensor.shape, generator=generator) + 0.5)


def test_load_model_restores_trained_resnet_backbone(tmp_path, synthetic_region, compose_training_config):
    """REQ-V1INTEG-05: evaluate.load_model on a ResNet run reproduces the saved backbone's output."""
    evaluate = importlib.import_module("champollion.evaluate")
    raw_config = compose_training_config(*synthetic_region, "backbone=ResNet", process=False)
    config = compose_training_config(*synthetic_region, "backbone=ResNet")
    model = SSLModel(config, sample_data=DataModule_Learning(config))
    _randomize(model.backbones[0], seed=0)

    model_dir = tmp_path / "model"
    (model_dir / ".hydra").mkdir(parents=True)
    OmegaConf.save(raw_config, model_dir / ".hydra" / "config.yaml")
    ckpt_dir = model_dir / "logs" / "lightning_logs" / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    torch.save({"state_dict": model.state_dict()}, ckpt_dir / "epoch=0-step=1.ckpt")

    in_shape = tuple(config.data[0].input_size)
    loaded = evaluate.load_model(str(model_dir), in_shape)

    x = torch.rand(3, *in_shape, generator=torch.Generator().manual_seed(1))
    backbone = model.backbones[0].eval()
    with torch.no_grad():
        expected = backbone(x)
        got = loaded(x)
    assert tuple(got.shape) == (3, config.backbone_output_size)
    torch.testing.assert_close(got, expected)
