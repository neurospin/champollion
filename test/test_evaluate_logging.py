"""Log-channel test for REQ-V1PR3-BDRABCZUK-04E8272FE87B and -9308E87B8E72.

``evaluate.load_model`` sends the checkpoint ``state_dict`` key list (debug
severity) and the weights-loaded message (info severity) to the logging
framework, and writes nothing to standard output.

The encoder class and the checkpoint are stubbed: only the log channel is
under test (the embedding output is covered by test_smoke_evaluate.py).
"""

import logging

import torch
from champollion import evaluate
from omegaconf import OmegaConf

KEYS = ["backbones.0.conv.weight", "backbones.0.conv.bias", "head.weight"]


class _StubConvNet:
    def __init__(self, **kwargs):
        self.loaded = None

    def load_state_dict(self, state_dict):
        self.loaded = state_dict

    def eval(self):
        return self


def _prepare(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "logs" / "lightning_logs" / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "m.ckpt").write_bytes(b"")
    hydra_dir = tmp_path / ".hydra"
    hydra_dir.mkdir()
    cfg = {
        "backbone_name": "convnet",
        "data": [{"input_size": "(1, 4, 4, 4)"}],
        "encoder_depth": 1, "block_depth": 1, "filters": [2],
        "initial_kernel_size": 3, "initial_stride": 1, "max_pool": False,
        "backbone_output_size": 4, "adaptive_pooling": "max", "drop_rate": 0.0,
    }
    OmegaConf.save(OmegaConf.create(cfg), hydra_dir / "config.yaml")
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: {"state_dict": {k_: 0 for k_ in KEYS}}
    )
    monkeypatch.setattr(evaluate, "ConvNet", _StubConvNet)


def _load(tmp_path):
    return evaluate.load_model(str(tmp_path), (1, 4, 4, 4))


def test_load_model_writes_nothing_to_stdout(tmp_path, monkeypatch, capsys):
    _prepare(tmp_path, monkeypatch)
    _load(tmp_path)
    assert capsys.readouterr().out == ""


def test_key_list_is_a_debug_record(tmp_path, monkeypatch, caplog):
    _prepare(tmp_path, monkeypatch)
    with caplog.at_level(logging.DEBUG):
        _load(tmp_path)
    records = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG and all(k in r.getMessage() for k in KEYS)
    ]
    assert len(records) == 1


def test_weights_loaded_is_an_info_record(tmp_path, monkeypatch, caplog):
    _prepare(tmp_path, monkeypatch)
    with caplog.at_level(logging.DEBUG):
        _load(tmp_path)
    records = [
        r for r in caplog.records
        if r.levelno == logging.INFO and "Model weights loaded" in r.getMessage()
    ]
    assert len(records) == 1
