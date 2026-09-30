"""Tests for REQ-TRAIN-SUMMARY-01 (champollion_pipeline ledger, TASK-098).

``champollion/train.py`` imports ``summary`` from ``torchinfo`` but called it
with the old torch-summary API (``input_data=<shape tuple>, batch_dim=0``).
torchinfo treats ``input_data`` as the actual forward() inputs, so the shape
ints were unpacked as positional args ("forward() takes from 2 to 3
positional arguments but 6 were given") and every single-region training
crashed before epoch 1.

Verification seam (agreed in REQ-TRAIN-SUMMARY-01): the summary step is
extracted from ``train()`` into a module-level
``champollion.train.print_model_summary(model, config)`` that ``train()``
calls with its ``model`` and ``config``. The behavioural tests run that helper
against a tiny CPU module whose ``forward(self, x, idx_region=None)`` mirrors
``SSLModel.forward``: it indexes ``x[0]`` and feeds it to a ``Conv3d`` +
``BatchNorm3d`` backbone, as ``SSLModel`` does with ``self.backbones[i](x[i])``.
Training passes ``x`` as a list of per-region 5D tensors, so the summary must
too: a bare ``(1, *input_size)`` tensor makes ``x[0]`` 4D and BatchNorm3d
raises "expected 5D input (got 4D input)" -- the same failure measured on the
real SSLModel (see ARCHITECTURE.md, COMP-CHAMPOLLION-V1-TRAIN). The config is
shaped like the real one (``config.data[0].input_size``, ``config.dataset``,
``config.device``).

Offline, CPU only. Run from the champollion_pipeline root:
    pixi run test-specific external/champollion_V1/test/test_train_model_summary.py
"""

import ast
import importlib
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

TRAIN_PY = Path(__file__).resolve().parents[1] / "champollion" / "train.py"

INPUT_SIZE = [1, 4, 5, 6]  # (C, D, H, W) as in config.data[0].input_size
EXPECTED_SHAPE = (1, 1, 4, 5, 6)  # batch dim 1 prepended


class _TinySSLLike(nn.Module):
    """Minimal model with SSLModel.forward's input contract; records its inputs.

    Like SSLModel, ``x`` is indexed by region and ``x[0]`` goes through a
    Conv3d + BatchNorm3d backbone, so a bare tensor input fails the same way
    the real model does (BatchNorm3d: expected 5D input, got 4D).
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(2)
        self.calls = []

    def forward(self, x, idx_region=None):
        self.calls.append((x, idx_region))
        return self.bn(self.conv(x[0])).mean(dim=(2, 3, 4))


def _single_region_config():
    return OmegaConf.create(
        {
            "data": [{"input_size": list(INPUT_SIZE)}],
            "dataset": {"only_region": {}},
            "device": "cpu",
        }
    )


@pytest.fixture
def train_module(tmp_path, monkeypatch):
    """Import champollion.train with cwd in tmp_path.

    Importing the module instantiates a TensorBoard SummaryWriter/logger that
    create directories in the cwd; keep them out of the repo.
    """
    monkeypatch.chdir(tmp_path)
    return importlib.import_module("champollion.train")


def _helper(train_module):
    helper = getattr(train_module, "print_model_summary", None)
    if helper is None:
        pytest.fail(
            "champollion.train.print_model_summary(model, config) is not defined "
            "(summary step not extracted from train(); see REQ-TRAIN-SUMMARY-01)"
        )
    return helper


def test_single_region_summary_does_not_raise(train_module):
    """Single-region summary completes without torchinfo raising."""
    model = _TinySSLLike()
    _helper(train_module)(model, _single_region_config())


def test_single_region_summary_forwards_list_of_one_batched_input_tensor(train_module):
    """forward() receives one positional list holding one (1, *input_size) tensor."""
    model = _TinySSLLike()
    _helper(train_module)(model, _single_region_config())

    assert model.calls, "summary never ran the model's forward()"
    for x, idx_region in model.calls:
        assert isinstance(x, list), f"forward() got {type(x).__name__}, expected a list of per-region tensors"
        assert len(x) == 1
        assert isinstance(x[0], torch.Tensor)
        assert tuple(x[0].shape) == EXPECTED_SHAPE
        assert idx_region is None


def test_train_calls_print_model_summary_with_model_and_config():
    """Wiring guard: train() delegates its summary step to the helper."""
    tree = ast.parse(TRAIN_PY.read_text())
    train_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    calls = [
        node
        for node in ast.walk(train_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print_model_summary"
    ]
    assert len(calls) == 1, "train() must call print_model_summary exactly once"
    args = [a.id for a in calls[0].args if isinstance(a, ast.Name)]
    assert args == ["model", "config"]
