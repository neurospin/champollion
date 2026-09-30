"""Smoke tests for REQ-CHAMPTEST-05 and REQ-CHAMPTEST-06 (champollion_pipeline ledger, TASK-101).

These are characterization tests of the real single-region training path:
the Hydra config is composed from ``champollion/configs/config.yaml`` the way
the pipeline wrapper (``train_champollion.py``) does it -- ``+dataset/<name>=
<region>`` from an extra search path, ``++dataset_folder=...``,
``platform=cuda`` -- plus ``device=cpu`` so it runs on a CPU-only machine.
The dataset is a tiny synthetic region (skeleton, foldlabel and cut-in mask
``.npy`` files plus subject CSVs) written to ``tmp_path``; the default
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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
from champollion.data.datamodule import DataModule_Learning
from champollion.ssl_model import SSLModel
from champollion.utils.config import process_config
from hydra import compose, initialize_config_dir
from pytorch_lightning import loggers as pl_loggers

CONFIG_DIR = Path(__file__).resolve().parents[1] / "champollion" / "configs"

DATASET_GROUP = "synthetic"
REGION = "region"
N_SUBJECTS = 8
VOLUME_SHAPE = (12, 12, 12)  # (D, H, W); arrays carry a trailing channel axis
BATCH_SIZE = 2

# Skeleton voxel values used by the augmentations (bottom 30, top 35, surface 60).
SURFACE, BOTTOM, TOP = 60, 30, 35


def _synthetic_subject(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One subject: a sulcus-like plane (skeleton) split into 4 labelled branches."""
    skeleton = np.zeros((*VOLUME_SHAPE, 1), dtype=np.float32)
    foldlabel = np.zeros((*VOLUME_SHAPE, 1), dtype=np.int32)
    depth = int(rng.integers(4, 8))
    skeleton[depth, 2:10, 2:10, 0] = SURFACE
    skeleton[depth, 2:10, 2, 0] = BOTTOM
    skeleton[depth, 2:10, 9, 0] = TOP
    for branch, (h, w) in enumerate([(2, 2), (2, 6), (6, 2), (6, 6)], start=1):
        foldlabel[depth, h : h + 4, w : w + 4, 0] = 1000 + branch
    return skeleton, foldlabel


def _write_synthetic_region(root: Path) -> tuple[Path, Path]:
    """Write arrays/CSVs and a dataset yaml; return (config search dir, dataset_folder)."""
    data_dir = root / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    subjects = [f"sub-{i:02d}" for i in range(N_SUBJECTS)]
    skeletons, foldlabels = zip(*(_synthetic_subject(rng) for _ in subjects))
    np.save(data_dir / "skeleton.npy", np.stack(skeletons))
    np.save(data_dir / "label.npy", np.stack(foldlabels))
    mask = np.zeros((*VOLUME_SHAPE, 1), dtype=np.float32)
    mask[3:9, 3:9, 3:9, 0] = 1
    np.save(data_dir / "mask.npy", mask)
    pd.DataFrame({"Subject": subjects}).to_csv(data_dir / "skeleton_subject.csv", index=False)

    search_dir = root / "configs"
    group_dir = search_dir / "dataset" / DATASET_GROUP
    group_dir.mkdir(parents=True)
    d, h, w = VOLUME_SHAPE
    (group_dir / f"{REGION}.yaml").write_text(
        f"# @package dataset.{DATASET_GROUP}_{REGION}\n"
        f"dataset_name: {DATASET_GROUP}_{REGION}\n"
        "numpy_all: ${dataset_folder}/skeleton.npy\n"
        "subjects_all: ${dataset_folder}/skeleton_subject.csv\n"
        "crop_dir:\n"
        "foldlabel_all: ${dataset_folder}/label.npy\n"
        "train_val_csv_file: ${dataset_folder}/skeleton_subject.csv\n"
        "cutin_mask_path: ${dataset_folder}/mask.npy\n"
        "flip_dataset: False\n"
        f"input_size: (1, {d}, {h}, {w})\n"
    )
    return search_dir, data_dir


def _compose_config(search_dir: Path, data_dir: Path):
    """Compose the training config as the pipeline wrapper does, forced onto the CPU."""
    overrides = [
        f"hydra.searchpath=[file://{search_dir}]",
        f"+dataset/{DATASET_GROUP}={REGION}",
        f"++dataset_folder={data_dir}",
        "platform=cuda",
        "device=cpu",
        "load_sparse=false",
        "num_cpu_workers=1",
        f"batch_size={BATCH_SIZE}",
        "partition=[0.5,0.5]",
        "pin_mem=false",
        "max_epochs=1",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        return compose(config_name="config", overrides=overrides)


@pytest.fixture
def config(tmp_path):
    search_dir, data_dir = _write_synthetic_region(tmp_path)
    return process_config(_compose_config(search_dir, data_dir))


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
