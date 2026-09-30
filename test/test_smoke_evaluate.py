"""Smoke test for REQ-CHAMPTEST-07 (champollion_pipeline ledger, TASK-102).

Characterization test of the embedding-generation path the pipeline runs:
``champollion_pipeline/src/champollion_pipeline/generate_embeddings.py`` invokes
``champollion/evaluate.py`` once per region model as

    python evaluate.py -m <model dir> -sk <crop>/mask/<side>skeleton.npy
                       -i <crop>/mask/<side>skeleton_subject.csv
                       -s <output>/<region>/full_embeddings.csv

The model directory is built the way a real training run leaves it and the
way ``evaluate.load_model`` reads it: ``.hydra/config.yaml`` (the composed,
unprocessed config Hydra saves at job start) plus the Lightning checkpoint
under ``logs/lightning_logs/version_0/checkpoints/`` (the default checkpoint
location for ``TensorBoardLogger('logs')``, the logger ``champollion.train``
uses, with cwd = the run directory). The weights come from a real one-step
fit of ``SSLModel`` on the shared synthetic single-region dataset
(conftest.py).

The subjects evaluated are a separate crop in the pipeline's layout, with a
different subject count than training and per-subject random skeletons, and
the expected value of each row is the trained encoder applied to that subject
alone. For that oracle to tell subjects apart, the encoder's BatchNorm running
statistics are recalibrated on a separate draw of inputs (not the evaluated
ones, so a missing model.eval() in evaluate.py still changes the output)
before the checkpoint is saved: after a single step they are still near their initial values, and the
eval-mode encoder maps every input to the same embedding to within ~1e-6
(the same effect that makes val_loss NaN in test_smoke_train_step.py), which
would hide rows written in the wrong order.

Offline, CPU only (CUDA hidden from the evaluate subprocess). Run from the
champollion_pipeline root:
    pixi run test-champollion
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from champollion.data.datamodule import DataModule_Learning
from champollion.ssl_model import SSLModel
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers

EVALUATE_SCRIPT = Path(__file__).resolve().parents[1] / "champollion" / "evaluate.py"
EVAL_SUBJECTS = [f"sub-eval{i}" for i in range(5)]


def _random_skeletons(volume_shape: tuple, seed: int) -> np.ndarray:
    """One random-density skeleton per evaluated subject, shape (N, D, H, W, 1)."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            (rng.random(volume_shape) < density).astype(np.float32)
            for density in np.linspace(0.05, 0.5, len(EVAL_SUBJECTS))
        ]
    )


def _as_model_input(skeletons: np.ndarray) -> torch.Tensor:
    """Skeletons as evaluate.py documents its input: binarized, channel axis first."""
    return torch.from_numpy((skeletons != 0).astype(np.float32)).permute(0, 4, 1, 2, 3)


def _evaluate_inputs(skels_path: Path) -> torch.Tensor:
    return _as_model_input(np.load(skels_path))


def _recalibrate_batchnorm(encoder: torch.nn.Module, inputs: torch.Tensor) -> None:
    """Set every BatchNorm's running stats to the exact statistics of ``inputs``."""
    norms = [m for m in encoder.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for norm in norms:
        norm.reset_running_stats()
        norm.momentum = None  # cumulative average: one pass gives the exact batch statistics
    encoder.train()
    with torch.no_grad():
        encoder(inputs)
    encoder.eval()


@pytest.fixture
def trained_model(
    tmp_path, monkeypatch, synthetic_region, compose_training_config, eval_crop
) -> tuple[Path, torch.nn.Module]:
    """Train one step, lay the run out as a model directory; return (dir, trained encoder in eval mode)."""
    raw_config = compose_training_config(*synthetic_region, process=False)
    config = compose_training_config(*synthetic_region)

    model_dir = tmp_path / "model"
    (model_dir / ".hydra").mkdir(parents=True)
    OmegaConf.save(raw_config, model_dir / ".hydra" / "config.yaml")

    # Hydra runs training with cwd = run dir (hydra.job.chdir); SSLModel and the
    # logger write cwd-relative logs/ there.
    monkeypatch.chdir(model_dir)
    data_module = DataModule_Learning(config)
    model = SSLModel(config, sample_data=data_module)
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
    )
    trainer.fit(model, data_module)
    assert trainer.global_step == 1

    encoder = model.backbones[0]
    volume_shape = np.load(eval_crop[0], mmap_mode="r").shape[1:]
    _recalibrate_batchnorm(encoder, _as_model_input(_random_skeletons(volume_shape, seed=2)))
    # Where Lightning's default ModelCheckpoint writes for TensorBoardLogger("logs").
    trainer.save_checkpoint(Path(trainer.log_dir) / "checkpoints" / "epoch=0-step=1.ckpt")
    monkeypatch.chdir(tmp_path)

    checkpoints = list((model_dir / "logs" / "lightning_logs" / "version_0" / "checkpoints").glob("*.ckpt"))
    assert len(checkpoints) == 1, f"fixture: expected one checkpoint, got {checkpoints}"
    assert encoder.num_representation_features == config.backbone_output_size
    return model_dir, encoder


@pytest.fixture
def eval_crop(tmp_path, synthetic_region) -> tuple[Path, Path]:
    """Write <crop>/mask/Lskeleton.npy (N, D, H, W, 1) and Lskeleton_subject.csv; return their paths.

    Same volume shape as the training data (the model's input_size).
    """
    volume_shape = np.load(synthetic_region[1] / "skeleton.npy", mmap_mode="r").shape[1:]
    mask_dir = tmp_path / "crops" / "2mm" / "S.C.-sylv." / "mask"
    mask_dir.mkdir(parents=True)
    np.save(mask_dir / "Lskeleton.npy", _random_skeletons(volume_shape, seed=1))
    pd.DataFrame({"Subject": EVAL_SUBJECTS}).to_csv(mask_dir / "Lskeleton_subject.csv", index=False)
    return mask_dir / "Lskeleton.npy", mask_dir / "Lskeleton_subject.csv"


def _expected_embeddings(encoder: torch.nn.Module, skels_path: Path) -> np.ndarray:
    """The trained encoder applied to each subject alone, on the input evaluate.py documents.

    Independent of evaluate.load_model, so a model rebuilt wrongly or left with
    untrained weights shows up as a value mismatch.
    """
    inputs = _evaluate_inputs(skels_path)
    with torch.no_grad():
        expected = np.stack([encoder(x.unsqueeze(0)).squeeze(0).numpy() for x in inputs])
    # Fixture sanity: rows must be distinguishable, or a row-order bug would go unseen.
    gaps = np.abs(expected[:, None, :] - expected[None, :, :]).max(axis=2)
    assert gaps[~np.eye(len(expected), dtype=bool)].min() > 1e-3, "fixture: subject embeddings not distinct"
    return expected


def test_evaluate_cli_writes_each_subjects_trained_embedding(tmp_path, trained_model, eval_crop):
    """REQ-CHAMPTEST-07: evaluate.py, run as the pipeline runs it, writes each subject's trained embedding."""
    model_dir, encoder = trained_model
    skels_path, subjects_path = eval_crop
    expected = _expected_embeddings(encoder, skels_path)
    saving_path = tmp_path / "embeddings" / "S.C.-sylv._left" / "full_embeddings.csv"
    cmd = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "-m",
        str(model_dir),
        "-sk",
        str(skels_path),
        "-i",
        str(subjects_path),
        "-s",
        str(saving_path),
    ]

    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"evaluate.py failed:\n{result.stdout}\n{result.stderr}"
    assert saving_path.is_file()
    embeddings = pd.read_csv(saving_path)
    assert list(embeddings.columns) == ["ID", *(f"dim{i}" for i in range(1, expected.shape[1] + 1))]
    assert embeddings["ID"].tolist() == EVAL_SUBJECTS
    values = embeddings.drop(columns="ID").to_numpy()
    assert values.dtype.kind == "f"
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values, expected, rtol=1e-4, atol=1e-5)
