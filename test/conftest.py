"""Shared test fixtures for the champollion_V1 test suite.

- cwd isolation: keep cwd-relative side effects of champollion imports out of
  the checkout (collection and every test run in a temporary directory).
- A tiny synthetic single-region dataset and a factory composing the training
  config from it the way the pipeline wrapper (``train_champollion.py``) does.
"""

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from champollion.utils.config import process_config
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "champollion" / "configs"

DATASET_GROUP = "synthetic"
REGION = "region"
N_SUBJECTS = 8
VOLUME_SHAPE = (12, 12, 12)  # (D, H, W); arrays carry a trailing channel axis
BATCH_SIZE = 2

# Skeleton voxel values used by the augmentations (bottom 30, top 35, surface 60).
SURFACE, BOTTOM, TOP = 60, 30, 35

_COLLECT_DIR_KEY = pytest.StashKey[Path]()


def _collect_dir(config: pytest.Config) -> Path:
    if _COLLECT_DIR_KEY not in config.stash:
        config.stash[_COLLECT_DIR_KEY] = Path(tempfile.mkdtemp(prefix="champollion-collect-"))
    return config.stash[_COLLECT_DIR_KEY]


# champollion.train creates runs/ at import time, which happens during
# collection, before any fixture can change the working directory.
@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    previous = os.getcwd()
    os.chdir(_collect_dir(collector.config))
    try:
        return (yield)
    finally:
        os.chdir(previous)


def pytest_unconfigure(config):
    collect_dir = config.stash.get(_COLLECT_DIR_KEY, None)
    if collect_dir is not None:
        shutil.rmtree(collect_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


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


@pytest.fixture
def synthetic_region(tmp_path) -> tuple[Path, Path]:
    """Write arrays/CSVs and a dataset yaml; return (config search dir, dataset_folder).

    ``dataset_folder`` holds ``skeleton.npy`` (N, D, H, W, 1), ``label.npy``,
    ``mask.npy`` and ``skeleton_subject.csv`` (one ``Subject`` column, N rows).
    """
    data_dir = tmp_path / "data"
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

    search_dir = tmp_path / "configs"
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


@pytest.fixture
def compose_training_config() -> Callable[..., DictConfig]:
    """Factory: compose the training config as the pipeline wrapper does, forced onto the CPU.

    ``factory(search_dir, data_dir, *extra_overrides, process=True)`` returns
    ``process_config(compose(...))``; with ``process=False`` it returns the
    composed config as-is, i.e. what Hydra saves as ``<run>/.hydra/config.yaml``.
    """

    def factory(search_dir: Path, data_dir: Path, *extra_overrides: str, process: bool = True) -> DictConfig:
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
            *extra_overrides,
        ]
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
            config = compose(config_name="config", overrides=overrides)
        return process_config(config) if process else config

    return factory
