"""Unit tests for champollion/data/create_datasets.py (sparse, per-subject path).

Task TASK-V1PR3 (champollion_v1 ledger). Synthetic configs and CSVs only;
``SSLDataset`` is replaced by a recorder so no array is loaded.

Characterized behaviour (pinned, not judged):
- ``train_val_csv_file`` is read with ``usecols=['Subject']``: only that column survives.
- With ``load_sparse`` the coords / skeleton / foldlabel paths are ``<dir>/<sub>.npy``.
- ``deterministic_translations`` CSV rows map ``str(ID)`` to ``(-shift_x, -shift_y, -shift_z)``.
- A region without ``coords_all`` raises ``ValueError``.
"""

import os
from pathlib import Path

import pytest
from champollion.data import create_datasets
from omegaconf import OmegaConf

SUBJECTS = ["s1", "s2", "s3", "s4"]


class _RecordingDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def _recorder(monkeypatch):
    monkeypatch.setattr(create_datasets, "SSLDataset", _RecordingDataset)


def _dirs(tmp_path: Path) -> dict[str, str]:
    out = {}
    for key in ("coords", "skel", "fold"):
        d = tmp_path / key
        d.mkdir()
        out[key] = str(d)
    return out


def _config(tmp_path: Path, *, with_coords: bool = True, translations: str | None = None, **region_extra):
    d = _dirs(tmp_path)
    region = {"numpy_all": d["skel"], "foldlabel_all": d["fold"], **region_extra}
    if with_coords:
        region["coords_all"] = d["coords"]
    return OmegaConf.create(
        {
            "data": [region],
            "partition": [0.5, 0.5],
            "seed": 0,
            "deterministic_translations": translations,
            "apply_augmentations": False,
        }
    ), d


def _train_val_csv(tmp_path: Path) -> str:
    path = tmp_path / "train_val.csv"
    path.write_text("Subject,Extra\n" + "".join(f"{s},{i}\n" for i, s in enumerate(SUBJECTS)))
    return str(path)


def test_header_row_excluded_and_only_subject_column_kept(tmp_path):
    cfg, _ = _config(tmp_path, train_val_csv_file=_train_val_csv(tmp_path))
    ds = create_datasets.create_sets_without_labels_without_load(cfg)
    frame = ds["train_val"].kwargs["filenames"][0]
    assert list(frame.columns) == ["Subject"]
    assert list(frame.Subject) == SUBJECTS
    assert "Subject" not in set(frame.Subject)


def test_train_and_val_partition_train_val(tmp_path):
    cfg, _ = _config(tmp_path, train_val_csv_file=_train_val_csv(tmp_path))
    ds = create_datasets.create_sets_without_labels_without_load(cfg)
    train = list(ds["train"].kwargs["filenames"][0].Subject)
    val = list(ds["val"].kwargs["filenames"][0].Subject)
    assert len(train) == len(val) == 2
    assert sorted(train + val) == SUBJECTS


def test_sparse_paths_are_dir_slash_subject_npy(tmp_path):
    cfg, d = _config(tmp_path, train_val_csv_file=_train_val_csv(tmp_path))
    ds = create_datasets.create_sets_without_labels_without_load(cfg)
    kw = ds["train_val"].kwargs
    for key, arg in (("coords", "coords_arrays_dirs"), ("skel", "skeleton_arrays_dirs"), ("fold", "foldlabel_arrays_dirs")):
        assert list(kw[arg][0]) == [os.path.join(d[key], f"{s}.npy") for s in SUBJECTS]


def test_separate_train_and_val_csv_files_headerless(tmp_path):
    train_csv, val_csv = tmp_path / "tr.csv", tmp_path / "va.csv"
    train_csv.write_text("s1\ns2\n")
    val_csv.write_text("s3\n")
    cfg, _ = _config(tmp_path, train_csv_file=str(train_csv), val_csv_file=str(val_csv))
    ds = create_datasets.create_sets_without_labels_without_load(cfg)
    assert list(ds["train"].kwargs["filenames"][0].Subject) == ["s1", "s2"]
    assert list(ds["val"].kwargs["filenames"][0].Subject) == ["s3"]
    assert list(ds["train_val"].kwargs["filenames"][0].Subject) == ["s1", "s2", "s3"]


def test_deterministic_translations_parsed_and_negated(tmp_path):
    csv = tmp_path / "tr.csv"
    csv.write_text("ID,shift_x,shift_y,shift_z\n101,1,-2,3\n202,0,4,-5\n")
    cfg, _ = _config(tmp_path, translations=str(csv), train_val_csv_file=_train_val_csv(tmp_path))
    ds = create_datasets.create_sets_without_labels_without_load(cfg)
    assert ds["train"].kwargs["fixed_mask_tr"] == {"101": (-1, 2, -3), "202": (0, -4, 5)}


def test_no_translations_gives_none(tmp_path):
    cfg, _ = _config(tmp_path, train_val_csv_file=_train_val_csv(tmp_path))
    ds = create_datasets.create_sets_without_labels_without_load(cfg)
    assert ds["val"].kwargs["fixed_mask_tr"] is None


def test_missing_coords_all_raises_value_error(tmp_path):
    cfg, _ = _config(tmp_path, with_coords=False, train_val_csv_file=_train_val_csv(tmp_path))
    with pytest.raises(ValueError, match="coords_all"):
        create_datasets.create_sets_without_labels_without_load(cfg)


def test_non_directory_array_path_raises_value_error(tmp_path):
    cfg, d = _config(tmp_path, train_val_csv_file=_train_val_csv(tmp_path))
    f = tmp_path / "file.npy"
    f.write_text("x")
    cfg.data[0].numpy_all = str(f)
    with pytest.raises(ValueError, match="folders"):
        create_datasets.create_sets_without_labels_without_load(cfg)
