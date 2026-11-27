"""Characterization tests for the subject validation in ``champollion.evaluate``.

``load_subjects`` reads the subjects CSV and must name the file and its actual
columns when the ``Subject`` column is absent. ``main`` refuses to label
embeddings by position when the subject count differs from the number of
samples on the first axis of the skeletons array; the check runs before any
model is loaded, so no checkpoint is needed.
"""

import argparse

import numpy as np
import pytest
from champollion import evaluate


def _write_csv(path, text):
    path.write_text(text)
    return path


def _args(tmp_path, subjects_path, n_samples):
    skels_path = tmp_path / "skels.npy"
    np.save(skels_path, np.ones((n_samples, 4, 4, 4, 1), dtype=np.float32))
    return argparse.Namespace(
        model_path=str(tmp_path / "no_model"),
        skels_path=str(skels_path),
        subjects_path=str(subjects_path),
        saving_path=str(tmp_path / "out" / "emb.csv"),
    )


def test_load_subjects_returns_ids_in_file_order(tmp_path):
    csv = _write_csv(tmp_path / "subjects.csv", "Subject,Age\nsub-c,30\nsub-a,20\nsub-b,25\n")

    subjects = evaluate.load_subjects(csv)

    assert list(subjects) == ["sub-c", "sub-a", "sub-b"]


def test_load_subjects_without_subject_column_raises_naming_file_and_columns(tmp_path):
    csv = _write_csv(tmp_path / "bad_subjects.csv", "ID,Age\nsub-a,20\n")

    with pytest.raises(ValueError) as excinfo:
        evaluate.load_subjects(csv)

    message = str(excinfo.value)
    assert str(csv) in message
    assert "'Subject'" in message
    assert "['ID', 'Age']" in message


@pytest.mark.parametrize("n_samples", [2, 4])
def test_main_raises_when_subject_count_differs_from_sample_count(tmp_path, n_samples):
    csv = _write_csv(tmp_path / "subjects.csv", "Subject\nsub-a\nsub-b\nsub-c\n")
    args = _args(tmp_path, csv, n_samples)

    with pytest.raises(ValueError, match="Subject count mismatch") as excinfo:
        evaluate.main(args)

    message = str(excinfo.value)
    assert "3 subjects" in message
    assert f"{n_samples} samples" in message
    assert not (tmp_path / "out").exists()
