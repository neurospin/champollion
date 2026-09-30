"""Characterization tests for ``champollion.utils.put_together_embeddings_files.get_model_paths``.

``get_model_paths`` is the only part of that module still used: the pipeline's
``src/generate_umap_reference.py`` imports it (by path) to find the trained
models below a models directory, then strips ``models_dir + "/"`` from each
returned path to recover the ``<region>_<hemi>`` directory name. The pipeline's
own tests mock it out, so these tests pin its real behavior (REQ-CHAMPTEST-20).
"""

from pathlib import Path

from champollion.utils.put_together_embeddings_files import get_model_paths


def _make_model(model_dir: Path) -> Path:
    (model_dir / ".hydra").mkdir(parents=True)
    (model_dir / ".hydra" / "config.yaml").write_text("model: synthetic\n")
    return model_dir


def test_returns_model_dirs_at_any_depth_and_ignores_files_and_non_model_dirs(tmp_path):
    root = tmp_path / "models"
    shallow = _make_model(root / "SC-sylv_left")
    deep = _make_model(root / "FColl-SRh_right" / "run_a" / "12-00-00_1")
    sibling = _make_model(root / "FColl-SRh_right" / "run_b")
    (root / "empty_region").mkdir()
    (root / "no_config" / ".hydra").mkdir(parents=True)  # .hydra without config.yaml
    (root / "no_config" / ".hydra" / "overrides.yaml").write_text("[]\n")
    (root / "README.txt").write_text("not a model\n")
    (root / "FColl-SRh_right" / "notes.csv").write_text("ID\n")

    found = get_model_paths(str(root))

    assert sorted(found) == sorted(f"{root}/{p.relative_to(root).as_posix()}" for p in (shallow, deep, sibling))


def test_does_not_descend_into_a_model_dir(tmp_path):
    root = tmp_path / "models"
    outer = _make_model(root / "SC-sylv_left")
    _make_model(outer / "nested_model")

    assert get_model_paths(str(root)) == [f"{root}/SC-sylv_left"]


def test_root_itself_is_not_returned_even_when_it_is_a_model(tmp_path):
    root = _make_model(tmp_path / "models")
    child = _make_model(root / "SC-sylv_left")

    assert get_model_paths(str(root)) == [f"{root}/{child.name}"]


def test_empty_root_returns_empty_list(tmp_path):
    assert get_model_paths(str(tmp_path)) == []


def test_returned_paths_are_root_prefixed_and_start_with_the_top_level_dir(tmp_path):
    """Each path is ``root + '/' + rel`` with no ``//``, and its first component is the top-level dir."""
    root = tmp_path / "models"
    _make_model(root / "S.T.s.ter.pf.or._left" / "run" / "seed_1")
    _make_model(root / "FColl-SRh_right")
    prefix = str(root) + "/"

    paths = get_model_paths(str(root))

    assert all(p.startswith(prefix) and "//" not in p for p in paths)
    assert sorted(p[len(prefix) :].split("/")[0] for p in paths) == ["FColl-SRh_right", "S.T.s.ter.pf.or._left"]
