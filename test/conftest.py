"""Keep cwd-relative side effects of champollion imports out of the checkout."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

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
