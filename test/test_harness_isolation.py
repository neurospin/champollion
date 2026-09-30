"""Tests for REQ-CHAMPTEST-02 (champollion_pipeline ledger, TASK-100).

Importing ``champollion.train`` has side effects in the current working
directory: a module-level ``SummaryWriter()`` creates ``runs/<stamp>/`` with
an event file, and ``TensorBoardLogger('logs')`` targets ``logs/``. Tests run
from the repo (or from the champollion_pipeline root via its
``test-champollion`` pixi task) would litter those trees.

Requirement: every test collected from this ``test/`` directory executes with
a working directory outside both the champollion_V1 checkout and the pytest
invocation directory. These tests deliberately do NOT change directory
themselves, so they only pass when the isolation is systematic (e.g. an
autouse fixture in ``test/conftest.py``), which later tests inherit for free.

Offline, CPU only. Run from the champollion_pipeline root:
    pixi run test-specific external/champollion_V1/test/test_harness_isolation.py
"""

import importlib
import shutil
import sys
from pathlib import Path

import champollion

CHAMPOLLION_V1_ROOT = Path(__file__).resolve().parents[1]
SIDE_EFFECT_DIRS = ("runs", "logs")


def _protected_dirs(request) -> list[Path]:
    invocation_dir = Path(request.config.invocation_params.dir).resolve()
    return sorted({CHAMPOLLION_V1_ROOT, invocation_dir})


def _is_within(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent)


def _snapshot(dirs: list[Path]) -> dict[Path, set[str] | None]:
    """Entries of each ``<dir>/runs`` and ``<dir>/logs`` (None if absent)."""
    snap = {}
    for base in dirs:
        for name in SIDE_EFFECT_DIRS:
            target = base / name
            snap[target] = {p.name for p in target.iterdir()} if target.is_dir() else None
    return snap


def test_cwd_is_outside_checkout_and_invocation_dir(request):
    cwd = Path.cwd().resolve()
    offending = [str(d) for d in _protected_dirs(request) if _is_within(cwd, d)]
    assert not offending, f"test runs with cwd {cwd}, inside {offending}; the test harness must isolate cwd"


def test_importing_train_leaves_no_runs_or_logs_in_checkout_or_invocation_dir(request, monkeypatch):
    protected = _protected_dirs(request)
    before = _snapshot(protected)

    # Force a fresh import so the module-level side effects run inside this test.
    monkeypatch.delitem(sys.modules, "champollion.train", raising=False)
    monkeypatch.delattr(champollion, "train", raising=False)
    train = importlib.import_module("champollion.train")
    train.writer.close()

    after = _snapshot(protected)
    created = []
    for target, entries_after in after.items():
        if entries_after is None:
            continue
        entries_before = before[target]
        if entries_before is None:
            created.append(str(target))
            shutil.rmtree(target, ignore_errors=True)  # don't leave litter behind a red run
        else:
            for entry in sorted(entries_after - entries_before):
                created.append(str(target / entry))
                path = target / entry
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)

    assert not created, (
        f"importing champollion.train created {created} (SummaryWriter log_dir: {train.writer.get_logdir()})"
    )
