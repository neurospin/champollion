"""Unit pins for the whole-brain sparse path of ``SSLDataset``.

Tiny synthetic sparse skeletons (``coords`` + ``values`` ``.npy`` files) and a
whole-brain mask with a trailing channel axis are written to ``tmp_path``.
They pin the sparse load, the mask cropping, the ``fixed_mask_tr`` translation
and the seeded random mask translation (``max_tr``, ``offset_proba_tr``).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from champollion.data.datasets import SSLDataset
from champollion.data.utils import sample_spherical, skel_to_crop, translate_mask

BRAIN_SHAPE = (10, 11, 12)
# Non-cubic box so that a swapped axis changes the crop shape.
MASK_BOX = (slice(3, 7), slice(4, 7), slice(2, 4))  # extents 4 x 3 x 2
CROP_SHAPE = (4, 3, 2, 1)
SUBJECTS = ["sub-00", "sub-01"]


@pytest.fixture
def mask_path(tmp_path):
    mask = np.zeros((*BRAIN_SHAPE, 1), dtype=np.float32)
    mask[MASK_BOX + (0,)] = 1
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    return path


@pytest.fixture
def skeleton_dirs(tmp_path):
    """One sparse skeleton per subject: voxels inside and outside the mask."""
    coords = np.array(
        [
            [3, 4, 2],  # inside, crop corner (0, 0, 0)
            [6, 6, 3],  # inside, crop corner (3, 2, 1)
            [0, 0, 0],  # outside the mask
            [9, 10, 11],  # outside the mask
        ]
    ).T
    values = np.array([60, 60, 60, 60], dtype=np.float32)
    paths = []
    for subject in SUBJECTS:
        coords_path = tmp_path / f"{subject}_coords.npy"
        skel_path = tmp_path / f"{subject}_skel.npy"
        np.save(coords_path, coords)
        np.save(skel_path, values)
        paths.append((coords_path, skel_path))
    return [[c for c, _ in paths]], [[s for _, s in paths]]


def _config(mask_path, max_tr=0, offset_proba_tr=1.0):
    region = SimpleNamespace(
        input_size=list(CROP_SHAPE[-1:] + CROP_SHAPE[:-1]),
        cutin_mask_path=str(mask_path),
        mask_path=str(mask_path),
    )
    return SimpleNamespace(
        data=[region], fill_value=0, max_tr=max_tr, offset_proba_tr=offset_proba_tr
    )


def _dataset(mask_path, skeleton_dirs, config=None, fixed_mask_tr=None):
    coords_dirs, skel_dirs = skeleton_dirs
    filenames = [pd.DataFrame({"Subject": SUBJECTS})]
    return SSLDataset(
        filenames,
        config or _config(mask_path),
        apply_transform=False,
        coords_arrays_dirs=coords_dirs,
        skeleton_arrays_dirs=skel_dirs,
        fixed_mask_tr=fixed_mask_tr,
    )


def _view(dataset, idx=0):
    (views, subject), = dataset[idx]
    return views.numpy(), subject


def test_sample_shape_equals_mask_crop_shape(mask_path, skeleton_dirs):
    views, subject = _view(_dataset(mask_path, skeleton_dirs))
    assert subject == SUBJECTS[0]
    assert views.shape == (2, 1, *CROP_SHAPE[:-1])


def test_sparse_voxels_land_in_crop_and_outside_ones_are_dropped(mask_path, skeleton_dirs):
    views, _ = _view(_dataset(mask_path, skeleton_dirs))
    for view in views:
        crop = view[0]
        assert crop.sum() == 2
        assert crop[0, 0, 0] == 1
        assert crop[3, 2, 1] == 1


def test_missing_fixed_translation_raises_key_error_naming_count(mask_path, skeleton_dirs):
    with pytest.raises(KeyError, match="No translation for 1 subjects"):
        _dataset(mask_path, skeleton_dirs, fixed_mask_tr={SUBJECTS[0]: [0, 0, 0]})


def test_fixed_translation_is_applied_to_the_mask(mask_path, skeleton_dirs):
    fixed = {SUBJECTS[0]: [1, 0, 0], SUBJECTS[1]: [0, 0, 0]}
    dataset = _dataset(mask_path, skeleton_dirs, fixed_mask_tr=fixed)
    # Mask moved by +1 along x: voxel (3, 4, 2) leaves it, (6, 6, 3) stays.
    # The (7, 6, 3) voxel enters it; the crop keeps the box extents.
    views, _ = _view(dataset, 0)
    crop = views[0, 0]
    assert crop.shape == CROP_SHAPE[:-1]
    assert crop[0, 0, 0] == 0  # (3, 4, 2) is outside the shifted mask
    assert crop[2, 2, 1] == 1  # (6, 6, 3) is crop index (2, 2, 1) of the shifted box
    unshifted, _ = _view(dataset, 1)
    assert unshifted[0, 0, 0, 0, 0] == 1


def test_fixed_translation_values_are_rounded_to_integers(mask_path, skeleton_dirs):
    fixed = {s: [0.6, -0.4, 0.2] for s in SUBJECTS}
    dataset = _dataset(mask_path, skeleton_dirs, fixed_mask_tr=fixed)
    assert all(tr.tolist() == [1, 0, 0] for tr in dataset.fixed_mask_tr.values())


def test_random_translation_with_zero_max_tr_is_identity(mask_path, skeleton_dirs):
    np.random.seed(0)
    config = _config(mask_path, max_tr=0, offset_proba_tr=0.0)
    views, _ = _view(_dataset(mask_path, skeleton_dirs, config))
    reference, _ = _view(_dataset(mask_path, skeleton_dirs))
    np.testing.assert_array_equal(views, reference)


def test_offset_proba_one_never_translates(mask_path, skeleton_dirs):
    config = _config(mask_path, max_tr=3, offset_proba_tr=1.0)
    reference, _ = _view(_dataset(mask_path, skeleton_dirs))
    for seed in range(5):
        np.random.seed(seed)
        views, _ = _view(_dataset(mask_path, skeleton_dirs, config))
        np.testing.assert_array_equal(views, reference)


def test_offset_proba_zero_translates_with_seeded_rng(mask_path, skeleton_dirs):
    seed, max_tr = 3, 3
    # Replay the dataset's draws: direction, offset flag, radius.
    np.random.seed(seed)
    direction = sample_spherical(3)
    flag = int(np.random.rand() > 0.0)
    radius = np.random.randint(0, max_tr + 1)
    translation = np.round(direction * radius * flag).astype(int)
    assert translation.any(), "pick a seed whose translation is not null"

    mask = np.load(mask_path)
    coords = np.array([[3, 4, 2], [6, 6, 3], [0, 0, 0], [9, 10, 11]]).T
    dense = np.zeros((*BRAIN_SHAPE, 1), dtype=np.float32)
    dense[tuple(coords)] = 60
    expected = skel_to_crop(dense, translate_mask(mask, translation))
    expected = (expected[..., 0] > 0).astype(np.float32)

    np.random.seed(seed)
    config = _config(mask_path, max_tr=max_tr, offset_proba_tr=0.0)
    # The cropped shape follows the (translated) mask box, not the unshifted one.
    config.data[0].input_size = [1, *expected.shape]
    views, _ = _view(_dataset(mask_path, skeleton_dirs, config))
    np.testing.assert_array_equal(views[0, 0], expected)
