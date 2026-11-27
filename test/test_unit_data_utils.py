"""Unit tests for the whole-brain crop helpers of champollion/data/utils.py.

Pins bbox_3d, skel_to_crop, translate_mask and sample_spherical on small
synthetic arrays (TASK-V1PR3). Masks carry the trailing channel axis
(x, y, z, 1) where the helper expects it.

Characterized convention: bbox_3d returns (zmin, zmax, ymin, ymax, xmin, xmax)
with x the first array axis and z the last.
"""

import numpy as np
import pytest

from champollion.data.utils import bbox_3d, sample_spherical, skel_to_crop, translate_mask


def _block_mask():
    mask = np.zeros((6, 7, 8), dtype=np.uint8)
    mask[1:3, 2:5, 3:7] = 1  # x 1..2, y 2..4, z 3..6
    return mask


def test_bbox_3d_inclusive_bounds_order():
    assert tuple(int(v) for v in bbox_3d(_block_mask())) == (3, 6, 2, 4, 1, 2)


def test_bbox_3d_single_voxel():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[2, 1, 3] = True
    assert tuple(int(v) for v in bbox_3d(mask)) == (3, 3, 1, 1, 2, 2)


def test_bbox_3d_empty_mask_raises():
    with pytest.raises(ValueError, match="empty"):
        bbox_3d(np.zeros((3, 3, 3), dtype=np.uint8))


def test_skel_to_crop_shape_and_values():
    mask = _block_mask()[..., None]
    skel = np.arange(6 * 7 * 8, dtype=np.float64).reshape(6, 7, 8)[..., None]
    crop = skel_to_crop(skel, mask)
    assert crop.shape == (2, 3, 4, 1)
    np.testing.assert_array_equal(crop, skel[1:3, 2:5, 3:7])


def test_skel_to_crop_zeroes_voxels_outside_mask():
    mask = _block_mask()
    mask[1, 2, 3] = 0  # hole inside the box; bbox unchanged
    mask = mask[..., None]
    skel = np.ones((6, 7, 8, 1))
    crop = skel_to_crop(skel, mask)
    assert crop.shape == (2, 3, 4, 1)
    assert crop[0, 0, 0, 0] == 0
    assert crop.sum() == 2 * 3 * 4 - 1


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("shift", [2, -2])
def test_translate_mask_shift_zeroes_wrapped_voxels(axis, shift):
    mask = np.ones((5, 6, 7), dtype=np.uint8)
    translation = [0, 0, 0]
    translation[axis] = shift
    out = translate_mask(mask, translation)
    expected = np.ones((5, 6, 7), dtype=np.uint8)
    index = [slice(None)] * 3
    index[axis] = slice(0, shift) if shift > 0 else slice(shift, None)
    expected[tuple(index)] = 0
    np.testing.assert_array_equal(out, expected)


def test_translate_mask_moves_voxel_both_signs():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 1
    out = translate_mask(mask, (1, -1, 2))
    assert out.sum() == 1
    assert out[3, 1, 4] == 1


def test_translate_mask_zero_translation_is_identity():
    mask = _block_mask()
    np.testing.assert_array_equal(translate_mask(mask, (0, 0, 0)), mask)


def test_sample_spherical_unit_norm_seeded():
    np.random.seed(0)
    first = sample_spherical(3)
    assert first.shape == (3,)
    assert np.linalg.norm(first) == pytest.approx(1.0)
    np.random.seed(0)
    np.testing.assert_array_equal(sample_spherical(3), first)


@pytest.mark.parametrize("ndim", [2, 5])
def test_sample_spherical_ndim(ndim):
    np.random.seed(1)
    vec = sample_spherical(ndim)
    assert vec.shape == (ndim,)
    assert np.linalg.norm(vec) == pytest.approx(1.0)
