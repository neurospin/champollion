"""Unit tests for champollion/augmentations.py and champollion/data/transforms.py.

Requirements REQ-CHAMPTEST-08..16 (champollion_pipeline ledger, TASK-103).

Pure CPU numpy/tensor checks, no dataloader, no training. Arrays follow the
layout the dataset feeds to the transforms: one subject is a channel-last
``(D, H, W, 1)`` tensor, ``input_size`` is ``(1, D, H, W)``, and inside
``transform_mixed`` the augmentations see the ``(3, D, H, W, 1)`` stack
``[skeleton, foldlabel, cutin mask]`` built by ``ConcatTensor``.

Several augmentations call ``np.random.seed()`` without an argument (reseed
from OS entropy), so they cannot be made reproducible with a fixed seed; the
tests below therefore assert invariants that must hold for every random draw,
repeated over a few draws, and pin exact outputs only where the randomness can
be removed (zero magnitude, fixed cutout ``localization``).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from champollion.augmentations import PartialCutOutTensor_Roll, RotateTensor, TranslateTensor
from champollion.data.transforms import transform_mixed, transform_only_padding
from conftest import BOTTOM, SURFACE, TOP

SHAPE = (10, 10, 10)  # (D, H, W) of the padded input
INPUT_SIZE = (1, *SHAPE)
N_DRAWS = 5


def _skeleton(shape=SHAPE) -> np.ndarray:
    """Channel-last sulcus-like plane with SURFACE, BOTTOM, TOP and a value-11 voxel."""
    d, h, w = shape
    arr = np.zeros((d, h, w, 1), dtype=np.float32)
    arr[d // 2, 1 : h - 1, 1 : w - 1, 0] = SURFACE
    arr[d // 2, 1 : h - 1, 1, 0] = BOTTOM
    arr[d // 2, 1 : h - 1, w - 2, 0] = TOP
    arr[d // 2 - 1, h // 2, w // 2, 0] = 11  # value SimplifyTensor maps to background
    return arr


def _foldlabel(skeleton: np.ndarray) -> np.ndarray:
    """Foldlabel with four branches (1001..1004) tiling the skeleton plane by quadrant."""
    label = np.zeros(skeleton.shape, dtype=np.int32)
    d, h, w, _ = skeleton.shape
    quadrant = (np.arange(h)[:, None] >= h // 2) * 2 + (np.arange(w)[None, :] >= w // 2)
    label[d // 2, :, :, 0] = 1001 + quadrant
    label[skeleton == 0] = 0
    return label


def _stack(skeleton: np.ndarray) -> torch.Tensor:
    """The (3, D, H, W, 1) [skeleton, foldlabel, mask] stack ConcatTensor builds."""
    mask = np.ones(skeleton.shape, dtype=np.float32)
    return torch.from_numpy(np.stack((skeleton, _foldlabel(skeleton).astype(np.float32), mask)))


def _augmentation_config(**overrides) -> SimpleNamespace:
    """Augmentation settings of configs/augmentations/mixed.yaml, all probabilities 0 by default."""
    values = dict(
        fill_value=0,
        proba_translation=0,
        max_translation=1,
        proba_rotation=0,
        max_angle=18,
        proba_cutout=0,
        patch_size_cutout=[5, 30],
        keep_proba_per_branch_cutout=0.5,
        proba_cutin=0,
        patch_size_cutin=[20, 90],
        keep_proba_per_branch_cutin=0.5,
        keep_extremity="bottom",
        mask_constraint=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def mixed_inputs(tmp_path):
    """(skeleton tensor, foldlabel tensor, cutin mask path) as SSLDataset passes them to transform_mixed."""
    skeleton = _skeleton()
    mask = np.zeros((*SHAPE, 1), dtype=np.float32)
    mask[2:8, 2:8, 2:8, 0] = 1
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)
    return skeleton, torch.from_numpy(_foldlabel(skeleton)), mask_path


# REQ-CHAMPTEST-08
def test_padding_transform_binarizes_centred_padded_skeleton():
    small = _skeleton((7, 8, 9))  # odd margins on D and W, even on H
    expected = np.zeros((1, *SHAPE), dtype=np.float32)
    # np.pad-style centring: margin // 2 before, the rest after.
    expected[0, 1:8, 1:9, 0:9] = ((small[..., 0] != 0) & (small[..., 0] != 11)).astype(np.float32)

    out = transform_only_padding(INPUT_SIZE, _augmentation_config())(torch.from_numpy(small.copy()))

    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == INPUT_SIZE
    np.testing.assert_array_equal(out.numpy(), expected)


# REQ-CHAMPTEST-09
@pytest.mark.parametrize(
    "branch",
    [
        {},
        {"proba_rotation": 1},
        {"proba_translation": 1},
        {"proba_cutout": 1},
        {"proba_cutin": 1},
        {"proba_rotation": 1, "proba_translation": 1, "proba_cutout": 0.5, "proba_cutin": 0.5},
    ],
    ids=["none", "rotation", "translation", "cutout", "cutin", "all"],
)
def test_mixed_transform_returns_binary_input_sized_float32(mixed_inputs, branch):
    skeleton, foldlabel, mask_path = mixed_inputs
    config = _augmentation_config(**branch)
    for _ in range(N_DRAWS):
        out = transform_mixed(foldlabel, mask_path, INPUT_SIZE, config)(torch.from_numpy(skeleton.copy()))
        assert tuple(out.shape) == INPUT_SIZE
        assert out.dtype == torch.float32
        assert set(np.unique(out.numpy())) <= {0.0, 1.0}


# REQ-CHAMPTEST-10
def test_mixed_transform_without_augmentation_equals_padding_transform(mixed_inputs):
    skeleton, foldlabel, mask_path = mixed_inputs
    config = _augmentation_config()

    mixed = transform_mixed(foldlabel, mask_path, INPUT_SIZE, config)(torch.from_numpy(skeleton.copy()))
    padded = transform_only_padding(INPUT_SIZE, config)(torch.from_numpy(skeleton.copy()))

    np.testing.assert_array_equal(mixed.numpy(), padded.numpy())


# REQ-CHAMPTEST-11
def test_zero_angle_rotation_is_identity():
    stack = _stack(_skeleton())
    out = RotateTensor(max_angle=0)(stack.clone())
    np.testing.assert_array_equal(out.numpy(), stack.numpy())


# REQ-CHAMPTEST-11
def test_zero_voxel_translation_is_identity():
    skeleton = _skeleton()
    out = TranslateTensor(n_voxel=0)(torch.from_numpy(skeleton.copy()))
    np.testing.assert_array_equal(out.numpy(), skeleton)


def _shift(arr: np.ndarray, offsets) -> np.ndarray:
    """Zero-filled shift of the three leading axes by integer offsets."""
    out = np.zeros_like(arr)
    src, dst = [], []
    for size, off in zip(arr.shape[:3], offsets):
        src.append(slice(max(-off, 0), size - max(off, 0)))
        dst.append(slice(max(off, 0), size - max(-off, 0)))
    out[tuple(dst)] = arr[tuple(src)]
    return out


# REQ-CHAMPTEST-12
@pytest.mark.parametrize("n_voxel", [1, 2, [0, 1, 2]])
def test_translation_is_bounded_zero_filled_shift(n_voxel):
    rng = np.random.default_rng(1)
    arr = rng.integers(1, 5, size=(*SHAPE, 1)).astype(np.float32)  # no zeros: every shift is identifiable
    bounds = [n_voxel] * 3 if isinstance(n_voxel, int) else n_voxel
    candidates = [
        (a, b, c)
        for a in range(-bounds[0], bounds[0] + 1)
        for b in range(-bounds[1], bounds[1] + 1)
        for c in range(-bounds[2], bounds[2] + 1)
    ]
    for seed in range(N_DRAWS * 4):
        np.random.seed(seed)
        out = TranslateTensor(n_voxel)(torch.from_numpy(arr.copy())).numpy()
        assert out.shape == arr.shape
        matches = [off for off in candidates if np.array_equal(out, _shift(arr, off))]
        assert matches, f"seed {seed}: output is not a zero-filled shift within {bounds} voxels"


# REQ-CHAMPTEST-13
def test_rotation_preserves_shape_dtype_and_value_domain():
    stack = _stack(_skeleton())
    allowed = set(np.unique(stack.numpy())) | {0.0}
    for _ in range(N_DRAWS):
        out = RotateTensor(max_angle=18)(stack.clone())
        assert tuple(out.shape) == tuple(stack.shape)
        assert out.dtype == stack.dtype
        assert set(np.unique(out.numpy())) <= allowed


def _box(shape, start, size) -> np.ndarray:
    """Boolean box over the (D, H, W, 1) skeleton grid."""
    box = np.zeros(shape, dtype=bool)
    box[tuple(slice(s, s + n) for s, n in zip(start, size))] = True
    return box


CUT_SIZE = [1, 4, 4, 4]  # crop-size patch, input_size order (C, D, H, W)
CUT_CENTRE = [5, 3, 3, 0]  # localization on the (D, H, W, 1) grid -> box start = centre - size // 2


# REQ-CHAMPTEST-14
def test_cutout_keeps_skeleton_outside_box_and_only_bottom_inside():
    skeleton = _skeleton()
    box = _box(skeleton.shape, [3, 1, 1, 0], [4, 4, 4, 1])
    expected = np.where(box, skeleton * (skeleton == BOTTOM), skeleton)

    cut = PartialCutOutTensor_Roll(
        from_skeleton=True, keep_extremity="bottom", patch_size=CUT_SIZE, localization=CUT_CENTRE
    )
    out = cut(_stack(skeleton)).numpy()

    assert out.shape == (3, *skeleton.shape)
    np.testing.assert_array_equal(out[0], expected)
    assert (box & (skeleton == SURFACE)).any() and (box & (skeleton == BOTTOM)).any()  # fixture sanity


# REQ-CHAMPTEST-15
def test_cutin_keeps_skeleton_inside_box_and_only_bottom_outside():
    skeleton = _skeleton()
    box = _box(skeleton.shape, [3, 1, 1, 0], [4, 4, 4, 1])
    expected = np.where(box, skeleton, skeleton * (skeleton == BOTTOM))

    cut = PartialCutOutTensor_Roll(
        from_skeleton=False, keep_extremity="bottom", patch_size=CUT_SIZE, localization=CUT_CENTRE
    )
    out = cut(_stack(skeleton)).numpy()

    np.testing.assert_array_equal(out[0], expected)
    assert (~box & (skeleton == TOP)).any()  # fixture sanity: TOP voxels exist outside the box


# REQ-CHAMPTEST-16
@pytest.mark.parametrize("from_skeleton", [True, False], ids=["cutout", "cutin"])
def test_branch_dropping_keeps_or_drops_whole_branches(from_skeleton):
    skeleton = _skeleton()
    branches = _foldlabel(skeleton) % 1000
    # Cutout: extremity-only region = inside the box (whole volume here);
    # cutin: extremity-only region = outside the box (whole volume, empty box).
    size = [1, *SHAPE] if from_skeleton else [1, 0, 0, 0]
    bottom = skeleton == BOTTOM
    bottom_branches = set(np.unique(branches[bottom])) - {0}
    assert len(bottom_branches) >= 2  # fixture sanity
    kept_patterns = set()
    for _ in range(N_DRAWS * 4):
        cut = PartialCutOutTensor_Roll(
            from_skeleton=from_skeleton,
            keep_extremity="bottom",
            keep_proba_per_branch=0.5,
            patch_size=size,
            localization=[0, 0, 0, 0],
        )
        out = cut(_stack(skeleton)).numpy()[0]
        assert not (out[~bottom]).any()
        kept = set()
        for b in bottom_branches:
            voxels = out[bottom & (branches == b)]
            assert (voxels == BOTTOM).all() or (voxels == 0).all(), f"branch {b} partially kept"
            if (voxels == BOTTOM).all():
                kept.add(b)
        kept_patterns.add(frozenset(kept))
    assert len(kept_patterns) > 1  # branches really are dropped at random
