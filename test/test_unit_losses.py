"""Unit tests for champollion/losses.py.

Requirements REQ-CHAMPTEST-17..19 (champollion_pipeline ledger, TASK-103).

Each loss is compared with an independent, loop-based reference written from
its published definition, plus small closed-form cases. CPU, float64 where the
comparison needs tight tolerances.

Characterized conventions (not asserted to be "right", only pinned):
- NTXenLoss returns the *sum* of the two per-view mean cross-entropies (not
  their average over 2N anchors).
- BarlowTwinsLoss standardizes with the unbiased std (``Tensor.std``) but
  divides the correlation by N, so identical views give C_ii = (N-1)/N and an
  invariance term of D/N**2 rather than 0.
"""

import math

import pytest
import torch
from champollion.losses import BarlowTwinsLoss, NTXenLoss

N, D = 6, 5


def _pair(seed: int, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(N, D, generator=gen, dtype=dtype), torch.randn(N, D, generator=gen, dtype=dtype)


def _ntxent_reference(z_i: torch.Tensor, z_j: torch.Tensor, t: float) -> float:
    """SimCLR NT-Xent: per view, mean over anchors of -log softmax of the positive among the 2N-1 others."""
    n = len(z_i)
    views = [[v / v.norm() for v in z_i], [v / v.norm() for v in z_j]]
    total = 0.0
    for a in (0, 1):
        b = 1 - a
        view_loss = 0.0
        for k in range(n):
            anchor = views[a][k]
            positive = math.exp(float(anchor @ views[b][k]) / t)
            negatives = sum(math.exp(float(anchor @ views[b][m]) / t) for m in range(n) if m != k)
            negatives += sum(math.exp(float(anchor @ views[a][m]) / t) for m in range(n) if m != k)
            view_loss += -math.log(positive / (positive + negatives))
        total += view_loss / n
    return total


def _barlow_reference(z_a: torch.Tensor, z_b: torch.Tensor, lambda_param: float) -> tuple[float, float, float]:
    # Characterizes current behaviour (unbiased std, 1/N normalisation), not a
    # confirmed intended one; see pipeline ledger TASK-111 before changing it.
    n, d = z_a.shape

    def standardize(z):
        cols = []
        for j in range(d):
            col = [float(z[k, j]) for k in range(n)]
            mean = sum(col) / n
            std = math.sqrt(sum((x - mean) ** 2 for x in col) / (n - 1))
            cols.append([(x - mean) / std for x in col])
        return cols

    a, b = standardize(z_a), standardize(z_b)
    c = [[sum(a[p][k] * b[q][k] for k in range(n)) / n for q in range(d)] for p in range(d)]
    invariance = sum((1 - c[p][p]) ** 2 for p in range(d))
    redundancy = lambda_param / d * sum(c[p][q] ** 2 for p in range(d) for q in range(d) if p != q)
    return invariance + redundancy, invariance, redundancy


# REQ-CHAMPTEST-17
@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_ntxent_matches_reference_definition(temperature, seed):
    z_i, z_j = _pair(seed)
    loss = NTXenLoss(temperature=temperature)(z_i, z_j)
    assert loss.dim() == 0
    assert float(loss) == pytest.approx(_ntxent_reference(z_i, z_j, temperature), rel=1e-9)


# REQ-CHAMPTEST-17
def test_ntxent_identical_orthogonal_views_closed_form():
    t = 0.5
    z = torch.eye(N, dtype=torch.float64) * 3.0  # orthogonal rows; scale removed by normalization
    expected = 2 * (-1 / t + math.log(math.exp(1 / t) + 2 * (N - 1)))
    assert float(NTXenLoss(temperature=t)(z, z.clone())) == pytest.approx(expected, rel=1e-12)


# REQ-CHAMPTEST-17
def test_ntxent_is_symmetric_non_negative_and_returns_logits():
    z_i, z_j = _pair(3)
    loss_fn = NTXenLoss(temperature=0.1, return_logits=True)
    loss_ij, sim_zij, sim_zii, sim_zjj = loss_fn(z_i, z_j)
    loss_ji = loss_fn(z_j, z_i)[0]
    assert float(loss_ij) >= 0
    assert float(loss_ij) == pytest.approx(float(loss_ji), rel=1e-12)
    expected_zij = (torch.nn.functional.normalize(z_i, dim=-1) @ torch.nn.functional.normalize(z_j, dim=-1).T) / 0.1
    torch.testing.assert_close(sim_zij, expected_zij)
    assert sim_zii.shape == sim_zjj.shape == (N, N)


# REQ-CHAMPTEST-18
@pytest.mark.parametrize("lambda_param", [5e-3, 1.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_barlow_twins_cross_matches_reference_definition(lambda_param, seed):
    z_a, z_b = _pair(seed)
    loss, invariance, redundancy = BarlowTwinsLoss("cpu", correlation="cross", lambda_param=lambda_param)(z_a, z_b)
    ref_loss, ref_invariance, ref_redundancy = _barlow_reference(z_a, z_b, lambda_param)
    assert float(invariance) == pytest.approx(ref_invariance, rel=1e-9)
    assert float(redundancy) == pytest.approx(ref_redundancy, rel=1e-9)
    assert float(loss) == pytest.approx(ref_loss, rel=1e-9)
    assert float(loss) == pytest.approx(float(invariance + redundancy), rel=1e-12)
    assert min(float(loss), float(invariance), float(redundancy)) >= 0


# REQ-CHAMPTEST-18
def test_barlow_twins_identical_views_invariance_closed_form():
    # D/N**2 rather than 0 characterizes current behaviour; see pipeline ledger TASK-111.
    z, _ = _pair(4)
    _, invariance, _ = BarlowTwinsLoss("cpu", correlation="cross")(z, z.clone())
    assert float(invariance) == pytest.approx(D / N**2, rel=1e-9)


# REQ-CHAMPTEST-19
@pytest.mark.parametrize(
    "loss_fn",
    [NTXenLoss(temperature=0.1), lambda a, b: BarlowTwinsLoss("cpu", correlation="cross")(a, b)[0]],
    ids=["ntxent", "barlow_twins"],
)
def test_loss_backpropagates_finite_non_zero_gradients_to_both_views(loss_fn):
    z_a, z_b = (z.float().requires_grad_() for z in _pair(5))
    loss = loss_fn(z_a, z_b)
    assert torch.isfinite(loss)
    loss.backward()
    for grad in (z_a.grad, z_b.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
