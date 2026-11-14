import itertools
import numpy as np
import pytest
import torch
from fp_transformer_torch.output import TrajectoryArchive
from fp_transformer_torch.reconstruct_FES.archive import (
    joint_kde,
    project_log_probability,
    reconstruct_sfpt,
)
from fp_transformer_torch.reconstruct_FES.train_max_likelihood_torch import (
    load_cv_datasets,
)


@pytest.mark.parametrize("dim,periodic", itertools.product([2, 3], [True, False]))
def test_separable_kde_matches_direct_source_kernel(dim, periodic):
    rng = np.random.default_rng(4)
    cv = rng.normal(size=(2, 5, dim))
    bw = rng.uniform(0.1, 0.5, (2, 5))
    axis = np.linspace(-2, 2, 7)
    actual = joint_kde(cv, bw, [axis] * dim, periodic=periodic, chunk=3)
    points = np.stack(np.meshgrid(*([axis] * dim), indexing="ij"), -1).reshape(-1, dim)
    delta = points[:, None, :] - cv.reshape(1, -1, dim)
    distance = 2 - 2 * np.cos(delta).mean(-1) if periodic else (delta**2).mean(-1)
    b = bw.reshape(1, -1)
    expected = (
        (np.exp(-0.5 * distance / b**2) / (2 * np.pi * b**2) ** (dim / 2))
        .mean(-1)
        .reshape(actual.shape)
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-14)


def test_archive_round_trip_and_neural_fitter_uses_only_production(tmp_path):
    path = tmp_path / "trajectory.npz"
    writer = TrajectoryArchive(path, metadata={"seed": 0})
    x = np.zeros((4, 2, 3))
    cv = np.arange(8).reshape(4, 2) * 0.1
    writer.append("heal", 5, 0.001, 0.5, x, cv, np.ones(4) * 0.3)
    writer.append("production", 0, 0.001, 0.1, x, cv, np.ones(4) * 0.3)
    writer.append("production", 10, 0.001, 0.1, x, cv, np.ones(4) * 0.3)
    writer.save(complete=True)
    with np.load(path, allow_pickle=False) as z:
        assert z["cv"].shape == (3, 4, 2)
        assert z["coordinates"].shape == (3, 4, 2, 3)
        assert z["complete"]
    samples, lambdas, counts, dim = load_cv_datasets(tmp_path)
    assert lambdas == [0.1] and counts == [4] and dim == 2
    np.testing.assert_allclose(samples[0], cv)
    result = reconstruct_sfpt(
        path, [np.linspace(-np.pi, np.pi, 12, endpoint=False)] * 2
    )
    assert np.isfinite(result["joint_fes"]).all()
    np.testing.assert_allclose(np.exp(result["joint_log_probability"]).sum(), 1.0)


def test_reconstruct_joint_before_marginalizing():
    density = np.asarray([[0.1, 0.8], [0.3, 0.4]])
    logp = 11 * np.log(density)
    result = project_log_probability(logp, [np.array([-1.0, 1.0])] * 2)
    physical = density**11
    physical /= physical.sum()
    expected = -0.001987204259 * 300 * np.log(physical.sum(axis=1))
    expected -= expected.min()
    np.testing.assert_allclose(result["fes_1d_0"], expected)
    incorrect = -0.001987204259 * 300 * 11 * np.log(density.sum(axis=1))
    incorrect -= incorrect.min()
    assert not np.allclose(incorrect, expected)


def test_incomplete_output_is_explicit(tmp_path):
    writer = TrajectoryArchive(tmp_path / "trajectory.npz")
    writer.save()
    with pytest.raises(ValueError, match="incomplete"):
        reconstruct_sfpt(tmp_path / "trajectory.npz", [np.array([-1.0, 1.0])] * 2)


def test_nonperiodic_one_dimensional_archive(tmp_path):
    writer = TrajectoryArchive(tmp_path / "trajectory.npz")
    writer.append(
        "production",
        10,
        0.001,
        0.1,
        np.zeros((4, 1, 3)),
        np.array([[-1.0], [-0.8], [0.5], [1.0]]),
        np.full(4, 0.4),
    )
    writer.save(complete=True)
    axis = np.linspace(-2, 2, 31)
    result = reconstruct_sfpt(writer.path, [axis], periodic=False)
    probability = np.exp(result["joint_log_probability"])
    expected = (
        0.001987204259
        * 300
        * np.log(probability[axis < 0].sum() / probability[axis >= 0].sum())
    )
    np.testing.assert_allclose(result["delta_g_kcal_mol"], expected)
    assert np.isfinite(result["fes_1d_0"]).all()
