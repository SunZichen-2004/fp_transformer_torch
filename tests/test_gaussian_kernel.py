import torch
import pytest

from fp_transformer_torch.distance import pairwise_distance, torus_pairwise_distance
from fp_transformer_torch.kernel import GaussianKernel


def test_kernel_matrix_symmetric_positive():
    x = torch.randn(6, 1, 1, dtype=torch.float64)
    kernel = GaussianKernel(
        6, bandwidth=0.5, distance_function=pairwise_distance, dtype=torch.float64
    )
    matrix = kernel.kernel_matrix(x)
    assert matrix.shape == (6, 6)
    assert torch.allclose(matrix, matrix.T, atol=1e-10)
    assert torch.all(matrix > 0)


def test_kernel_gradient_finite():
    x = torch.randn(4, 2, 1, dtype=torch.float64)
    kernel = GaussianKernel(
        4, bandwidth=0.3, distance_function=pairwise_distance, dtype=torch.float64
    )
    grad = kernel.kernel_gradient(x)
    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize("periodic", [False, True])
def test_default_preserves_joint_query_and_center_gradient(periodic):
    x = torch.tensor(
        [[-0.5, 0.4], [0.2, -0.7], [1.1, 1.7], [2.0, -0.1]], dtype=torch.float64
    ).unsqueeze(-1)
    bw = x.new_tensor([0.4, 0.6, 0.9, 0.7])
    distance = torus_pairwise_distance if periodic else pairwise_distance
    kernel = GaussianKernel(4, bandwidth=bw, distance_function=distance)
    diff = x.flatten(1)[:, None, :] - x.flatten(1)[None, :, :]
    squared_distance = (
        (2 - 2 * diff.cos()).mean(-1) if periodic else diff.square().mean(-1)
    )
    log_kernel = (
        -squared_distance / (2 * bw.square()[None, :])
        - torch.log(2 * torch.pi * bw.square())[None, :]
    )
    weight = torch.softmax(log_kernel, dim=1) / (2 * bw.square()[None, :])
    pair_derivative = weight[:, :, None] * (diff.sin() if periodic else diff)
    expected = (-pair_derivative.sum(dim=1) + pair_derivative.sum(dim=0)).unsqueeze(-1)
    torch.testing.assert_close(
        kernel.kernel_gradient(x), expected, atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("distance", [pairwise_distance, torus_pairwise_distance])
def test_cross_distance_matches_explicit_pair_formula(distance):
    query = torch.tensor([[-0.5, 0.4], [0.2, -0.7]], dtype=torch.float64).unsqueeze(-1)
    centers = torch.tensor(
        [[1.0, 2.0], [0.0, -0.1], [2.1, 0.7]], dtype=query.dtype
    ).unsqueeze(-1)
    difference = query.flatten(1)[:, None, :] - centers.flatten(1)[None, :, :]
    expected = (
        (2 - 2 * difference.cos()).mean(-1)
        if distance is torus_pairwise_distance
        else difference.square().mean(-1)
    )
    torch.testing.assert_close(distance(query, centers), expected)
