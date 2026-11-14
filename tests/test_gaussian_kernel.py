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


def test_two_point_score_freezes_centers():
    x = torch.tensor([0.0, 1.0], dtype=torch.float64).reshape(2, 1, 1)
    kernel = GaussianKernel(
        2, bandwidth=1.0, distance_function=pairwise_distance, freeze_centers=True
    )
    expected = x.new_tensor([1.0, -1.0]).reshape_as(x) / (1 + x.new_tensor(0.5).exp())
    torch.testing.assert_close(kernel.kernel_gradient(x), expected)


@pytest.mark.parametrize("periodic", [False, True])
def test_default_preserves_joint_query_and_center_gradient(periodic):
    x = torch.tensor(
        [[-0.5, 0.4], [0.2, -0.7], [1.1, 1.7], [2.0, -0.1]], dtype=torch.float64
    ).unsqueeze(-1)
    bw = x.new_tensor([0.4, 0.6, 0.9, 0.7])
    distance = torus_pairwise_distance if periodic else pairwise_distance
    kernel = GaussianKernel(4, bandwidth=bw, distance_function=distance)
    assert kernel.freeze_centers is False
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
    explicit_original = GaussianKernel(
        4, bandwidth=bw, distance_function=distance, freeze_centers=False
    )
    torch.testing.assert_close(
        kernel.kernel_gradient(x), explicit_original.kernel_gradient(x), atol=0, rtol=0
    )


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("custom_callable", [False, True])
def test_score_matches_independent_fixed_center_finite_difference(
    periodic, custom_callable
):
    x = torch.tensor(
        [[-0.5, 0.4], [0.2, -0.7], [1.1, 1.7], [2.0, -0.1]], dtype=torch.float64
    ).unsqueeze(-1)
    bandwidth = torch.tensor([0.4, 0.6, 0.9, 0.7], dtype=x.dtype, requires_grad=True)
    function = torus_pairwise_distance if periodic else pairwise_distance
    # A one-argument user callable must remain supported, with frozen centers.
    distance = (lambda s: function(s)) if custom_callable else function
    kernel = GaussianKernel(
        4, bandwidth=bandwidth, distance_function=distance, freeze_centers=True
    )

    def independent_log_density(query):
        diff = query.flatten(1)[:, None, :] - x.flatten(1)[None, :, :]
        distance = (2 - 2 * diff.cos()).mean(-1) if periodic else diff.square().mean(-1)
        log_kernel = (
            -distance / (2 * bandwidth.detach().square()[None, :])
            - torch.log(2 * torch.pi * bandwidth.detach().square())[None, :]
        )
        return torch.logsumexp(log_kernel, dim=1)

    actual = kernel.kernel_gradient(x)
    numerical = torch.zeros_like(x)
    for row in range(len(x)):
        for feature in range(2):
            plus, minus = x.clone(), x.clone()
            plus[row, feature] += 1e-6
            minus[row, feature] -= 1e-6
            numerical[row, feature] = (
                independent_log_density(plus)[row] - independent_log_density(minus)[row]
            ) / 2e-6
    torch.testing.assert_close(actual, numerical, atol=1e-8, rtol=1e-7)
    assert not actual.requires_grad
    assert bandwidth.grad is None
    torch.testing.assert_close(
        kernel.log_kernel_density(x).flatten(), independent_log_density(x)
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
