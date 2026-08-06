import torch

from fp_transformer_torch.distance import pairwise_distance
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
