import torch

from fp_transformer_torch.distance import pairwise_distance, torus_pairwise_distance


def test_pairwise_distance_shape_and_symmetry():
    x = torch.randn(4, 2, 3, dtype=torch.float64)
    d = pairwise_distance(x)
    assert d.shape == (4, 4)
    assert torch.allclose(d, d.T, atol=1e-10)
    assert torch.allclose(torch.diag(d), torch.zeros(4, dtype=torch.float64), atol=1e-10)


def test_torus_distance_periodic():
    a = torch.zeros(2, 1, 1, dtype=torch.float64)
    a[1, 0, 0] = 2 * torch.pi
    d = torus_pairwise_distance(a)
    assert d.shape == (2, 2)
    assert float(d[0, 1]) < 1e-10
