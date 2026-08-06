import torch

from fp_transformer_torch import ParticlePairDistance, Torsion


def test_particle_pair_distance_shape():
    x = torch.randn(5, 38, 2, dtype=torch.float64)
    cv = ParticlePairDistance([0, 1])(x)
    assert cv.shape == (5, 1, 1)
    manual = torch.linalg.norm(x[:, 0] - x[:, 1], dim=-1)
    assert torch.allclose(cv[:, 0, 0], manual)


def test_torsion_shape_and_grad():
    x = torch.randn(3, 20, 3, dtype=torch.float64, requires_grad=True)
    indices = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
    angles = Torsion(indices)(x)
    assert angles.shape == (3, 2, 1)
    loss = angles.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
