"""Numerical contracts for local sample-point bandwidth updates."""

import pytest
import torch

from fp_transformer_torch.distance import pairwise_distance, torus_pairwise_distance
from fp_transformer_torch.kernel.gaussian import GaussianKernel, PerReplicaRescaleConfig


def make_kernel(n, method, **kwargs):
    return GaussianKernel(
        n,
        distance_function=pairwise_distance,
        update_bandwidth_method=method,
        rescale_per_replica=True,
        rescale_per_replica_config=PerReplicaRescaleConfig(**kwargs),
        dtype=torch.float64,
    )


def test_local_mass_solves_center_occupancy():
    x = torch.randn(
        48, 2, 1, generator=torch.Generator().manual_seed(12), dtype=torch.float64
    )
    kernel = make_kernel(48, "local_mass", target_occupancy=4, solver_iterations=32)
    bandwidth = kernel.update_bandwidth(x)
    mass = torch.exp(-pairwise_distance(x) / (2 * bandwidth.square())).sum(dim=0)
    torch.testing.assert_close(mass, torch.full_like(mass, 4), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("method", ["local_mass", "knn", "log_mass"])
def test_coincident_ensemble_keeps_previous_positive_scale(method):
    kernel = make_kernel(16, method)
    before = kernel.bandwidth.clone()
    torch.testing.assert_close(kernel.update_bandwidth(torch.zeros(16, 2, 1)), before)


@pytest.mark.parametrize("method", ["local_mass", "knn"])
def test_local_rule_is_permutation_and_scale_equivariant(method):
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(32, 2, 1, generator=generator, dtype=torch.float64)
    permutation = torch.randperm(32, generator=generator)
    expected = make_kernel(32, method).update_bandwidth(x)
    actual = make_kernel(32, method).update_bandwidth(3 * x[permutation])
    torch.testing.assert_close(
        actual, 3 * expected[permutation], atol=1e-10, rtol=1e-10
    )


def test_torus_rule_respects_periodic_seam():
    x = torch.randn(
        32, 2, 1, generator=torch.Generator().manual_seed(7), dtype=torch.float64
    )
    kernel = make_kernel(32, "local_mass")
    kernel.distance_function = torus_pairwise_distance
    expected = kernel.update_bandwidth(x).clone()
    x[:16] += 2 * torch.pi
    torch.testing.assert_close(
        kernel.update_bandwidth(x), expected, atol=1e-10, rtol=1e-10
    )


def test_invalid_occupancy_is_rejected():
    with pytest.raises(ValueError, match="target_occupancy"):
        make_kernel(8, "local_mass", target_occupancy=8)


@pytest.mark.parametrize("method", ["local_mass", "knn", "log_mass"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_identical_nonzero_angles_do_not_create_a_roundoff_scale(method, dtype):
    x = torch.randn(1, 2, 1, generator=torch.Generator().manual_seed(1), dtype=dtype)
    x = x.expand(32, -1, -1).clone()
    kernel = GaussianKernel(
        32,
        distance_function=torus_pairwise_distance,
        update_bandwidth_method=method,
        dtype=dtype,
    )
    before = kernel.bandwidth.clone()
    torch.testing.assert_close(kernel.update_bandwidth(x), before)


def test_explicit_method_selects_local_update_without_legacy_flag():
    kernel = GaussianKernel(
        8,
        distance_function=pairwise_distance,
        update_bandwidth_method="knn",
    )
    x = torch.arange(8, dtype=torch.float64).reshape(8, 1, 1)
    actual = kernel.update_bandwidth(x)
    expected = pairwise_distance(x).kthvalue(4, dim=0).values.sqrt() / 2**0.5
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("method", ["local_mass", "knn", "log_mass"])
def test_local_rules_do_not_use_legacy_protection_parameters(method):
    x = torch.randn(
        32, 2, 1, generator=torch.Generator().manual_seed(9), dtype=torch.float64
    )
    expected = make_kernel(32, method).update_bandwidth(x)
    altered = make_kernel(
        32,
        method,
        bandwidth_min=0.9,
        bandwidth_max=0.91,
        global_pull=0.8,
        log_step_size=10,
        log_error_clip=0,
        max_rescale_iterations=1,
        adaptive_band_low=0.01,
        adaptive_band_high=100,
        fixed_row_sum_min=0.01,
        fixed_row_sum_max=1e6,
    ).update_bandwidth(x)
    torch.testing.assert_close(altered, expected, atol=0, rtol=0)
