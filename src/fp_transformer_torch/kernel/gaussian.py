"""Full Gaussian kernel on replica CVs (no RFF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import math

import torch


def bandwidth_column_squared(bandwidth: torch.Tensor) -> torch.Tensor:
    """σ_i² as ``(1, batch)`` for column-indexed centers."""
    vector = bandwidth.reshape(-1)
    return (vector * vector).unsqueeze(0)


def gaussian_kernel_log_normalization(
    bandwidth: torch.Tensor, feature_dim: float
) -> torch.Tensor:
    """``ln 1/(2π σ_i²)^{d/2}`` as ``(1, batch)``."""
    sigma_squared = bandwidth_column_squared(bandwidth)
    return -0.5 * feature_dim * torch.log(2.0 * torch.pi * sigma_squared)


@dataclass
class PerReplicaRescaleConfig:
    enabled: bool = False
    target_mode: str = "adaptive_quantile"
    adaptive_ref_quantile: float = 0.5
    adaptive_band_low: float = 0.8
    adaptive_band_high: float = 1.2
    bandwidth_min: float = 1e-3
    bandwidth_max: float = 8.0
    log_step_size: float = 0.05
    log_error_clip: float = 0.5
    global_pull: float = 0.02
    # Only used when target_mode == "fixed" (absolute bounds on mean-row = row_sum/B).
    fixed_row_sum_min: float = 2
    fixed_row_sum_max: float = 4
    max_rescale_iterations: int = 128
    # Rescale metric: (normalized K row-sum) / num_replicas; band = median × [low, high].
    use_normalized_kernel: bool = True
    normalize_row_sum_by_replicas: bool = True


class GaussianKernel:
    """Non-RFF Gaussian kernel with optional per-replica bandwidth rescale."""

    def __init__(
        self,
        num_replicas: int,
        *,
        bandwidth: float | torch.Tensor = 1.0,
        distance_function: Callable[[torch.Tensor], torch.Tensor],
        update_bandwidth_method: str = "rescale",
        rescale_per_replica: bool = False,
        rescale_per_replica_config: Optional[PerReplicaRescaleConfig] = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        if update_bandwidth_method != "rescale":
            raise ValueError("only update_bandwidth_method='rescale' is supported")
        self.num_replicas = int(num_replicas)
        self.distance_function = distance_function
        self.dtype = dtype
        self.device = torch.device(device or "cpu")
        bandwidth_tensor = torch.as_tensor(bandwidth, dtype=dtype, device=self.device)
        if bandwidth_tensor.ndim == 0:
            bandwidth_tensor = bandwidth_tensor.expand(self.num_replicas).clone()
        elif bandwidth_tensor.numel() == 1:
            bandwidth_tensor = bandwidth_tensor.reshape(()).expand(self.num_replicas).clone()
        else:
            bandwidth_tensor = bandwidth_tensor.reshape(-1).to(
                device=self.device, dtype=dtype
            )
            if int(bandwidth_tensor.numel()) != self.num_replicas:
                raise ValueError("bandwidth length must match num_replicas")
        self.bandwidth = bandwidth_tensor

        cfg = rescale_per_replica_config or PerReplicaRescaleConfig()
        if rescale_per_replica:
            cfg = PerReplicaRescaleConfig(**{**cfg.__dict__, "enabled": True})
        self.rescale_config = cfg
        self._last_row_sum_stats: dict[str, float | str] = {}

    def kernel_matrix(
        self,
        collective_variable: torch.Tensor,
        bandwidth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalized Gaussian ``K_ji`` matrix ``(batch, batch)``."""
        bandwidth_use = self.bandwidth if bandwidth is None else bandwidth
        if bandwidth_use.ndim == 0:
            bandwidth_use = bandwidth_use.expand(collective_variable.shape[0])
        distance_matrix = self.distance_function(collective_variable)
        feature_dim = float(
            collective_variable.shape[-2] * collective_variable.shape[-1]
        )
        gaussian_exponent = -0.5 * distance_matrix
        log_kernel = gaussian_exponent / bandwidth_column_squared(
            bandwidth_use
        ) + gaussian_kernel_log_normalization(bandwidth_use, feature_dim)
        return torch.exp(log_kernel)

    def log_kernel_density(
        self,
        collective_variable: torch.Tensor,
        bandwidth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``logsumexp_i log K_ji`` shaped ``(batch, 1, 1)`` (matches JAX ``kernel_``)."""
        bandwidth_use = self.bandwidth if bandwidth is None else bandwidth
        if bandwidth_use.ndim == 0:
            bandwidth_use = bandwidth_use.expand(collective_variable.shape[0])
        distance_matrix = self.distance_function(collective_variable)
        feature_dim = float(
            collective_variable.shape[-2] * collective_variable.shape[-1]
        )
        gaussian_exponent = -0.5 * distance_matrix
        log_kernel = gaussian_exponent / bandwidth_column_squared(
            bandwidth_use
        ) + gaussian_kernel_log_normalization(bandwidth_use, feature_dim)
        log_density = torch.logsumexp(log_kernel, dim=1, keepdim=True)
        return log_density.unsqueeze(-1)

    def kernel_gradient(
        self,
        collective_variable: torch.Tensor,
        bandwidth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """∇_s Σ_j log-kernel-density_j (same as JAX ``grad(sum(kernel_))``)."""
        variable = collective_variable.detach().requires_grad_(True)
        bandwidth_use = self.bandwidth if bandwidth is None else bandwidth
        total = torch.sum(self.log_kernel_density(variable, bandwidth_use))
        (gradient,) = torch.autograd.grad(total, variable, create_graph=False)
        return gradient.detach()

    def _unnormalized_kernel_matrix(
        self,
        collective_variable: torch.Tensor,
        bandwidth: torch.Tensor,
    ) -> torch.Tensor:
        """``K_ji = exp(-0.5 D_ji / σ_i²)`` without Gaussian normalization."""
        distance_matrix = self.distance_function(collective_variable)
        sigma_sq = bandwidth_column_squared(bandwidth)
        return torch.exp((-0.5 * distance_matrix) / sigma_sq)

    def _rescale_metric(
        self,
        collective_variable: torch.Tensor,
        bandwidth: torch.Tensor,
    ) -> torch.Tensor:
        """Per-replica rescale metric from kernel row sums.

        Unnormalized path (dialanine): ``m_j = Σ_i K̃_ji`` (no ``/B``).
        Normalized path: ``m_j = (Σ_i K_ji) / B`` when
        ``normalize_row_sum_by_replicas`` is True.
        """
        cfg = self.rescale_config
        if cfg.use_normalized_kernel:
            matrix = self.kernel_matrix(collective_variable, bandwidth)
        else:
            matrix = self._unnormalized_kernel_matrix(collective_variable, bandwidth)
        row_sum = torch.sum(matrix, dim=-1)
        if cfg.normalize_row_sum_by_replicas:
            return row_sum / float(self.num_replicas)
        return row_sum

    def update_bandwidth(self, collective_variable: torch.Tensor) -> torch.Tensor:
        """Rescale σ so mean-row metric stays in an adaptive band around its median.

        Metric (default)::

            m_j = (Σ_i K_ji) / B ,   K = normalized Gaussian kernel

        Adaptive target::

            ref = quantile(m, 0.5)
            low, high = ref * band_low, ref * band_high   # default 0.8 / 1.2

        No absolute ``[2, 4]`` clip on this metric (that was for raw unnormalized
        row sums and couples strongly to ``B``).
        """
        cfg = self.rescale_config
        bandwidth = self.bandwidth.clone()
        batch = int(collective_variable.shape[0])
        if batch != self.num_replicas:
            raise ValueError(
                f"CV batch {batch} != num_replicas {self.num_replicas}"
            )

        if not cfg.enabled:
            metric = self._rescale_metric(collective_variable, bandwidth)
            mean_m = float(metric.mean().item())
            target = float(torch.quantile(metric, 0.5).item()) if metric.numel() else 1.0
            if mean_m > 0 and target > 0:
                factor = (target / mean_m) ** 0.5
                bandwidth = torch.clamp(
                    bandwidth * factor,
                    min=cfg.bandwidth_min,
                    max=cfg.bandwidth_max,
                )
            self.bandwidth = bandwidth
            return self.bandwidth

        eps = 1e-12
        pull = float(cfg.global_pull)
        bw_min = float(cfg.bandwidth_min)
        bw_max = float(cfg.bandwidth_max)
        log_bw_min = float(math.log(bw_min))
        log_bw_max = float(math.log(bw_max))

        bandwidth = torch.clamp(bandwidth, min=bw_min, max=bw_max)
        metric = self._rescale_metric(collective_variable, bandwidth)
        low = high = None
        prev = None

        def _adaptive_bounds(metric_vec: torch.Tensor):
            ref = torch.quantile(metric_vec, cfg.adaptive_ref_quantile)
            low_b = ref * cfg.adaptive_band_low
            high_b = ref * cfg.adaptive_band_high
            # Legacy JAX path (unnormalized raw row_sum): also clip band to
            # fixed_row_sum_[min,max] (often [2,4]). New metric path skips this.
            if not (cfg.use_normalized_kernel and cfg.normalize_row_sum_by_replicas):
                low_b = torch.clamp(
                    low_b, min=cfg.fixed_row_sum_min, max=cfg.fixed_row_sum_max
                )
                high_b = torch.clamp(
                    high_b, min=cfg.fixed_row_sum_min, max=cfg.fixed_row_sum_max
                )
            return low_b, high_b

        for _ in range(int(cfg.max_rescale_iterations)):
            if cfg.target_mode == "adaptive_quantile":
                low, high = _adaptive_bounds(metric)
            else:
                low = torch.full_like(metric, cfg.fixed_row_sum_min)
                high = torch.full_like(metric, cfg.fixed_row_sum_max)

            too_high = metric > high
            too_low = metric < low
            if not bool(torch.any(too_high | too_low)):
                break

            m_safe = torch.clamp(metric, min=eps)
            err = torch.zeros_like(metric)
            err = torch.where(
                too_high,
                torch.log(torch.clamp(high, min=eps)) - torch.log(m_safe),
                err,
            )
            err = torch.where(
                too_low,
                torch.log(torch.clamp(low, min=eps)) - torch.log(m_safe),
                err,
            )
            err = torch.clamp(err, -cfg.log_error_clip, cfg.log_error_clip)
            log_bandwidth = torch.log(torch.clamp(bandwidth, min=bw_min))
            log_bandwidth = log_bandwidth + cfg.log_step_size * err
            log_bandwidth = torch.clamp(log_bandwidth, min=log_bw_min, max=log_bw_max)
            if pull != 0.0:
                log_bandwidth = (1.0 - pull) * log_bandwidth + pull * log_bandwidth.mean()
            log_bandwidth = torch.clamp(log_bandwidth, min=log_bw_min, max=log_bw_max)
            bandwidth = torch.exp(log_bandwidth)
            bandwidth = torch.clamp(bandwidth, min=bw_min, max=bw_max)

            # Recompute metric each iter (σ enters both exp and prefactor for
            # normalized K; power update is invalid with the normalization).
            metric = self._rescale_metric(collective_variable, bandwidth)

            if prev is not None and torch.equal(bandwidth, prev):
                break
            prev = bandwidth.clone()
        else:
            if cfg.target_mode == "adaptive_quantile":
                low, high = _adaptive_bounds(metric)
            else:
                low = torch.tensor(cfg.fixed_row_sum_min, dtype=metric.dtype)
                high = torch.tensor(cfg.fixed_row_sum_max, dtype=metric.dtype)

        assert low is not None and high is not None
        self.bandwidth = bandwidth
        low_v = float(low.reshape(-1)[0].item()) if torch.is_tensor(low) else float(low)
        high_v = (
            float(high.reshape(-1)[0].item()) if torch.is_tensor(high) else float(high)
        )
        self._last_row_sum_stats = {
            "row_sum_min": float(metric.min().item()),
            "row_sum_max": float(metric.max().item()),
            "row_sum_median": float(metric.median().item()),
            "target_low": low_v,
            "target_high": high_v,
            "target_mode": str(cfg.target_mode),
            "metric": "normalized_K_row_sum/B"
            if cfg.use_normalized_kernel and cfg.normalize_row_sum_by_replicas
            else "row_sum",
        }
        return self.bandwidth
