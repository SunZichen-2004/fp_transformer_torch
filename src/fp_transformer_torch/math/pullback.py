"""Pull CV-space gradients back to Cartesian coordinates."""

from __future__ import annotations

from typing import Callable

import torch


def pullback_collective_variable_gradient(
    coordinate: torch.Tensor,
    collective_variable_fn: Callable[[torch.Tensor], torch.Tensor],
    collective_variable_gradient: torch.Tensor,
) -> torch.Tensor:
    """VJP of ``collective_variable_fn`` applied to ``collective_variable_gradient``.

    ``coordinate`` must require grad (or will be cloned with ``requires_grad_``).
    """
    if not coordinate.requires_grad:
        coordinate = coordinate.detach().requires_grad_(True)
    collective_variable = collective_variable_fn(coordinate)
    # Match shapes for the cotangent.
    cotangent = collective_variable_gradient.reshape_as(collective_variable)
    (cartesian_gradient,) = torch.autograd.grad(
        collective_variable,
        coordinate,
        grad_outputs=cotangent,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    return cartesian_gradient
