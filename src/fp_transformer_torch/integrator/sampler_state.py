"""Integrator state container."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplerState:
    coordinate: torch.Tensor
    momentum: torch.Tensor
    generator: torch.Generator
