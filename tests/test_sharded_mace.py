"""Unit tests for multi-GPU MACE sharding helpers (CPU-safe)."""

from __future__ import annotations

import torch

from fp_transformer_torch.potential.sharded_mace import (
    parse_cuda_devices,
    shard_system_ranges,
)


def test_shard_system_ranges_even():
    assert shard_system_ranges(8, 4) == [(0, 2), (2, 4), (4, 6), (6, 8)]


def test_shard_system_ranges_uneven():
    assert shard_system_ranges(10, 4) == [(0, 3), (3, 6), (6, 8), (8, 10)]


def test_shard_system_ranges_more_shards_than_systems():
    assert shard_system_ranges(3, 8) == [(0, 1), (1, 2), (2, 3)]


def test_parse_devices_string():
    devs = parse_cuda_devices("0,1,2")
    assert [str(d) for d in devs] == ["cuda:0", "cuda:1", "cuda:2"]


def test_parse_devices_int_one():
    # Without CUDA this falls back; with CUDA returns cuda:0
    devs = parse_cuda_devices(1, fallback="cpu")
    assert len(devs) == 1
    assert isinstance(devs[0], torch.device)
