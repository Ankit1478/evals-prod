"""Isolation. The bug class that costs weeks."""

import pytest

from evalkit.env.memory import MemoryEnv
from tests.conftest import WORLD


async def test_reset_restores_the_seed():
    env = MemoryEnv()
    await env.setup(WORLD)
    await env.call("cancel_order", {"order_id": "123"})
    assert env.state["orders"]["123"]["status"] == "cancelled"

    await env.reset()
    assert env.state["orders"]["123"]["status"] == "active"


async def test_mutation_cannot_reach_the_seed():
    """If the live state shared objects with the seed, reset would restore
    the ALREADY-MUTATED data and silently do nothing."""
    env = MemoryEnv()
    await env.setup(WORLD)
    await env.call("cancel_order", {"order_id": "123"})
    assert env._seed["orders"]["123"]["status"] == "active"


async def test_the_caller_s_dict_is_never_mutated():
    """WORLD is shared by every test in this file. If setup() aliased it,
    tests would corrupt each other - exactly the bug this prevents."""
    env = MemoryEnv()
    await env.setup(WORLD)
    await env.call("cancel_order", {"order_id": "123"})
    assert WORLD["orders"]["123"]["status"] == "active"


async def test_snapshot_is_a_copy_not_a_view():
    env = MemoryEnv()
    await env.setup(WORLD)
    snap = await env.snapshot()
    await env.call("cancel_order", {"order_id": "123"})
    assert snap["orders"]["123"]["status"] == "active"
