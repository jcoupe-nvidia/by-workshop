"""Shared fixtures for the test suite."""
from __future__ import annotations

import pytest

from src.rollouts.scripted_traces import build_successful_episode, build_repair_episode
from src.rollouts.episode_runner import enrich_episode


@pytest.fixture
def successful_episode():
    """Raw successful episode (no rewards)."""
    return build_successful_episode()


@pytest.fixture
def repair_episode():
    """Raw repair episode with fallback events (no rewards)."""
    return build_repair_episode()


@pytest.fixture
def enriched_success(successful_episode):
    """Successful episode enriched with environment rewards."""
    return enrich_episode(successful_episode)


@pytest.fixture
def enriched_repair(repair_episode):
    """Repair episode enriched with environment rewards."""
    return enrich_episode(repair_episode)
