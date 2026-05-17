from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


@pytest.fixture
def signal_3c() -> np.ndarray:
    t = np.linspace(0, 2, 500)
    return np.stack([np.sin(2 * np.pi * (i + 1) * t) for i in range(3)], axis=1)


@pytest.fixture
def meta_3c() -> dict:
    return {"fs": 250.0, "phi": np.eye(3)}
