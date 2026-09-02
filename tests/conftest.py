"""Shared pytest fixtures.

Adds the repository root to ``sys.path`` so the tests run against the source
tree with no ROS workspace, no ``colcon build``, and no install step. That is
the whole point of keeping ``drone_bringup.core`` free of ROS imports.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drone_bringup.core.geodesy import LocalOrigin  # noqa: E402

#: PX4 SITL default home (Zurich Irchel Park). Used across the tests so that
#: hand-computed expectations stay comparable between files.
ZURICH = (47.397742, 8.545594, 488.0)


@pytest.fixture
def origin() -> LocalOrigin:
    """A LocalOrigin at the PX4 SITL default home position."""
    return LocalOrigin(*ZURICH)


@pytest.fixture
def config_dir() -> Path:
    """Path to the package's ``config/`` directory."""
    return REPO_ROOT / "config"
