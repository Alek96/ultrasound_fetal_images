"""This file prepares config fixtures for other tests."""

from pathlib import Path

import pytest
import rootutils


@pytest.fixture(scope="package")
def data_path() -> Path:
    return rootutils.find_root(indicator=".project-root") / "data"
