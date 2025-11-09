"""Pytest fixtures compartidas."""

import pytest
import yaml

@pytest.fixture
def config_fixture():
    """Carga config.yaml para tests."""
    with open('config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
