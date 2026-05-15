import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end integration tests that require a running docker-compose stack",
    )
