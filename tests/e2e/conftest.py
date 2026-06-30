"""Auto-apply 'e2e' marker to pytest tests in this directory."""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
