from pathlib import Path

import pytest


TMP_PATH = "/tmp/glofas_test"


@pytest.fixture(autouse=True)
def all_tests_initialise(monkeypatch):
    monkeypatch.setenv("AA_DATA_DIR", TMP_PATH)
