import pytest


TMP_PATH = "/tmp/test_aa_data_dir"


@pytest.fixture(autouse=True)
def all_tests_initialise(monkeypatch):
    monkeypatch.setenv("AA_DATA_DIR", TMP_PATH)
