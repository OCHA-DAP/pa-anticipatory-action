from os import environ, mkdir
from os.path import join, exists
from shutil import rmtree
from tempfile import gettempdir

import pytest


TMP_PATH = join(gettempdir(), "test_anticipatory_action")


@pytest.fixture(scope="session", autouse=True)
def all_tests_initialise(request, session_mocker):
    if exists(TMP_PATH):
        rmtree(TMP_PATH)
    mkdir(TMP_PATH)
    session_mocker.patch.dict(environ, {"AA_DATA_DIR": TMP_PATH})
    yield
    if request.node.testsfailed == 0:
        rmtree(TMP_PATH)


