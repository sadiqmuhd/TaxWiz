import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    from app import app

    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client
