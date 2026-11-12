# conftest.py — shared pytest setup for Employee Portal tests
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    return TestClient(app)
