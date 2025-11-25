import pytest

@pytest.fixture
def sample_data():
    return {"name": "Alice", "age": 20}

def test_is_adult(sample_data):
    assert sample_data["age"] >= 18