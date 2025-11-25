import pytest


def add(x, y):
    return x + y


def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0


@pytest.mark.parametrize("x,y,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_add(x, y, expected):
    assert add(x, y) == expected



def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 4  # 故意写错，用于演示失败场景
