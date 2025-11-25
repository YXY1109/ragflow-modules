# test_class.py

class TestMath:
    def test_add(self):
        assert 1 + 2 == 3

    def test_multiply(self):
        assert 2 * 3 == 6

class TestString:
    def test_upper(self):
        assert "hello".upper() == "HELLO"

    def test_split(self):
        assert "hello world".split() == ["hello", "world"]