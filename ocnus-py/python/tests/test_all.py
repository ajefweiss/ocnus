import pytest
import ocnus_py


def test_sum_as_string():
    assert ocnus_py.sum_as_string(1, 1) == "2"
