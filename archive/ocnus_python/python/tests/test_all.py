import pytest
import ocnus


def test_sum_as_string():
    assert ocnus.sum_as_string(1, 1) == "2"
