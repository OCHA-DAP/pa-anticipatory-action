from src.utils_general import math


def test_round_to_n():
    assert math.round_to_n(0.5, 5) == 0
    assert math.round_to_n(7.5, 5) == 10
    assert math.round_to_n(-2.5, 5) == 0
    assert math.round_to_n(-7.5, 5) == -10
