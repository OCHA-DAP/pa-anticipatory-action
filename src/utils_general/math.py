import numpy as np


def round_to_n(x: float, n: int) -> int:
    """
    Round float x to the nearest multiple of n. Uses the default bankers
    rounding (to the nearest even number).
    :param x: The number to be rounded
    :param n: The integer multiple to round to
    :return: Rounded integer
    """
    return (np.around(x / n, decimals=0) * n).astype(int)
