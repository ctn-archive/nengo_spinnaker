import collections
import numpy as np


def bitsk(value, n_bits=32, n_frac=15, signed=True):
    """Convert the given value(s) into a fixed point representation.

    :param value: a value or iterable to convert
    :param n_bits: total number of bits for the representation
    :param n_frac: number of fractional bits
    :param signed: signed or unsigned representation
    :returns: an int or array of ints representing the given value in fixed
              point.
    """
    max_fracts = sum([2**-n for n in range(1, n_frac+1)])
    max_value = (1 << (n_bits - n_frac - (1 if signed else 0))) - 1
    min_value = -max_value - 1 if signed else 0

    max_value += max_fracts
    min_value -= max_fracts  # Check this!

    if isinstance(value, (int, float)):
        # Saturate
        value = float(value)
        value = min(value, max_value)
        value = max(value, min_value)

        # Shift
        value *= 2**n_frac

        # Negate if necessary
        if signed and value < 0:
            value += (1 << n_bits)
            value = int(value) & sum([1 << n for n in range(n_bits)])

        return int(value)
    elif isinstance(value, collections.Iterable):
        return [bitsk(v) for v in value]


def kbits(value, n_bits=32, n_frac=15, signed=True):
    """Convert the given value(s) from a fixed point representation."""
    if isinstance(value, int):
        if signed and value & (1 << (n_bits - 1)):
            value -= (1 << n_bits)

        return value * 2**-n_frac

    value = np.asarray(value)
    if signed:
        value[value >= (1 << (n_bits - 1))] -= (1 << n_bits)

    return value * 2**-n_frac
