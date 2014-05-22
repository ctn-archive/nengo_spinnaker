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

    if signed:
        v = 1 << (n_bits-1)
        max_value = kbits(v-1,
                          n_bits=n_bits, n_frac=n_frac, signed=signed)
        min_value = kbits(v,
                          n_bits=n_bits, n_frac=n_frac, signed=signed)
    else:
        max_value = kbits((1 << n_bits)-1,
                          n_bits=n_bits, n_frac=n_frac, signed=signed)
        min_value = 0

    if isinstance(value, (int, long, float)):
        # Saturate
        value = min(value, max_value)
        value = max(value, min_value)

        # Shift
        value *= 2**n_frac

        # do the cast to int before the sign adjustment so that it rounds
        # towards zero rather than always rounding down
        value = int(value)

        # Negate if necessary
        if signed and value < 0:
            value += (1 << n_bits)

        # Just to be on the safe side -- this should never happen
        assert 0 <= value < (1 << n_bits)

        return value
    elif isinstance(value, collections.Iterable):
        return [bitsk(v, n_bits=n_bits, n_frac=n_frac, signed=signed)
                for v in value]
    else:
        raise TypeError('Values must be numbers or iterables')


def kbits(value, n_bits=32, n_frac=15, signed=True):
    """Convert the given value(s) from a fixed point representation."""
    if isinstance(value, (int, long)):
        if signed and value & (1 << (n_bits - 1)):
            value -= (1 << n_bits)

        return value * 2**-n_frac
    elif isinstance(value, collections.Iterable):
        return [kbits(v, n_bits=n_bits, n_frac=n_frac, signed=signed)
                for v in value]
    else:
        raise TypeError('Values must be ints or iterables')
