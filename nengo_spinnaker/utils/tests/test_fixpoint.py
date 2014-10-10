"""Tests for the fixed-point utility module.
"""

from ..fixpoint import bitsk, kbits


def test_bitsk_basic():
    assert bitsk(0.0) == 0
    assert bitsk(1.0) == 0x8000
    assert bitsk(2.0) == 0x10000
    assert bitsk(0.5) == 0x4000
    assert bitsk(2**16) == 0x7FFFFFFF

    assert bitsk(-1.0) == 0x100000000 - 0x8000
    assert bitsk(-2.0) == 0x100000000 - 0x10000
    assert bitsk(-0.5) == 0x100000000 - 0x4000
    assert bitsk(-(2**16)) == 0x80000000


def test_bitsk_unsigned():
    assert bitsk(0.0, signed=False) == 0
    assert bitsk(1.0, signed=False) == 0x8000
    assert bitsk(2.0, signed=False) == 0x10000
    assert bitsk(0.5, signed=False) == 0x4000
    assert bitsk(2**16, signed=False) == 0x80000000
    assert bitsk(2**17, signed=False) == 0xFFFFFFFF

    assert bitsk(-1.0, signed=False) == 0
    assert bitsk(-2.0, signed=False) == 0
    assert bitsk(-0.5, signed=False) == 0
    assert bitsk(-(2**16), signed=False) == 0


def test_bitsk_byte():
    kwargs = dict(n_bits=8, n_frac=3)
    assert bitsk(0.0, **kwargs) == 0
    assert bitsk(1.0, **kwargs) == 0x8
    assert bitsk(2.0, **kwargs) == 0x10
    assert bitsk(0.5, **kwargs) == 0x4
    assert bitsk(2**4, **kwargs) == 0x7F

    assert bitsk(-1.0, **kwargs) == 0x100 - 0x8
    assert bitsk(-2.0, **kwargs) == 0x100 - 0x10
    assert bitsk(-0.5, **kwargs) == 0x100 - 0x4
    assert bitsk(-(2**4), **kwargs) == 0x80


def test_bitsk_byte_unsigned():
    kwargs = dict(n_bits=8, n_frac=3, signed=False)
    assert bitsk(0.0, **kwargs) == 0
    assert bitsk(1.0, **kwargs) == 0x8
    assert bitsk(2.0, **kwargs) == 0x10
    assert bitsk(0.5, **kwargs) == 0x4
    assert bitsk(2**4, **kwargs) == 0x80
    assert bitsk(2**5, **kwargs) == 0xFF

    assert bitsk(-1.0, **kwargs) == 0
    assert bitsk(-2.0, **kwargs) == 0
    assert bitsk(-0.5, **kwargs) == 0
    assert bitsk(-(2**4), **kwargs) == 0


def test_bitsk_no_frac():
    kwargs = dict(n_bits=8, n_frac=0, signed=True)
    assert bitsk(0.0, **kwargs) == 0
    assert bitsk(1.0, **kwargs) == 0x1
    assert bitsk(2.0, **kwargs) == 0x2
    assert bitsk(2**7, **kwargs) == 0x7F
    assert bitsk(2**8, **kwargs) == 0x7F

    assert bitsk(-1.0, **kwargs) == 0x100 - 0x1
    assert bitsk(-2.0, **kwargs) == 0x100 - 0x2
    assert bitsk(-(2**7), **kwargs) == 0x80
    assert bitsk(-(2**8), **kwargs) == 0x80

    kwargs = dict(n_bits=8, n_frac=0, signed=False)
    assert bitsk(0.0, **kwargs) == 0
    assert bitsk(1.0, **kwargs) == 0x1
    assert bitsk(2.0, **kwargs) == 0x2
    assert bitsk(2**7, **kwargs) == 0x80
    assert bitsk(2**8, **kwargs) == 0xFF

    assert bitsk(-1.0, **kwargs) == 0
    assert bitsk(-2.0, **kwargs) == 0
    assert bitsk(-(2**7), **kwargs) == 0
    assert bitsk(-(2**8), **kwargs) == 0


def test_bitsk_neg_frac():
    kwargs = dict(n_bits=8, n_frac=-8, signed=True)
    assert bitsk(0.0, **kwargs) == 0
    assert bitsk(1.0, **kwargs) == 0
    assert bitsk(2.0, **kwargs) == 0
    assert bitsk(2**7, **kwargs) == 0x0
    assert bitsk(2**8, **kwargs) == 0x1
    assert bitsk(2**9, **kwargs) == 0x2
    assert bitsk(2**16, **kwargs) == 0x7F

    assert bitsk(-1.0, **kwargs) == 0
    assert bitsk(-2.0, **kwargs) == 0
    assert bitsk(-(2**7), **kwargs) == 0
    assert bitsk(-(2**8), **kwargs) == 0x100 - 0x1
    assert bitsk(-(2**8)-1, **kwargs) == 0x100 - 0x1

    kwargs = dict(n_bits=8, n_frac=-8, signed=False)
    assert bitsk(0.0, **kwargs) == 0
    assert bitsk(1.0, **kwargs) == 0
    assert bitsk(2.0, **kwargs) == 0
    assert bitsk(2**7, **kwargs) == 0
    assert bitsk(2**8, **kwargs) == 0x1
    assert bitsk(2**9, **kwargs) == 0x2

    assert bitsk(-1.0, **kwargs) == 0
    assert bitsk(-2.0, **kwargs) == 0
    assert bitsk(-(2**7), **kwargs) == 0
    assert bitsk(-(2**8), **kwargs) == 0
    assert bitsk(-(2**9), **kwargs) == 0


def test_bitsk_large_frac():
    kwargs = dict(n_bits=8, n_frac=16, signed=True)
    assert bitsk(0.0, **kwargs) == 0
    assert bitsk(2**-16, **kwargs) == 1
    assert bitsk(2**-15, **kwargs) == 2
    assert bitsk(2**-10, **kwargs) == 0x40
    assert bitsk(2**-9, **kwargs) == 0x7F
    assert bitsk(2**-8, **kwargs) == 0x7F
    assert bitsk(-(2**-10), **kwargs) == 0x100 - 0x40
    assert bitsk(-(2**-9), **kwargs) == 0x80
    assert bitsk(-(2**-17), **kwargs) == 0
    assert bitsk(-(2**-16), **kwargs) == 0x100 - 1

    kwargs = dict(n_bits=8, n_frac=16, signed=False)
    assert bitsk(0.0, **kwargs) == 0
    assert bitsk(2**-16, **kwargs) == 1
    assert bitsk(2**-15, **kwargs) == 2
    assert bitsk(2**-10, **kwargs) == 0x40
    assert bitsk(2**-9, **kwargs) == 0x80
    assert bitsk(2**-8, **kwargs) == 0xFF
    assert bitsk(-(2**-10), **kwargs) == 0
    assert bitsk(-(2**-9), **kwargs) == 0
    assert bitsk(-(2**-17), **kwargs) == 0
    assert bitsk(-(2**-16), **kwargs) == 0


def test_random():
    import random
    for i in range(10000):
        n_bits = random.randrange(1, 32)
        n_frac = random.randrange(-32, 32)
        signed = random.random() < 0.5

        fixed_value = random.getrandbits(n_bits)
        float_value = kbits(fixed_value, n_bits=n_bits, n_frac=n_frac,
                            signed=signed)
        fixed_value2 = bitsk(float_value, n_bits=n_bits, n_frac=n_frac,
                             signed=signed)
        assert fixed_value == fixed_value2
