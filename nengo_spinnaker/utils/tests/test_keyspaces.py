import pytest
import random

from nengo_spinnaker import utils


def test_keyspace_init_size():
    # Assert that we can't create a new keyspace with more than 32 bits
    with pytest.raises(ValueError):
        class KeyTest(utils.keyspaces.KeySpace):
            fields = [('x', 31), ('y', 2)]
            routing_fields = ['x', 'y']


def test_keyspace_masks():
    # Create a new keyspace and check that appropriate masks can be retrieved
    # from it.
    class Nengo88556(utils.keyspaces.KeySpace):
        fields = [('x', 8), ('y', 8), ('p', 5), ('i', 5), ('d', 6)]
        routing_fields = ['x', 'y', 'p', 'i']

    # Assert that we get the correct masks out
    ks = Nengo88556()
    assert ks.mask_x == 0xff000000
    assert ks.mask_y == 0x00ff0000
    assert ks.mask_p == 0x0000f800
    assert ks.mask_i == 0x000007c0
    assert ks.mask_d == 0x0000003f

    # Assert that we can't change a mask
    with pytest.raises(AttributeError):
        ks.mask_x = 0

    assert ks.routing_mask == 0xffffffc0


def test_keyspace_key():
    # Test generation of a single key and routing key
    ks = utils.keyspaces.nengo_default()
    for _ in range(1000):
        x = random.randint(0, 31)
        assert(ks.routing_key(x=x) == ks.key(x=x) == (x << 24))

        y = random.randint(0, 31)
        assert(ks.routing_key(x=x, y=y) == ks.key(x=x, y=y) ==
               (x << 24) | (y << 16))

        p = random.randint(0, 17)
        assert(ks.routing_key(x=x, y=y, p=p) == ks.key(x=x, y=y, p=p) ==
               (x << 24) | (y << 16) | (p << 11))

        i = random.randint(0, 31)
        assert(ks.routing_key(x=x, y=y, p=p, i=i) ==
               ks.key(x=x, y=y, p=p, i=i) ==
               (x << 24) | (y << 16) | (p << 11) | (i << 6))

        d = random.randint(0, 63)
        assert(ks.key(x=x, y=y, p=p, i=i, d=d) ==
               (x << 24) | (y << 16) | (p << 11) | (i << 6) | d)
        assert(ks.routing_key(x=x, y=y, p=p, i=i, d=d) ==
               ks.key(x=x, y=y, p=p, i=i))  # d is not in the routing key


def test_keyspace_equivalence():
    # Check that equivalent spaces with no values are equivalent
    ks1 = utils.keyspaces.nengo_default()
    ks2 = utils.keyspaces.create_keyspace(
        'KS2', [('x', 8), ('y', 8), ('p', 5), ('i', 5), ('d', 6)],
        "xypi"
    )()

    assert ks1 == ks2

    # Check that non-equivalent field allocations are not equivalent
    ks3 = utils.keyspaces.create_keyspace(
        'KS3', [('x', 8), ('y', 8), ('p', 5), ('i', 7), ('d', 4)],
        "xypi")()

    assert ks1 != ks3

    # Check that equivalent field allocations but non-equivalent routing keys
    # are not equivalent
    ks4 = utils.keyspaces.create_keyspace(
        'KS4', [('x', 8), ('y', 8), ('p', 5), ('i', 5), ('d', 6)],
        "xypd"
    )()
    assert ks1 != ks4
