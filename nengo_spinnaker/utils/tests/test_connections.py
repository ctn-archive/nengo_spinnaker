"""Tests for Connection management utilities.
"""

import numpy as np
import pytest

import nengo
from nengo_spinnaker.utils import connections


def test_equivalent_source():
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(1, 1)
        b = nengo.Ensemble(1, 1)
        c = nengo.Ensemble(1, 1)

        c1 = nengo.Connection(a, c)
        c2 = nengo.Connection(b, c)

    with pytest.raises(AssertionError):
        connections.Connections([c1, c2])


def test_square_transforms():
    for size_in in range(1, 6):
        size_out = size_in

        model = nengo.Network()
        with model:
            a = nengo.Ensemble(1, size_in)
            b = nengo.Ensemble(1, size_out)
            c = nengo.Connection(a, b)

        tc = connections.Connections([c])

        assert(tc.width == size_out)  # Size should be same as size_out

        for t in tc.transforms_functions:
            if np.all(np.eye(size_out) == t.transform):
                break
        else:
            raise Exception("Transform not in transforms list")

        assert(c in tc)  # ID is given for connection


def test_scaled_square_transforms():
    size_in = size_out = 10
    for scale in np.random.uniform(-1., 1., 10):
        model = nengo.Network()
        with model:
            a = nengo.Ensemble(1, size_in)
            b = nengo.Ensemble(1, size_out)
            c = nengo.Connection(a, b, transform=scale)

        tc = connections.Connections([c])

        assert(tc.width == size_out)  # Size should be same as size_out

        for t in tc.transforms_functions:
            if np.all(scale*np.eye(size_out) == t.transform):
                break
        else:
            raise Exception("Transform not in transforms list")

        assert(c in tc)  # ID is given for connection


def test_rectangular_transforms():
    for size_in in range(1, 6):
        for size_out in range(1, 6):
            if size_in == size_out:
                break

            transform = np.random.uniform(-1., 1., (size_out, size_in))

            model = nengo.Network()
            with model:
                a = nengo.Ensemble(1, size_in)
                b = nengo.Ensemble(1, size_out)
                c = nengo.Connection(a, b, transform=transform)

            tc = connections.Connections([c])

            assert(tc.width == size_out)


def test_multiple_transforms():
    a_size_in = 4
    b_size_in = 5
    c_size_in = 3
    d_size_in = a_size_in

    model = nengo.Network()
    with model:
        a = nengo.Ensemble(1, a_size_in)
        b = nengo.Ensemble(1, b_size_in)
        c = nengo.Ensemble(1, c_size_in)
        d = nengo.Ensemble(1, d_size_in)

        a_b = nengo.Connection(
            a, b, transform=np.zeros((b_size_in, a_size_in)))
        a_c = nengo.Connection(
            a, c, transform=np.zeros((c_size_in, a_size_in)))
        a_d = nengo.Connection(
            a, d, transform=np.zeros((d_size_in, a_size_in)))

    tc = connections.Connections([a_b, a_c, a_d])
    assert(tc.width == b_size_in + c_size_in + d_size_in)


def test_equivalent_transforms():
    """A->C and A->D should share a transform."""
    a_size_in = 4
    b_size_in = 5
    c_size_in = a_size_in
    d_size_in = a_size_in

    model = nengo.Network()
    with model:
        a = nengo.Ensemble(1, a_size_in)
        b = nengo.Ensemble(1, b_size_in)
        c = nengo.Ensemble(1, c_size_in)
        d = nengo.Ensemble(1, d_size_in)

        a_b = nengo.Connection(
            a, b, transform=np.zeros((b_size_in, a_size_in)))
        a_c = nengo.Connection(
            a, c, transform=np.zeros((c_size_in, a_size_in)))
        a_d = nengo.Connection(
            a, d, transform=np.zeros((d_size_in, a_size_in)))

    tc = connections.Connections([a_b, a_c, a_d])
    assert(tc.width == b_size_in + c_size_in)
    assert(tc[a_c] == tc[a_d])


def test_nonequivalent_functions():
    """A->C and A->D should NOT share a transform."""
    a_size_in = 4
    b_size_in = 5
    c_size_in = a_size_in
    d_size_in = a_size_in

    model = nengo.Network()
    with model:
        a = nengo.Ensemble(1, a_size_in)
        b = nengo.Ensemble(1, b_size_in)
        c = nengo.Ensemble(1, c_size_in)
        d = nengo.Ensemble(1, d_size_in)

        a_b = nengo.Connection(
            a, b, transform=np.zeros((b_size_in, a_size_in)))
        a_c = nengo.Connection(
            a, c, transform=np.zeros((c_size_in, a_size_in)),
            function=lambda v: v**2)
        a_d = nengo.Connection(
            a, d, transform=np.zeros((d_size_in, a_size_in)))

    tc = connections.Connections([a_b, a_c, a_d])
    assert(tc.width == b_size_in + c_size_in + d_size_in)
    assert(tc[a_c] != tc[a_d])


def test_equivalent_solvers():
    """A->B and A->C should share a transform/function pair."""
    a_size_in = b_size_in = c_size_in = 4

    model = nengo.Network()
    with model:
        a = nengo.Ensemble(1, a_size_in)
        b = nengo.Ensemble(1, b_size_in)
        c = nengo.Ensemble(1, c_size_in)

        a_b = nengo.Connection(a, b)
        a_c = nengo.Connection(a, c)

    tc = connections.ConnectionsWithSolvers([a_b, a_c])
    assert(tc.width == a_size_in)
    assert(tc[a_b] == tc[a_c])


def test_nonequivalent_solvers():
    """A->B and A->C should NOT share a transform/function pair."""
    a_size_in = b_size_in = c_size_in = 4

    model = nengo.Network()
    with model:
        a = nengo.Ensemble(1, a_size_in)
        b = nengo.Ensemble(1, b_size_in)
        c = nengo.Ensemble(1, c_size_in)

        a_b = nengo.Connection(a, b, solver=nengo.decoders.Solver)
        a_c = nengo.Connection(a, c, solver=nengo.decoders.LstsqNoise)

    tc = connections.ConnectionsWithSolvers([a_b, a_c])
    assert(tc.width == b_size_in + c_size_in)
    assert(tc[a_b] != tc[a_c])


def test_equivalent_eval_points():
    """A->B and A->C should share a transform/function pair."""
    a_size_in = b_size_in = c_size_in = 4

    model = nengo.Network()
    with model:
        a = nengo.Ensemble(1, a_size_in)
        b = nengo.Ensemble(1, b_size_in)
        c = nengo.Ensemble(1, c_size_in)

        a_b = nengo.Connection(a, b)
        a_c = nengo.Connection(a, c)

    tc = connections.ConnectionsWithSolvers([a_b, a_c])
    assert(tc.width == a_size_in)
    assert(tc[a_b] == tc[a_c])


def test_nonequivalent_eval_points():
    """A->B and A->C should NOT share a transform/function pair."""
    a_size_in = b_size_in = c_size_in = 4

    model = nengo.Network()
    with model:
        a = nengo.Ensemble(1, a_size_in)
        b = nengo.Ensemble(1, b_size_in)
        c = nengo.Ensemble(1, c_size_in)

        a_b = nengo.Connection(a, b, eval_points=np.zeros(100))
        a_c = nengo.Connection(a, c, eval_points=np.array([1]*100))

    tc = connections.ConnectionsWithSolvers([a_b, a_c])
    assert(tc.width == b_size_in + c_size_in)
    assert(tc[a_b] != tc[a_c])


def test_connection_offset():
    a_size_in = 4
    b_size_in = 5
    c_size_in = 6

    model = nengo.Network()
    with model:
        e = nengo.Ensemble(1, 1, label="Source")
        a = nengo.Ensemble(1, a_size_in, label="A")
        b = nengo.Ensemble(1, b_size_in, label="B")
        c = nengo.Ensemble(1, c_size_in, label="C")

        c1 = nengo.Connection(
            e, a, transform=np.random.uniform(-1, 1, (a_size_in, 1)))
        c2 = nengo.Connection(
            e, b, transform=np.random.uniform(-1, 1, (b_size_in, 1)))
        c3 = nengo.Connection(
            e, c, transform=np.random.uniform(-1, 1, (c_size_in, 1)))

    tc = connections.ConnectionsWithSolvers()
    tc.add_connection(c1)
    tc.add_connection(c2)
    tc.add_connection(c3)

    assert(tc.width == a_size_in + b_size_in + c_size_in)
    assert(tc.get_connection_offset(c1) == 0)
    assert(tc.get_connection_offset(c2) == a_size_in)
    assert(tc.get_connection_offset(c3) == a_size_in + b_size_in)


def test_connection_offset_with_sharing():
    a_size_in = 4
    b_size_in = 5
    c_size_in = 5
    d_size_in = 6

    model = nengo.Network()
    with model:
        e = nengo.Ensemble(1, 1, label="Source")
        a = nengo.Ensemble(1, a_size_in, label="A")
        b = nengo.Ensemble(1, b_size_in, label="B")
        c = nengo.Ensemble(1, c_size_in, label="C")
        d = nengo.Ensemble(1, d_size_in, label="C")

        c1 = nengo.Connection(
            e, a, transform=np.random.uniform(-1, 1, (a_size_in, 1)))
        c2 = nengo.Connection(
            e, b, transform=np.zeros((b_size_in, 1)))
        c3 = nengo.Connection(
            e, c, transform=np.zeros((c_size_in, 1)))
        c4 = nengo.Connection(
            e, d, transform=np.random.uniform(-1, 1, (d_size_in, 1)))

    tc = connections.ConnectionsWithSolvers()
    tc.add_connection(c1)
    tc.add_connection(c2)
    tc.add_connection(c3)
    tc.add_connection(c4)

    assert(tc.width == a_size_in + b_size_in + d_size_in)
    assert(tc[c2] == tc[c3])
    assert(tc.get_connection_offset(c1) == 0)
    assert(tc.get_connection_offset(c2) == a_size_in)
    assert(tc.get_connection_offset(c3) == a_size_in)
    assert(tc.get_connection_offset(c4) == a_size_in + b_size_in)


def test_connection_banks():
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(1, 5)
        b = nengo.Ensemble(1, 5)
        c = nengo.Ensemble(1, 5)

        c1 = nengo.Connection(a, c)
        c2 = nengo.Connection(b, c)
        c3 = nengo.Connection(a, b)

    tc = connections.ConnectionBank([c1, c2])

    assert(tc[c1] != tc[c2])

    with pytest.raises(KeyError):
        tc[c3]


def test_connection_banks_offset():
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(1, 5)
        b = nengo.Ensemble(1, 4)
        c = nengo.Ensemble(1, 5)
        d = nengo.Ensemble(1, 6)
        e = nengo.Ensemble(1, 5)

        c1 = nengo.Connection(a, c)
        c2 = nengo.Connection(a, d, transform=np.zeros((6, 5)))
        c3 = nengo.Connection(b, c, transform=np.zeros((5, 4)))
        c4 = nengo.Connection(b, d, transform=np.zeros((6, 4)))
        c5 = nengo.Connection(a, e)
        c6 = nengo.Connection(a, b, transform=np.zeros((4, 5)))

    tc = connections.ConnectionBank([c1, c2, c3, c4, c5])
    assert(tc.width == 5 + 6 + 5 + 6)
    assert(tc.get_connection_offset(c1) != tc.get_connection_offset(c2))
    assert(tc.get_connection_offset(c2) != tc.get_connection_offset(c3))
    assert(tc.get_connection_offset(c3) != tc.get_connection_offset(c4))
    assert(tc.get_connection_offset(c1) == tc.get_connection_offset(c5))

    with pytest.raises(KeyError):
        tc.get_connection_offset(c6)

def test_iter_connection_back():
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(1, 5)
        b = nengo.Ensemble(1, 4)
        c = nengo.Ensemble(1, 5)
        d = nengo.Ensemble(1, 6)
        e = nengo.Ensemble(1, 5)

        c1 = nengo.Connection(a, c)
        c2 = nengo.Connection(a, d, transform=np.zeros((6, 5)))
        c3 = nengo.Connection(b, c, transform=np.zeros((5, 4)))
        c4 = nengo.Connection(b, d, transform=np.zeros((6, 4)))
        c5 = nengo.Connection(a, e)
        c6 = nengo.Connection(a, b, transform=np.zeros((4, 5)))

    tc = connections.ConnectionBank([c1, c2, c3, c4, c5])

    seen = []
    for c in tc:
        seen.append(c)

    for c in [c1, c2, c3, c4, c5]:
        assert(c in seen)
    for c in seen:
        assert(c in [c1, c2, c3, c4, c5])

def test_contains_equivalent_connection():
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(1, 2)
        b = nengo.Ensemble(1, 3)
        c = nengo.Ensemble(1, 3)
        d = nengo.Ensemble(1, 3)
        e = nengo.Ensemble(1, 2)

        c1 = nengo.Connection(a, b, transform=[[1, 0], [0, 0], [0, 1]])
        c2 = nengo.Connection(a, c, transform=[[1, 0], [0, 0], [0, 1]])
        c3 = nengo.Connection(a, d, transform=[[0, 0], [0, 0], [0, 0]])
        c4 = nengo.Connection(e, b, transform=[[1, 0], [0, 0], [0, 1]])

    cs = connections.Connections([c1])
    assert(cs.contains_compatible_connection(c2))
    assert(not cs.contains_compatible_connection(c3))
    assert(not cs.contains_compatible_connection(c4))


def test_bank_contains_equivalent_connection():
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(1, 2)
        b = nengo.Ensemble(1, 2)
        c = nengo.Ensemble(1, 3)
        d = nengo.Ensemble(1, 3)

        c1 = nengo.Connection(a, c, transform=[[1, 0], [0, 0], [0, 1]])
        c2 = nengo.Connection(a, d, transform=[[0, 0], [0, 0], [0, 0]])
        c3 = nengo.Connection(a, d, transform=[[1, 0], [0, 0], [0, 1]])
        c4 = nengo.Connection(a, d, transform=[[1, 0], [0, 0], [0, 1]],
                              function=lambda v: v**2)
        c5 = nengo.Connection(b, d, transform=[[1, 0], [0, 0], [0, 1]])

    cs = connections.ConnectionBank([c1])
    assert(not cs.contains_compatible_connection(c2))
    assert(cs.contains_compatible_connection(c3))
    assert(not cs.contains_compatible_connection(c4))
    assert(not cs.contains_compatible_connection(c5))
