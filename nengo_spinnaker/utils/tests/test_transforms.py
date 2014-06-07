"""Tests for transform expansion utility module.
"""

import numpy as np

import nengo
from nengo_spinnaker.utils import transforms


def test_square_transforms():
    for size_in in range(1, 6):
        size_out = size_in

        model = nengo.Network()
        with model:
            a = nengo.Ensemble(1, size_in)
            b = nengo.Ensemble(1, size_out)
            c = nengo.Connection(a, b)

        tc = transforms.get_transforms([c])

        assert(tc.width == size_out)  # Size should be same as size_out

        for t in tc.transforms_functions:
            if np.all(np.eye(size_out) == t.transform):
                break
        else:
            raise Exception("Transform not in transforms list")

        assert(c in tc.connection_ids)  # ID is given for connection


def test_scaled_square_transforms():
    size_in = size_out = 10
    for scale in np.random.uniform(-1., 1., 10):
        model = nengo.Network()
        with model:
            a = nengo.Ensemble(1, size_in)
            b = nengo.Ensemble(1, size_out)
            c = nengo.Connection(a, b, transform=scale)

        tc = transforms.get_transforms([c])

        assert(tc.width == size_out)  # Size should be same as size_out

        for t in tc.transforms_functions:
            if np.all(scale*np.eye(size_out) == t.transform):
                break
        else:
            raise Exception("Transform not in transforms list")

        assert(c in tc.connection_ids)  # ID is given for connection


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

            tc = transforms.get_transforms([c])

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

    tc = transforms.get_transforms([a_b, a_c, a_d])
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

    tc = transforms.get_transforms([a_b, a_c, a_d])
    assert(tc.width == b_size_in + c_size_in)
    assert(tc.connection_ids[a_c] == tc.connection_ids[a_d])


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

    tc = transforms.get_transforms([a_b, a_c, a_d])
    assert(tc.width == b_size_in + c_size_in + d_size_in)
    assert(tc.connection_ids[a_c] != tc.connection_ids[a_d])


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

    tc = transforms.get_transforms_with_solvers([a_b, a_c])
    assert(tc.width == a_size_in)
    assert(tc.connection_ids[a_b] == tc.connection_ids[a_c])


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

    tc = transforms.get_transforms_with_solvers([a_b, a_c])
    assert(tc.width == b_size_in + c_size_in)
    assert(tc.connection_ids[a_b] != tc.connection_ids[a_c])
