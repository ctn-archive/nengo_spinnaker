"""Tests for utilities for constructing global inhibition connections.
"""
import nengo
import mock
import numpy as np
import pytest

from nengo_spinnaker import utils


def test_create_global_inhibition_connection():
    """Should take a connection which is globally inhibitive and return a
    reduced representation.
    """
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(100, 1)
        c = nengo.Ensemble(100, 2)

        gate = nengo.Connection(a, b.neurons, transform=[[-1]]*100)
        not_gate = nengo.Connection(a, b)
        not_gate2 = nengo.Connection(
            a, b.neurons, transform=
            np.random.uniform(-1, 1, (100, 1)))
        valid_gate3 = nengo.Connection(c, b.neurons, transform=[[-1, -1]]*100)

    with pytest.raises(AssertionError):
        utils.global_inhibition.create_inhibition_connection(not_gate)

    with pytest.raises(ValueError):
        utils.global_inhibition.create_inhibition_connection(not_gate2)

    conn = utils.global_inhibition.create_inhibition_connection(gate)
    assert(conn.transform == -1)
    assert(conn.pre == a)

    utils.global_inhibition.create_inhibition_connection(valid_gate3)


def test_create_global_inhibition_edge():
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(256, 1)
        c = nengo.Ensemble(100, 1)

        gate = nengo.Connection(a, c.neurons, transform=[[-1]]*100,
                                synapse=0.05)
        c1 = nengo.Connection(b, c)

    prevertex = mock.Mock()
    postvertex = mock.Mock()
    e = mock.PropertyMock(return_value=None)
    type(postvertex).inhibitory_edge = e

    with pytest.raises(AssertionError):
        utils.global_inhibition.create_inhibition_edge(
            c1, prevertex, postvertex)

    with pytest.raises(NotImplementedError):
        e = mock.PropertyMock(return_value=mock.Mock())
        type(postvertex).inhibitory_edge = e
        utils.global_inhibition.create_inhibition_edge(
            gate, prevertex, postvertex)

    e = mock.PropertyMock(return_value=None)
    type(postvertex).inhibitory_edge = e

    edge = utils.global_inhibition.create_inhibition_edge(
        gate, prevertex, postvertex)

    assert(edge.width == 1)
    assert(edge.synapse == gate.synapse)
    assert(edge.transform == -1)
    assert(edge.function == gate.function)

    # Check that the inhibitory edge of the post vertex was checked and then
    # set
    e.assert_any_call()
    e.assert_any_call(edge)


def test_full_transform_connection():
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(100, 1)

        c = nengo.Connection(a, b.neurons, transform=[[-1]]*100)


    c_ib = utils.global_inhibition.create_inhibition_connection(c)

    from nengo.utils.builder import full_transform
    full_transform(c_ib)
