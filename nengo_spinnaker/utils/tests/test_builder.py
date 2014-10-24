import mock
import nengo
import pytest

from .. import builder as builder_utils
from ...connection import IntermediateConnection


@pytest.fixture(scope='function')
def base_model():
    model = nengo.Network()
    with model:
        a = nengo.Node(lambda t: t, size_in=0, size_out=1)
        b = nengo.Node(None, size_in=1, size_out=1)
        c = nengo.Node(lambda t, x: None, size_in=1, size_out=0)
    return model, (a, b, c)


def test_create_replacement_connection_assertion_fail_connectivity(base_model):
    model, (a, b, c) = base_model
    with model:
        c1 = nengo.Connection(a, b)
        c2 = nengo.Connection(a, b)

    with pytest.raises(AssertionError):
        builder_utils.create_replacement_connection(c1, c2)


def test_create_replacement_connection_assertion_fail_types(base_model):
    model, (a, b, c) = base_model
    with model:
        c1 = nengo.Connection(a, b)
        c2 = nengo.Connection(b, c)
    b.output = lambda t, x: x

    with pytest.raises(AssertionError):
        builder_utils.create_replacement_connection(c1, c2)


def test_create_replacement_connection_fail_synapse(base_model):
    model, (a, b, c) = base_model
    with model:
        c1 = nengo.Connection(a, b, synapse=0.01)
        c2 = nengo.Connection(b, c, synapse=0.02)

    with pytest.raises(NotImplementedError):
        builder_utils.create_replacement_connection(c1, c2)


def test_create_replacement_connection_synapse_select_1(base_model):
    model, (a, b, c) = base_model
    with model:
        c1 = nengo.Connection(a, b, synapse=0.01)
        c2 = nengo.Connection(b, c, synapse=None)

    c3 = builder_utils.create_replacement_connection(c1, c2)
    assert c3.synapse is c1.synapse


def test_create_replacement_connection_synapse_select_2(base_model):
    model, (a, b, c) = base_model
    with model:
        c1 = nengo.Connection(a, b, synapse=None)
        c2 = nengo.Connection(b, c, synapse=0.01)

    c3 = builder_utils.create_replacement_connection(c1, c2)
    assert c3.synapse is c2.synapse


def test_create_replacement_connection_fail_function(base_model):
    model, (a, b, c) = base_model
    with model:
        c1 = nengo.Connection(a, b, function=lambda x: x**2, synapse=None)
        try:
            c2 = nengo.Connection(b, c, function=lambda x: x**2)
        except ValueError:
            # Nengo does not allow functions on connections from pass nodes.
            c2 = IntermediateConnection(b, c, function=lambda x: x**2,
                                        transform=1.)

    with pytest.raises(Exception):  # TODO This should be more specific!
        builder_utils.create_replacement_connection(c1, c2)


def test_create_replacement_connection_no_connection(base_model):
    model, (a, b, c) = base_model
    with model:
        c1 = nengo.Connection(a, b, transform=0., synapse=None)
        c2 = nengo.Connection(b, c, transform=0.)

    assert builder_utils.create_replacement_connection(c1, c2) is None


def test_create_replacement_connection_ks_combine_1(base_model):
    # Assert keyspace selected from first connection
    model, (a, b, c) = base_model
    with model:
        c1 = nengo.Connection(a, b, synapse=None)
        c2 = nengo.Connection(b, c)

    setattr(c1, 'keyspace', mock.Mock())
    c3 = builder_utils.create_replacement_connection(c1, c2)
    assert c3.keyspace is c1.keyspace


def test_create_replacement_connection_ks_combine_2(base_model):
    # Assert keyspace selected from second connection
    model, (a, b, c) = base_model
    with model:
        c1 = nengo.Connection(a, b, synapse=None)
        c2 = nengo.Connection(b, c)

    setattr(c2, 'keyspace', mock.Mock())
    c3 = builder_utils.create_replacement_connection(c1, c2)
    assert c3.keyspace is c2.keyspace


def test_create_replacement_connection_ks_combine_fail(base_model):
    # Assert keyspace collisions fail!
    model, (a, b, c) = base_model
    with model:
        c1 = nengo.Connection(a, b, synapse=None)
        c2 = nengo.Connection(b, c)

    setattr(c1, 'keyspace', mock.Mock())
    setattr(c2, 'keyspace', mock.Mock())

    with pytest.raises(NotImplementedError):
        builder_utils.create_replacement_connection(c1, c2)


def test_create_replacement_connection_type_combine_1(base_model):
    class FalseConnectionTypeA(IntermediateConnection):
        pass

    # Assert type selected from first connection
    model, (a, b, c) = base_model
    with model:
        c1 = nengo.Connection(a, b, synapse=None)
        c2 = nengo.Connection(b, c)

    c1 = FalseConnectionTypeA.from_connection(c1)
    c2 = IntermediateConnection.from_connection(c2)
    c3 = builder_utils.create_replacement_connection(c1, c2)
    assert c3.__class__ is FalseConnectionTypeA


def test_create_replacement_connection_type_combine_2(base_model):
    class FalseConnectionTypeA(IntermediateConnection):
        pass

    # Assert type selected from second connection
    model, (a, b, c) = base_model
    with model:
        c1 = nengo.Connection(a, b, synapse=None)
        c2 = nengo.Connection(b, c)

    c1 = IntermediateConnection.from_connection(c1)
    c2 = FalseConnectionTypeA.from_connection(c2)
    c3 = builder_utils.create_replacement_connection(c1, c2)
    assert c3.__class__ is FalseConnectionTypeA


def test_create_replacement_connection_type_combine_fail(base_model):
    class FalseConnectionTypeA(IntermediateConnection):
        pass

    class FalseConnectionTypeB(IntermediateConnection):
        pass

    # Assert types collide and can't be merged
    model, (a, b, c) = base_model
    with model:
        c1 = nengo.Connection(a, b, synapse=None)
        c2 = nengo.Connection(b, c)

    c1 = FalseConnectionTypeA.from_connection(c1)
    c2 = FalseConnectionTypeB.from_connection(c2)

    with pytest.raises(NotImplementedError):
        builder_utils.create_replacement_connection(c1, c2)
