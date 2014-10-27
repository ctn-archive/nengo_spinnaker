import copy
import mock
import nengo
import numpy as np
import pytest

from .. import connection

from ..connection import (
    IntermediateConnection, OutgoingReducedConnection,
    OutgoingReducedEnsembleConnection, IncomingReducedConnection
)


def comparison_ids(items):
    return ((i, j) for i in range(len(items)) for j in range(i+1, len(items)))


# Tests for the connection builder
class TestGenericConnectionBuilder(object):
    @pytest.fixture
    def network(self):
        model = nengo.Network()
        with model:
            a = nengo.Ensemble(100, 1)
            b = nengo.Ensemble(100, 1)

            c = nengo.Connection(a, b)

        c1 = IntermediateConnection.from_connection(c)
        return c1, c, a, b

    def test_build(self, network):
        # Create a Nengo connection
        c1, c, a, b = network
        c1.keyspace = mock.Mock()

        # Create a mock assembler
        # This assembler will return different objects for pre and post
        assembler = mock.Mock()
        prev = mock.Mock()
        postv = mock.Mock()
        assembler.get_object_vertex.side_effect = \
            lambda obj: prev if obj is a else postv

        # Call the builder function
        ne = connection.generic_connection_builder(c1, assembler)

        # Assert that the correct methods were called on the assembler
        assembler.get_object_vertex.assert_any_call(a)
        assembler.get_object_vertex.assert_any_call(b)

        # Assert that the correct pre-vertex and post-vertex objects are
        # present
        assert ne.pre_vertex is prev
        assert ne.post_vertex is postv
        assert ne.keyspace is c1.keyspace

    def test_no_keyspace(self, network):
        # Create a Nengo connection
        c1, c, a, b = network

        # Create a mock assembler
        # This assembler will return different objects for pre and post
        assembler = mock.Mock()
        prev = mock.Mock()
        postv = mock.Mock()
        assembler.get_object_vertex.side_effect = \
            lambda obj: prev if obj is a else postv

        # Call the builder function
        with pytest.raises(AssertionError):
            connection.generic_connection_builder(c1, assembler)

    def test_no_prevertex(self, network):
        # Create a Nengo connection
        c1, c, a, b = network

        # Create a mock assembler
        # This assembler will return different objects for pre and post
        assembler = mock.Mock()
        prev = None
        postv = mock.Mock()
        assembler.get_object_vertex.side_effect = \
            lambda obj: prev if obj is a else postv

        # Call the builder function
        ne = connection.generic_connection_builder(c1, assembler)
        assert ne is None

    def test_no_postvertex(self, network):
        # Create a Nengo connection
        c1, c, a, b = network

        # Create a mock assembler
        # This assembler will return different objects for pre and post
        assembler = mock.Mock()
        prev = mock.Mock()
        postv = None
        assembler.get_object_vertex.side_effect = \
            lambda obj: prev if obj is a else postv

        # Call the builder function
        ne = connection.generic_connection_builder(c1, assembler)
        assert ne is None


class TestIntermediateConnection(object):
    def test_get_filter_fail(self):
        ic = IntermediateConnection(mock.Mock(), mock.Mock(), mock.Mock())
        with pytest.raises(NotImplementedError) as excinfo:
            ic._get_filter()
            assert excinfo is mock.Mock

    def test_get_filter(self):
        pre_obj = mock.Mock(spec_set=[])
        post_obj = mock.Mock(spec_set=['size_in'])
        post_obj.size_in = 5

        synapse = nengo.Lowpass(0.03)

        ic = IntermediateConnection(pre_obj, post_obj, synapse)

        f = ic._get_filter()
        assert isinstance(f, connection.LowpassFilterParameter)
        assert f.tau == synapse.tau
        assert f.width == post_obj.size_in

    def test_required_transform_shape(self):
        # TODO Remove the method this relates to?
        pre_obj = mock.Mock(spec_set=['size_out'])
        pre_obj.size_out = 5
        post_obj = mock.Mock(spec_set=['size_in'])
        post_obj.size_in = 4

        ic = IntermediateConnection(pre_obj, post_obj)
        assert ic._required_transform_shape() == (5, 4)


class TestOutgoingReducedConnection(object):
    def test_eq(self):
        # Create some reduced connections and test for equivalence
        rcs = [
            OutgoingReducedConnection(np.eye(3), None),
            OutgoingReducedConnection(np.eye(3), None),
            OutgoingReducedConnection(np.eye(3), lambda x: x**2),
            OutgoingReducedConnection(np.eye(1), None),
            OutgoingReducedConnection(np.eye(3), None, mock.Mock()),
            OutgoingReducedEnsembleConnection(np.eye(3), None),
        ]

        # Define what the equivalences should be
        eqs = [True] + [False]*(len(rcs) - 1)

        # Define whether the hashes should be equivalent
        hashes = eqs[:]

        # Get the set of indices for equivalence
        i_s = ((i, j) for i in range(len(rcs)) for j in range(i+1, len(rcs)))
        for (i, j), eq, he in zip(i_s, eqs, hashes):
            assert (rcs[i] == rcs[j]) is eq, (i, j)
            assert (hash(rcs[i]) == hash(rcs[j])) is he, (i, j)

    def test_copy(self):
        # Ensure that when we copy we get copies of the transform but
        # references to everything else.
        c = connection.OutgoingReducedConnection(
            np.eye(3), lambda x: x**2, mock.Mock())
        d = copy.copy(c)

        # Assert the transforms are equivalent but not the same object
        assert np.all(d.transform == c.transform)
        assert c.transform is not d.transform, (id(c.transform),
                                                id(d.transform))

        # Assert that the function and keyspace are equivalent
        assert c.function is d.function
        assert c.keyspace is d.keyspace

    def test_copy_with_transform(self):
        # Ensure that we can copy a reduced connection but modify the transform
        # in the process.
        c = connection.OutgoingReducedConnection(np.eye(3), None)
        d = c.copy_with_transform(3.)

        assert np.all(d.transform == c.transform*3.)
        assert c.function is d.function
        assert c.keyspace is d.keyspace


class TestOutgoingReducedEnsembleConnection(object):
    def test_eq_no_func_equiv(self):
        # Create some reduced connections and test for equivalence without
        # considering function equivalence.
        eval_points = np.linspace(-1., 1.)
        rcs = [
            OutgoingReducedEnsembleConnection(np.eye(3), None, None,
                                              eval_points),
            OutgoingReducedEnsembleConnection(np.eye(3), None, None,
                                              eval_points),
            OutgoingReducedEnsembleConnection(np.eye(3), None, None, None),
            OutgoingReducedEnsembleConnection(np.eye(3), None, None,
                                              eval_points, mock.Mock()),
            OutgoingReducedConnection(np.eye(3), None, None),
        ]

        # Define what the equivalences should be
        eqs = [True] + [False]*(len(rcs) - 1)

        # Define whether the hashes should be equivalent
        hashes = eqs[:]

        # Get the set of indices for equivalence
        i_s = ((i, j) for i in range(len(rcs)) for j in range(i+1, len(rcs)))
        for (i, j), eq, he in zip(i_s, eqs, hashes):
            assert (rcs[i] == rcs[j]) is eq, (i, j)
            assert (hash(rcs[i]) == hash(rcs[j])) is he, (i, j)

    def test_eq(self):
        # Create some reduced connections and test for equivalence
        eval_points = np.linspace(-1., 1.)
        rcs = [
            OutgoingReducedEnsembleConnection(1., lambda x: x**2, None,
                                              eval_points),
            OutgoingReducedEnsembleConnection(1., lambda x: x**2, None,
                                              eval_points),
            OutgoingReducedEnsembleConnection(1., lambda x: x**3, None,
                                              eval_points),
        ]

        # Define what the equivalences should be
        eqs = [True] + [False]*(len(rcs) - 1)

        # Define whether the hashes should be equivalent
        hashes = eqs[:]

        # Get the set of indices for equivalence
        i_s = ((i, j) for i in range(len(rcs)) for j in range(i+1, len(rcs)))
        for (i, j), eq, he in zip(i_s, eqs, hashes):
            assert (rcs[i] == rcs[j]) is eq, (i, j)
            assert (hash(rcs[i]) == hash(rcs[j])) is he, (i, j)

    def test_eq_learning_rules(self):
        # Create some reduced connections and test for equivalence when
        # learning rules are present.
        lrule = mock.Mock()
        rcs = [
            OutgoingReducedEnsembleConnection(
                1., None, transmitter_learning_rule=lrule),
            OutgoingReducedEnsembleConnection(
                1., None, transmitter_learning_rule=lrule),
            OutgoingReducedEnsembleConnection(1., None)
        ]

        # Define what the equivalences should be
        eqs = [False]*(len(rcs))

        # Define whether the hashes should be equivalent
        hashes = [True] + eqs[1:]

        # Get the set of indices for equivalence
        i_s = ((i, j) for i in range(len(rcs)) for j in range(i+1, len(rcs)))
        for (i, j), eq, he in zip(i_s, eqs, hashes):
            assert (rcs[i] == rcs[j]) is eq, (i, j)
            assert (hash(rcs[i]) == hash(rcs[j])) is he, (i, j)


class TestIncomingReducedConnection(object):
    def test_eq(self):
        # Create some incoming reduced connections and test for equivalence.
        obj_a = mock.Mock()
        obj_b = mock.Mock()
        rcs = [
            IncomingReducedConnection(obj_a, None),
            IncomingReducedConnection(obj_a, None),  # Same target, same filter
            IncomingReducedConnection(obj_b, None),  # Diff target, same filter
            IncomingReducedConnection(obj_a, mock.Mock()),
            IncomingReducedConnection(obj_b, mock.Mock()),
        ]

        # Define what the equivalences should be
        eqs = [True] + [False]*(len(rcs) - 1)

        # Define whether the hashes should be equivalent
        hashes = eqs[:]

        # Get the set of indices for equivalence
        i_s = ((i, j) for i in range(len(rcs)) for j in range(i+1, len(rcs)))
        for (i, j), eq, he in zip(i_s, eqs, hashes):
            assert (rcs[i] == rcs[j]) is eq, (i, j)
            assert (hash(rcs[i]) == hash(rcs[j])) is he, (i, j)


class TestTargetObject(object):
    """Test target/port objects.
    """
    def test_target_eqs(object):
        # Check target equivalences
        o1 = mock.Mock()
        p1 = connection.StandardPorts.INPUT
        p2 = mock.Mock()
        o2 = connection.EnsemblePorts.GLOBAL_INHIBITION
        rcs = [
            connection.Target(o1, p1),
            connection.Target(o1, p1),
            connection.Target(o1, p2),
            connection.Target(o2, p1),
            connection.Target(o2, p2),
        ]

        eqs = [True] + [False]*(len(rcs) - 1)
        hashes = [True, False] + [False]*(len(rcs) - 2)

        i_s = ((i, j) for i in range(len(rcs)) for j in range(i+1, len(rcs)))
        for (i, j), eq, he in zip(i_s, eqs, hashes):
            assert (rcs[i] == rcs[j]) is eq, (i, j)
            assert (hash(rcs[i]) == hash(rcs[j])) is he, (i, j)


class TestLowpassFilterParameter(object):
    def test_eq(self):
        filters = [
            connection.LowpassFilterParameter(2, 0.03),
            connection.LowpassFilterParameter(1, 0.03),
            connection.LowpassFilterParameter(2, 0.03, is_accumulatory=False),
        ]
        i_s = comparison_ids(filters)

        for (i, j) in i_s:
            assert filters[i] != filters[j], (i, j)
            assert hash(filters[i]) != hash(filters[j]), (i, j)


class TestBuildConnectionTrees(object):
    """Test that connection trees can be built correctly.
    """
    def test_get_reduced_outgoing_connection_obj(self):
        # Test that getting reduced connections for connections between objects
        # which are not Ensembles results in the correct reduced types and
        # parameters.
        with nengo.Network():
            a = nengo.Node(lambda t: t, size_in=0, size_out=1)
            b = nengo.Node(lambda t, x: None, size_in=1, size_out=0)
            c = nengo.Connection(a, b, function=lambda x: x**2, synapse=0.01)

        # Create the intermediate connection
        ks = mock.Mock()
        ic = IntermediateConnection.from_connection(c, keyspace=ks)

        # Get the reduced connections
        outgoing = ic.get_reduced_outgoing_connection()

        # Check the outgoing component
        assert isinstance(outgoing, OutgoingReducedConnection)
        assert np.all(outgoing.transform == c.transform)
        assert outgoing.function is c.function
        assert outgoing.keyspace is ks

    def test_get_reduced_outgoing_connection_ens(self):
        # Test that getting reduced connections for connections between objects
        # which are not Ensembles results in the correct reduced types and
        # parameters.
        with nengo.Network():
            a = nengo.Ensemble(100, 1)
            b = nengo.Ensemble(100, 1)
            c = nengo.Connection(
                a, b, function=lambda x: x**2, synapse=0.01,
                eval_points=np.linspace(-1., 1.)[:, np.newaxis]
            )

        # Create the intermediate connection
        ks = mock.Mock()
        ic = IntermediateConnection.from_connection(c, keyspace=ks)

        # Get the reduced connections
        outgoing = ic.get_reduced_outgoing_connection()

        # Check the outgoing component
        assert isinstance(outgoing, OutgoingReducedEnsembleConnection)
        assert np.all(outgoing.transform == c.transform)
        assert outgoing.function is c.function
        assert outgoing.keyspace is ks
        assert outgoing.solver is c.solver
        assert np.all(outgoing.eval_points == c.eval_points)
        assert outgoing.transmitter_learning_rule is None

    def test_get_reduced_incoming_connection(self):
        with nengo.Network():
            a = nengo.Ensemble(100, 1)
            b = nengo.Ensemble(100, 1)
            c = nengo.Connection(
                a, b, function=lambda x: x**2, synapse=0.01,
                eval_points=np.linspace(-1., 1.)[:, np.newaxis],
            )

        # Create the intermediate connection
        ks = mock.Mock()
        ic = IntermediateConnection.from_connection(c, keyspace=ks)

        # Get the reduced connection
        inc = ic.get_reduced_incoming_connection()
        assert inc.origin is a
        assert inc.outgoing == ic.get_reduced_outgoing_connection()
        assert inc.incoming.target.target_object is b
        assert inc.incoming.target.port is connection.StandardPorts.INPUT
        assert isinstance(inc.incoming.filter_object,
                          connection.LowpassFilterParameter)
        assert inc.incoming.filter_object.tau == 0.01
