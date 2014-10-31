import mock
import pytest
import nengo
import numpy as np

from ..intermediate import IntermediateConnection
from ..reduced import (
    OutgoingReducedConnection, LowpassFilterParameter,
    OutgoingReducedEnsembleConnection, StandardInputPort,
)


class TestIntermediateConnection(object):
    def test_from_connection(self):
        # Create the objects to test with
        with nengo.Network():
            a = nengo.Ensemble(100, 3)
            b = nengo.Ensemble(100, 3)

        ks = mock.Mock()

        # Check that the function works at all
        with nengo.Network():
            c = nengo.Connection(a, b)

        ic = IntermediateConnection.from_connection(c, keyspace=ks,
                                                    is_accumulatory=False)
        assert ic.pre_obj is a
        assert ic.post_obj is b
        assert ic.keyspace is ks
        assert not ic.is_accumulatory
        assert np.all(ic.transform == np.eye(3))
        assert ic.width == 3

        # Check that the function works when we use slicing
        with nengo.Network():
            c = nengo.Connection(a[1], b[2])

        ic = IntermediateConnection.from_connection(c)

        assert ic.pre_obj is a
        assert ic.post_obj is b
        assert np.all(ic.transform == [[0, 0, 0], [0, 0, 0], [0, 1., 0]])
        assert ic.width == 3

        # If we try to get an intermediate object for something that isn't a
        # Nengo connection we should fail.
        c = mock.Mock()
        with pytest.raises(NotImplementedError) as excinfo:
            IntermediateConnection.from_connection(c)
            assert mock.Mock.__name__ in str(excinfo.value)

    def test_get_reduced_outgoing_connection_obj(self):
        # Create the objects to test with
        with nengo.Network():
            a = nengo.Node(3)
            b = nengo.Ensemble(100, 3)
            c = nengo.Connection(a, b[1], function=lambda x: x**2)

        ks = mock.Mock()
        ic = IntermediateConnection.from_connection(c, keyspace=ks)

        # Get the reduced outgoing connection
        rc = ic.get_reduced_outgoing_connection()
        assert isinstance(rc, OutgoingReducedConnection)
        assert rc.width == ic.width == 3
        assert np.all(rc.transform == [[0], [1], [0]])
        assert rc.function is c.function
        assert rc.keyspace is ks

    def test_get_reduced_outgoing_connection_ens(self):
        # Create the objects to test with
        with nengo.Network():
            a = nengo.Ensemble(100, 3)
            a.eval_points = np.random.uniform(size=(100, 3))
            b = nengo.Ensemble(100, 3)
            c = nengo.Connection(a[0], b[1], function=lambda x: x**2)

        ks = mock.Mock()
        ic = IntermediateConnection.from_connection(c, keyspace=ks)
        ic._expected_ensemble_type = nengo.Ensemble

        # Get the reduced outgoing connection
        rc = ic.get_reduced_outgoing_connection()
        assert isinstance(rc, OutgoingReducedEnsembleConnection)
        assert rc.width == ic.width == 3
        assert np.all(rc.transform == [[0], [1], [0]])
        assert rc.function is c.function
        assert rc.keyspace is ks
        assert np.all(rc.eval_points == a.eval_points)

    def test_get_reduced_outgoing_connection_advanced(self):
        # Create the objects to test with
        with nengo.Network():
            a = nengo.Ensemble(100, 3)
            b = nengo.Ensemble(100, 3)
            c = nengo.Connection(a[0], b[1], function=lambda x: x**2,
                                 synapse=nengo.Lowpass(.3))

        ks = mock.Mock()

        ic = IntermediateConnection.from_connection(c, keyspace=ks)

        placeholder = mock.Mock(spec_set=['eval_points', 'size_out'])
        placeholder.eval_points = np.random.uniform(size=(100, 3))
        placeholder.size_out = 3

        ic._expected_ensemble_type = placeholder.__class__
        ic.pre_obj = placeholder

        # Get the reduced outgoing connection
        rc = ic.get_reduced_outgoing_connection()
        assert isinstance(rc, OutgoingReducedEnsembleConnection)
        assert rc.width == ic.width == 3
        assert np.all(rc.transform == [[0], [1], [0]])
        assert rc.function is c.function
        assert rc.keyspace is ks

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
        assert isinstance(f, LowpassFilterParameter)
        assert f.tau == synapse.tau

    def test_get_incoming_reduced_connection(self):
        with nengo.Network():
            a = nengo.Ensemble(100, 3)
            b = nengo.Ensemble(100, 3)
            c = nengo.Connection(a[0], b[1], function=lambda x: x**2,
                                 synapse=nengo.Lowpass(.3))

        # Accumulatory
        ic = IntermediateConnection.from_connection(c).\
            get_reduced_incoming_connection()
        assert ic.target.target_object is b
        assert ic.target.port is StandardInputPort
        assert ic.filter_object.tau == .3
        assert ic.filter_object.is_accumulatory
        assert isinstance(ic.filter_object, LowpassFilterParameter)

        # Not accumulatory
        ic = IntermediateConnection.from_connection(c, is_accumulatory=False).\
            get_reduced_incoming_connection()
        assert ic.target.target_object is b
        assert ic.target.port is StandardInputPort
        assert ic.filter_object.tau == .3
        assert not ic.filter_object.is_accumulatory
        assert isinstance(ic.filter_object, LowpassFilterParameter)
