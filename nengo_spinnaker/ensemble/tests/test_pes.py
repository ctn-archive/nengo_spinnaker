import mock
import nengo

from .. import pes
from ... import connection


class TestIntermediatePESModulatoryConnection(object):
    """Test that Intermediate representations of PES modulatory connections
    function as desired.
    """
    def test_from_connection(self):
        with nengo.Network():
            # This is a stupid network
            error = nengo.Node(lambda t, x: x, size_in=1, size_out=1)
            ens = nengo.Ensemble(100, 1)
            observer = nengo.Ensemble(100, 1)

            error_conn = nengo.Connection(error, ens, modulatory=True)
            comm_channel = nengo.Connection(ens, observer)
            comm_channel.learning_rule = nengo.PES(error_conn, 0.5)

        # Check that we can create an appropriate intermediate PES connection
        pes_instance = pes.PESInstance(
            comm_channel.learning_rule.learning_rate, observer.size_in)
        ic = pes.IntermediatePESModulatoryConnection.from_connection(
            error_conn, pes_instance=pes_instance)
        assert ic.pes_instance is pes_instance

    def test_get_reduced_incoming_connection(self):
        # Create a new PES instance which should be referenced by the port of
        # reduced incoming connection.
        pes_instance = pes.PESInstance(0.3, 5)

        # Create a sample IntermediatePESModulatoryConnection and check that it
        # correctly produces a reduced incoming connection.
        pre_obj = mock.Mock(spec_set=['size_out'])
        pre_obj.size_out = 5
        post_obj = mock.Mock(spec_set=['size_in'])
        post_obj.size_in = 10

        ic = pes.IntermediatePESModulatoryConnection(
            pre_obj, post_obj, pes_instance=pes_instance,
            synapse=nengo.Lowpass(0.05)
        )

        # Assert the modulatory connection results in an appropriate reduced
        # incoming connection.
        incoming = ic.get_reduced_incoming_connection()
        assert incoming.target.target_object is post_obj
        assert incoming.target.port is pes_instance
        assert isinstance(incoming.filter_object,
                          connection.LowpassFilterParameter)
        assert incoming.filter_object.tau == 0.05
        assert incoming.filter_object.width == 5


class TestProcessPesConnections(object):
    """Test that PES connections are rerouted and modified correctly.
    """
    def test_process_pes_connections(self):
        with nengo.Network():
            # This is a stupid network
            error = nengo.Node(lambda t, x: x, size_in=1, size_out=1)
            ens = nengo.Ensemble(100, 1)
            observer = nengo.Ensemble(100, 1)

            a = nengo.Ensemble(100, 1)
            b = nengo.Ensemble(100, 1)
            c = nengo.Connection(a, b)

            error_conn = nengo.Connection(error, ens, modulatory=True)
            comm_channel = nengo.Connection(ens, observer)
            comm_channel.learning_rule = nengo.PES(error_conn, 0.5)

        # Call process_pes_connections and ensure that the results are sensible
        objs, conns = pes.process_pes_connections(
            [error, ens, observer, a, b],
            [c, error_conn, comm_channel],
            []
        )

        # Check all objects passed through correctly
        assert len(objs) == 5
        for o in [error, ens, observer, a, b]:
            assert o in objs

        # Check the connections were correctly handled
        assert len(conns) == 3

        assert c in conns
        conns.remove(c)

        pes_instance = None
        for conn in conns:
            # Assert both PES connections reference the same PES instance
            if pes_instance is None:
                pes_instance = conn.pes_instance
            else:
                assert pes_instance is conn.pes_instance

            if isinstance(conn, pes.IntermediatePESModulatoryConnection):
                # Assert that the modulatory connection has been rerouted
                assert conn.pre_obj is error
                assert conn.post_obj is ens
            else:
                # Assert that the other connection is sensible
                assert conn.pre_obj is ens
                assert conn.post_obj is observer
