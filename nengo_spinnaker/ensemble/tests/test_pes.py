import mock
import nengo
import numpy as np
import pytest

from ...builder import _convert_remaining_connections
from ..intermediate import IntermediateLearningRule
from .. import pes
from ...connections.connection_tree import ConnectionTree
from ...connections.reduced import LowpassFilterParameter, StandardInputPort
from ...utils.fixpoint import bitsk, kbits


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
            comm_channel.learning_rule_type = nengo.PES(error_conn, 0.5)

        # Check that we can create an appropriate intermediate PES connection
        pes_instance = pes.PESInstance(
            comm_channel.learning_rule_type.learning_rate, observer.size_in)
        ic = pes.IntermediatePESModulatoryConnection.from_connection(
            error_conn, pes_instance=pes_instance)
        assert ic.pes_instance is pes_instance

    def test_get_reduced_connections_for_modulatory(self):
        # Create a new PES instance which should be referenced by the port of
        # reduced incoming connection.
        pes_instance = pes.PESInstance(0.3, 5)

        # Create a sample IntermediatePESModulatoryConnection and check that it
        # correctly produces a reduced incoming connection.
        pre_obj = nengo.Ensemble(100, 5, add_to_container=False)
        pre_obj.eval_points = np.random.normal(size=(1000, 5))
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
                          LowpassFilterParameter)
        assert incoming.filter_object.tau == 0.05

        # Assert the modulatory connection results in an appropriate reduced
        # outgoing connection.
        outgoing = ic.get_reduced_outgoing_connection()
        assert outgoing.width == 5


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
            comm_channel.learning_rule_type = nengo.PES(error_conn, 0.5)

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


@pytest.fixture(scope='function')
def sample_ctree_with_pes():
    # Create a sample network that has 1 ensemble with two learnt connections
    with nengo.Network(label='Sample learning model') as model:
        # Origin of the error connections
        a = nengo.Node(lambda t: t**2, size_in=0, size_out=3)

        # Ensemble which has the learning rule
        b = nengo.Ensemble(100, 3)
        b.eval_points = np.random.normal(size=(1000, 3))

        # Terminators of the learnt connections
        c = nengo.Node(lambda t, x: None, size_in=3, size_out=0)
        d = nengo.Node(lambda t, x: None, size_in=1, size_out=0)

        # First learnt connection, rate = 1.0, width = 1
        lcm1 = nengo.Connection(a[0], d, synapse=0.3, modulatory=True)
        nengo.Connection(b, d, transform=[[1, 0, 0]],
                         learning_rule_type=nengo.PES(lcm1, 1.0))

        # Second learnt connection, rate = 0.5, width = 3
        lcm2 = nengo.Connection(a, c, synapse=0.5, modulatory=True)
        nengo.Connection(b, c, learning_rule_type=nengo.PES(lcm2, 0.5))

        # These connections should be ignored
        nengo.Connection(a, b)
        nengo.Connection(b, c)

    # Process the network using the PES tools
    (objs, conns) = pes.process_pes_connections(
        model.all_objects, model.all_connections, model.all_probes
    )
    conns = _convert_remaining_connections(conns)

    # Create the connection tree
    return ConnectionTree.from_intermediate_connections(conns), (a, b, c, d)


class TestGetPESRegions(object):
    """Check that various PES regions can be constructed together.
    """
    def test_make_filters(self, sample_ctree_with_pes):
        """Check that filters are built in the same order that the PES region
        points to them.
        """
        ctree, (a, b, c, d) = sample_ctree_with_pes

        # Get a list of the learning rules
        learning_rules = list()
        offset = 0
        for oc in ctree.get_outgoing_connections(b):
            if oc.transmitter_learning_rule is not None:
                learning_rules.append(
                    IntermediateLearningRule(oc.transmitter_learning_rule,
                                             offset)
                )
                offset += oc.transmitter_learning_rule.width

        # Get all the incoming connections
        inconns = ctree.get_incoming_connections(b)

        # Check the ports are the learning rules and Standard Input
        assert set(inconns.keys()) == (set(l.rule for l in learning_rules) |
                                       set([StandardInputPort]))

        # Get the PES region, PES filters and PES routing
        dt = 1.
        pes_region, pes_filters, pes_routing = pes.make_pes_regions(
            learning_rules, inconns, dt)

        # Check that the offsets of the PES region match the expected widths of
        # the filters...
        for pes_data in pes_region.matrix:
            rate, filter_index, decoder_index = pes_data.tolist()

            if rate == bitsk(1.0):
                # Filter width should be 1
                pes_filters[filter_index][-1] == 1
            elif rate == bitsk(0.5):
                # Filter width should be 3
                pes_filters[filter_index][-1] == 3
            else:
                assert False, "Unexpected learning rate {}".format(
                    kbits(rate))
