import mock
import nengo

from ...connections.connection_tree import ConnectionTree
from ...connections.intermediate import IntermediateConnection
from .. import ethernet as ethernet_io


class TestEthernetIO(object):
    """Tests for the EthernetIO object.
    """
    def test_prepare_connection_tree(self):
        """Test that a connection tree can be modified to include objects for
        transmitting and receiving SDP packets representing Node data.
        """
        # Create a sample connectivity tree including some Nodes
        obj_a = mock.Mock(spec_set=['size_in'])
        obj_a.size_in = 3
        node_a = nengo.Node(lambda t: t**2, size_in=0, add_to_container=False)
        node_b = nengo.Node(lambda t, x: None, size_in=3,
                            add_to_container=False)

        conns = [
            IntermediateConnection(node_a, obj_a, synapse=nengo.Lowpass(0.05)),
            IntermediateConnection(obj_a, node_b, synapse=nengo.Lowpass(0.05)),
        ]
        ctree = ConnectionTree.from_intermediate_connections(conns)

        # Pass the connection tree through the IO object
        new_ctree = ethernet_io.EthernetIO.prepare_connection_tree(ctree)

        # Check that the nodes are no longer present, but that the mock object
        # is.
        new_objs = new_ctree.get_objects()
        assert all([obj_a in new_objs,
                    node_a not in new_objs,
                    node_b not in new_objs])

        # Check that the connections reflect the correct connections of Rx and
        # Tx elements.
        for obj in new_objs:
            if obj is obj_a:
                # Outgoing connections for obj_a should be essentially
                # unchanged.
                assert (new_ctree.get_outgoing_connections(obj) ==
                        ctree.get_outgoing_connections(obj))
            elif isinstance(obj, ethernet_io.TransmitObject):
                # Object is the replacement for node_a, assert that the
                # connectivity is equivalent and that the new object refers to
                # the old.
                assert (new_ctree.get_outgoing_connections(obj) ==
                        ctree.get_outgoing_connections(node_a))
                assert obj.node is node_a
            elif isinstance(obj, ethernet_io.ReceiveObject):
                # Object is the replacement for node_b, assert that the
                # connectivity is equivalent and that the new object refers to
                # the old.
                assert (new_ctree.get_incoming_connections(obj) ==
                        ctree.get_incoming_connections(node_b))
                assert obj.node is node_b
            else:
                assert False, "Unexpected object {} found.".format(obj)


def test_get_output_node_slices_to_core_map():
    raise NotImplementedError


def test_get_input_core_to_node_slice_map():
    raise NotImplementedError
