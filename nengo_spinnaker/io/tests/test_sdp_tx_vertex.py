import mock
import nengo

from pacman.model.graph_mapper.slice import Slice
from ...assembler import Assembler
from ...connections.connection_tree import ConnectionTree
from ...connections.intermediate import IntermediateConnection
from .. import ethernet as ethernet_io
from ..sdp_tx_vertex import SDPTxVertex


def test_assemble_from_receive_object():
    """Test the assembly of a SDPTxVertex from a receive object."""
    # Create a small sample network
    obj_a = mock.Mock(name="ObjA")
    node_b = nengo.Node(lambda t, x: None, size_in=1, size_out=0,
                        add_to_container=False)
    keyspace = mock.Mock(name="Keyspace")

    ctree = ConnectionTree.from_intermediate_connections([
        IntermediateConnection(obj_a, node_b, synapse=nengo.Lowpass(0.05),
                               keyspace=keyspace),
    ])

    # Build using the ethernet builder, then assemble.
    ctree = ethernet_io.EthernetIO.prepare_connection_tree(ctree)
    (objs, conns) = Assembler.assemble(ctree, None, {}, 10., 0.001, 1000)

    # Extract the SDPTxVertex from the objects
    assert len(objs) == 2
    assert obj_a in objs
    for obj in objs:
        if obj is obj_a:
            pass
        elif type(obj) is SDPTxVertex:
            sdp_tx_vertex = obj
        else:
            assert False, "Unexpected object {} found.".format(obj)

    # Check the SDP Tx Vertex
    assert sdp_tx_vertex.node is node_b
    assert len(sdp_tx_vertex.regions) == 3

    # Ensure that we can get resources
    sdp_tx_vertex.get_resources_used_by_atoms(Slice(0, 9), None)
