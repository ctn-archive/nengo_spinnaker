import mock
import nengo
import numpy as np

from pacman.model.graph_mapper.slice import Slice
from ...assembler import Assembler
from ...connections.connection_tree import ConnectionTree
from ...connections.intermediate import IntermediateConnection
from .. import ethernet as ethernet_io
from ..sdp_rx_vertex import SDPRxVertex, SDPRxVertexSystemRegion


def test_assemble_from_receive_object():
    """Test the assembly of a SDPTxVertex from a receive object."""
    # Create a small sample network
    node_a = nengo.Node(lambda t: t, size_in=0, size_out=1,
                        add_to_container=False)
    obj_b = mock.Mock(name="ObjA", spec_set=['size_in'])
    obj_b.size_in = 1
    keyspace = mock.Mock(name="Keyspace")

    ctree = ConnectionTree.from_intermediate_connections([
        IntermediateConnection(node_a, obj_b, synapse=nengo.Lowpass(0.05),
                               keyspace=keyspace),
    ])

    # Build using the ethernet builder, then assemble.
    ctree = ethernet_io.EthernetIO.prepare_connection_tree(ctree)
    (objs, conns) = Assembler.assemble(ctree, None, {}, 10., 0.001, 1000)

    # Extract the SDPTxVertex from the objects
    assert len(objs) == 2
    assert obj_b in objs
    for obj in objs:
        if obj is obj_b:
            pass
        elif type(obj) is SDPRxVertex:
            sdp_rx_vertex = obj
        else:
            assert False, "Unexpected object {} found.".format(obj)

    # Check the SDP Tx Vertex
    assert sdp_rx_vertex.node is node_a
    assert len(sdp_rx_vertex.regions) == 2

    # Ensure that we can get resources
    sdp_rx_vertex.get_resources_used_by_atoms(Slice(0, 9), None)

    # Ensure the additional DTCM resources are returned correctly
    assert sdp_rx_vertex.get_dtcm_usage_static(Slice(0, 9)) == 10
    assert sdp_rx_vertex.get_dtcm_usage_static(Slice(0, 4)) == 5


def test_sdp_rx_system_region():
    """Test the creation and slicing of the SDP Rx system region."""
    region = SDPRxVertexSystemRegion(1000)

    # Check the sizing reports
    assert region.sizeof(Slice(0, 99)) == 2
    assert region.sizeof(slice(10)) == 2

    # Check the subregion data
    sr_data = np.frombuffer(region.create_subregion(Slice(0, 9), 1).data,
                            dtype=np.uint32).tolist()
    assert sr_data == [1000, 10]

    sr_data = np.frombuffer(region.create_subregion(Slice(5, 5), 1).data,
                            dtype=np.uint32).tolist()
    assert sr_data == [1000, 1]
