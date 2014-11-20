import numpy as np

from ..assembler import Assembler
from ..connections.reduced import StandardInputPort
from ..spinnaker.vertices import Vertex
from ..spinnaker import regions
from ..utils import filters as filter_utils


class ReceiveObject(object):
    """Object which receives input on behalf of a Node.

    Maps to SDPTxVertex because it receives multicast packets and
    transmits SDP packets.
    """
    __slots__ = ['node', 'transmission_delay']

    def __init__(self, node, transmission_delay=100):
        """Create a new ReceiveObject.

        :param node: The Nengo Node this receive object represents.
        :type node: :py:class:`~nengo.Node`
        :param int transmission_delay: The number of timesteps between
            transmitting SDP packets.
        """
        self.node = node
        self.transmission_delay = transmission_delay


class SDPTxVertex(Vertex):
    """Vertex which receives and filters multicast packets and periodically
    transmits its current state to the host using SDP.
    """
    # TODO Enable slicing to accept input > 64D

    def __init__(self, node, system_region, filters_region,
                 filter_routing_region):
        super(SDPTxVertex, self).__init__(
            1, label="{} for {}".format(self.__class__.__name__, node),
            regions=[system_region, filters_region, filter_routing_region])
        self.node = node

    def get_cpu_usage_for_atoms(self, vertex_slice):
        # Not significant for vertices of this type
        return 0


@Assembler.object_assembler(ReceiveObject)
def assemble_sdp_tx_vertex_from_receive_object(obj, connection_trees,
                                               config, rngs, runtime, dt,
                                               machine_timestep):
    """Convert a ReceiveObject into a SDPTxVertex.

    :type obj: :py:class:`ReceiveObject`
    :type connection_trees:
        :py:class:`~..connections.connection_tree.ConnectionTree`
    :rtype: :py:class:`SDPTxVertex`
    """
    # Create the system region
    system_region = _create_system_region(obj.node.size_in, machine_timestep,
                                          obj.transmission_delay)

    # Create the filter regions
    filters_region, filter_routing_region = filter_utils.get_filter_regions(
        connection_trees.get_incoming_connections(obj)[StandardInputPort], dt,
        obj.node.size_in
    )

    # Return the new vertex
    return SDPTxVertex(obj.node, system_region, filters_region,
                       filter_routing_region)


def _create_system_region(width, machine_timestep, transmission_delay):
    """Create a new system region for a SDPTxVertex.

    :param int width: Number of components in the vector expected as input for
        the Node.
    :param int machine_timestep: The machine timestep in usec.
    :param int transmission_delay: The number of machine timesteps to wait
        between transmitting SDP packets.
    :rtype:
        :py:class:`~regions.MatrixRegion`
    """
    if width > 64:
        raise NotImplementedError("Implement slicing to allow for nodes "
                                  "size_in > 64")

    data = np.array([width, machine_timestep, transmission_delay],
                    dtype=np.uint32)
    return regions.MatrixRegion(data)
