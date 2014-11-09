import numpy as np

from ..assembler import Assembler
from ..spinnaker import regions
from ..spinnaker.vertices import Vertex
from ..utils import connections as connection_utils


class TransmitObject(object):
    """Object which transmits output on behalf of a Node.

    Maps to SDPRxVertex because it receives SDP packets and transmits multicast
    packets.
    """
    __slots__ = ['node']

    def __init__(self, node):
        self.node = node

    def __repr__(self):  # pragma: no cover
        return "<{} for {}>".format(self.__class__.__name__, self.node)


class SDPRxVertex(Vertex):
    """Vertex which receives SDP packets and transmits their content as
    multicast packets.

    Each word in the SDP packet is interpreted as the value of a component of a
    vector and is transmitted with the key which describes the component.
    Slicing of this vertex allows different partitioned vertices to represent
    different parts of the output space.  The limit is 64 atoms per partition.
    """
    # TODO Ensure that the maximum number of atoms (64) is respected.

    def __init__(self, node, outgoing_width, system_region,
                 outgoing_keys_region):
        """Create a new SDPRxVertex.

        :type node: :py:class:`~nengo.Node`
        :param int n_atoms: The total number of packets which will be
            transmitted per-update by this SDPRxVertex.
        :type system_region: :py:class:`SDPRxVertexSystemRegion`
        :type outgoing_keys_region: :py:class:`~regions.KeysRegion`
        """
        super(SDPRxVertex, self).__init__(
            outgoing_width,
            label="{} for {}".format(self.__class__.__name__, node),
            regions=[system_region, outgoing_keys_region]
        )
        self.node = node

    def get_cpu_usage_for_atoms(self, vertex_slice):
        # Not really significant here
        return 0

    def get_dtcm_usage_static(self, vertex_slice):
        """Get the DTCM used that is not part of a region (in WORDS).

        In the case of a SDPRxVertex this is the cached version of the current
        output values.

        :param vertex_slice: Slice of the vertex.
        :type vertex_slice: :py:func:`slice`
        """
        return vertex_slice.stop - vertex_slice.start


@Assembler.object_assembler(TransmitObject)
def assemble_sdp_rx_vertex_from_transmit_object(obj, connection_trees, config,
                                                rngs, runtime, dt,
                                                machine_timestep):
    """Convert a TransmitObject into a SDPRxVertex.

    :type obj: :py:class:`TransmitObject`
    :type connection_trees:
        :py:class:`~..connections.connection_tree.ConnectionTree`
    :rtype: :py:class:`SDPRxVertex`
    """
    # TODO Get the number of atoms from the number of outgoing keys?
    # Create the system region
    system_region = SDPRxVertexSystemRegion(machine_timestep)

    # Create the output keys region
    output_keys = connection_utils.get_keyspaces_with_dimensions(
        connection_trees.get_outgoing_connections(obj))
    outgoing_width = len(output_keys)
    output_keys_region = regions.KeysRegion(output_keys, fill_in_field='s',
                                            partitioned=True)

    # Return the new vertex
    return SDPRxVertex(obj.node, outgoing_width, system_region,
                       output_keys_region)


class SDPRxVertexSystemRegion(regions.Region):
    def __init__(self, transmission_period):
        """Create a new system region for a SDPRxVertex.

        :param int transmission_period: The period between transmitting
            multicast packets (usually something like the machine timestep).
        """
        super(SDPRxVertexSystemRegion, self).__init__()
        self.transmission_period = transmission_period

    def sizeof(self, vertex_slice):
        """Get the size (in words) of the region."""
        return 2

    def create_subregion(self, vertex_slice, subvertex_index):
        """Create a smaller version of the region ready to write to memory.

        :type vertex_slice: :py:func:`slice`
        :type subvertex_index: int
        :rtype: :py:class:`~regions.Subregion_`
        """
        # Get the number of atoms, this is the 2nd word of the system region
        n_atoms = vertex_slice.stop - vertex_slice.start

        # Create the system region data
        data = np.array([self.transmission_period, n_atoms], dtype=np.uint32)

        return regions.Subregion(data, len(data), False)
