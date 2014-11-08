"""Objects which perform filtering but nothing else.
"""
import numpy as np

from ..connections.reduced import StandardInputPort
from ..utils import connections as connection_utils
from ..utils.filters import get_filter_regions
from ..utils.fixpoint import bitsk
from ..spinnaker.vertices import Vertex
from ..spinnaker import regions


class ValueFilter(object):
    def __init__(self, size_in, transmission_delay, interpacket_pause,
                 label=""):
        """Create a new ValueFilter.

        :param int size_in: The size of the vector expected as input.
        :param int transmission_delay: Delay between transmitting streams of
                                       packets, measured in machine timesteps.
        :param int interpacket_pause: Delay between transmitting packets,
                                      measured in usec.
        """
        self.size_in = size_in
        self.transmission_delay = transmission_delay
        self.interpacket_pause = interpacket_pause
        self.label = label


class ValueFilterVertex(Vertex):
    """Vertex which performs filtering and transforms on values.
    """
    def __init__(self, label, system_region, output_keys_region,
                 filters_region, filters_routing_region, transform_region):
        super(ValueFilterVertex, self).__init__(
            n_atoms=1,  # TODO: Partition on transform rows?
            label=label,
            regions=[
                system_region, output_keys_region, filters_region,
                filters_routing_region, transform_region,
            ]
        )

    def get_cpu_usage_for_atoms(self, vertex_slice):
        # Not really useful for this vertex yet
        return 0

    @classmethod
    def from_value_filter(cls, filter_object, connection_trees, config, rngs,
                          runtime, dt, machine_timestep):
        """Create a new ValueFilterVertex from a ValueFilter.

        :type filter_object: ValueFilter
        :type connection_trees: ..connections.connection_tree.ConnectionTree
        """
        # Get inputs and output for the filter
        inputs = connection_trees.get_incoming_connections(
            filter_object)[StandardInputPort]
        outputs = connection_trees.get_outgoing_connections(filter_object)

        # Create the output keys and transform region
        outgoing_keyspaces = \
            connection_utils.get_keyspaces_with_dimensions(outputs)
        output_keys_region = regions.KeysRegion(outgoing_keyspaces)

        transform_region = make_transform_region(outputs,
                                                 filter_object.size_in)

        # Create the filters and filter routing regions
        filters_region, filters_routing_region = get_filter_regions(
            inputs, dt, filter_object.size_in)

        # Create the system region
        system_region = make_filter_system_region(
            filter_object.size_in, len(outgoing_keyspaces), machine_timestep,
            filter_object.transmission_delay, filter_object.interpacket_pause
        )

        # Create and return the vertex
        return cls(filter_object.label, system_region, output_keys_region,
                   filters_region, filters_routing_region, transform_region)


def make_filter_system_region(size_in, size_out, machine_timestep,
                              transmission_delay, interpacket_pause):
    """Create the system region for a filter vertex.

    TODO: Remove the requirements for `size_in` and `size_out`, these should be
    read from the first two words of the transform region.
    """
    # Create a matrix region containing the data
    return regions.MatrixRegion(np.array([size_in, size_out, machine_timestep,
                                          transmission_delay,
                                          interpacket_pause],
                                         dtype=np.uint32), shape=(5,))


def make_transform_region(outgoing_conns, size_in):
    """Create a region representing the transform applied by the connections.
    """
    # Get the data for the region
    transform = make_transform_region_data(outgoing_conns, size_in)
    assert transform.shape[1] == size_in

    # Return a matrix region
    # TODO Append the number of rows and number of columns, to avoid reading
    # these out of the system region.
    return regions.MatrixRegionPartitionedByRows(transform, formatter=bitsk)


def make_transform_region_data(outgoing_conns, size_in):
    """Convert outgoing connections into a combined transform matrix.
    """
    # Build up a list of partial transforms
    transforms = list()

    for c in outgoing_conns:
        assert c.function is None  # Filters cannot apply functions

        # Add the padded version of this transform
        transforms.append(
            connection_utils.get_pre_padded_transform(c.pre_slice, size_in,
                                                      c.transform))

    # Stack and return the transforms
    return np.vstack(transforms)
