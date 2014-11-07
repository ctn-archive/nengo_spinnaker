"""Objects which perform filtering but nothing else.
"""
import numpy as np

from utils.connections import get_pre_padded_transform
from utils.fixpoint import bitsk
from .spinnaker.regions import MatrixRegion, MatrixRegionPartitionedByRows


def make_filter_system_region(size_in, size_out, machine_timestep,
                              transmission_delay, interpacket_pause):
    """Create the system region for a filter vertex.

    TODO: Remove the requirements for `size_in` and `size_out`, these should be
    read from the first two words of the transform region.
    """
    # Create a matrix region containing the data
    return MatrixRegion(np.array([size_in, size_out, machine_timestep,
                                  transmission_delay, interpacket_pause],
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
    return MatrixRegionPartitionedByRows(transform, formatter=bitsk)


def make_transform_region_data(outgoing_conns, size_in):
    """Convert outgoing connections into a combined transform matrix.
    """
    # Build up a list of partial transforms
    transforms = list()

    for c in outgoing_conns:
        assert c.function is None  # Filters cannot apply functions

        # Add the padded version of this transform
        transforms.append(get_pre_padded_transform(c.pre_slice, size_in,
                                                   c.transform))

    # Stack and return the transforms
    return np.vstack(transforms)
