"""Utils for working with connections.
"""

import copy
from math import ceil
import nengo
import numpy as np

from ..connections.intermediate import IntermediateConnection


def replace_objects_in_connections(connections, replaced_objects):
    """Replace connections where the pre- or post-object has been replaced.

    :param iterable connections: A list of connections to process.
    :param dict replaced_objects: A mapping of old to new objects.
    :returns iterable: A list of connections where connections that included
        references to old objects have been replaced with connections with
        references to new objects.
    """
    new_connections = []
    for c in connections:
        if (c.pre_obj not in replaced_objects and
                c.post_obj not in replaced_objects):
            new_connections.append(c)
            continue

        if isinstance(c, nengo.Connection):
            new_c = IntermediateConnection.from_connection(c)
        else:
            new_c = copy.copy(c)

        if c.pre_obj in replaced_objects:
            new_c.pre_obj = replaced_objects[c.pre_obj]
        if c.post_obj in replaced_objects:
            new_c.post_obj = replaced_objects[c.post_obj]

        new_connections.append(new_c)

    return new_connections


def get_pre_padded_transform(pre_slice, size_in, transform):
    """Get a new transform that accounts for slicing applied to the input.
    """
    # Get a copy of the transform
    transform = np.array(transform)

    # If there is no pre-slicing then simply return a copy of the provided
    # transform.
    if pre_slice == slice(None) and transform.ndim == 2:
        return transform

    if transform.ndim < 2:
        # The given transform is a scalar, so we create an identity matrix of
        # the correct size which we then pad.
        if pre_slice == slice(None):
            size = size_in
        else:
            size = int(ceil(
                (pre_slice.stop - pre_slice.start) /
                float(1 if pre_slice.step is None else pre_slice.step)
            ))
        transform = np.dot(transform, np.eye(size))

    # Otherwise apply the pre-padding
    if transform.ndim == 2:
        # The full transform is size_out x size_in:
        full_transform = np.zeros((transform.shape[0], size_in))

        # Copy in the transform that we already have
        full_transform[:, pre_slice] = transform
        return full_transform
    else:
        raise ValueError("Transforms with > 2 dims not supported")


def get_keyspaces_with_dimensions(outgoing_conns):
    """Get a list of outgoing keyspaces for the given connections.

    "Explodes" the keys for the list of outgoing connections, filling in the
    dimensions (`d`) field as it progresses.

    TODO: Change this to allow receiver dimension mapping.
    """
    keyspaces = list()

    # Add the keyspaces resulting from each outgoing connection
    for c in outgoing_conns:
        # Get the dimensions represented on this connection
        dims = range(c.width)[c.post_slice]

        # Create the keyspaces
        keyspaces.extend([c.keyspace(d=d) for d in dims])

    # Return all the keyspaces
    return keyspaces
