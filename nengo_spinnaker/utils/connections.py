"""Utils for working with connections.
"""

import collections
import copy
import numpy as np
import nengo

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
