"""Utils for working with connections.
"""

import collections
import copy
import numpy as np
import nengo

from .. import connection


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
            new_c = connection.IntermediateConnection.from_connection(c)
        else:
            new_c = copy.copy(c)

        if c.pre_obj in replaced_objects:
            new_c.pre_obj = replaced_objects[c.pre_obj]
        if c.post_obj in replaced_objects:
            new_c.post_obj = replaced_objects[c.post_obj]

        new_connections.append(new_c)

    return new_connections


class TransformFunctionKeyspaceConnection(
    collections.namedtuple('TransformFunctionKeyspace',
                           'transform function keyspace pre_obj post_obj')):
    """Minimum representation of a connection provided to allow trivial
    combination of equivalent connections.
    """
    def __new__(cls, connection):
        # Get the keyspace, or None by default
        ks = getattr(connection, 'keyspace', None)

        # Create a non-writeable copy of the transform applied to this
        # connection.
        transform = connection.transform.copy()
        transform.flags.writeable = False

        return super(TransformFunctionKeyspaceConnection, cls).__new__(
            cls, transform=transform, function=connection.function,
            keyspace=ks, pre_obj=connection.pre_obj,
            post_obj=connection.post_obj
        )

    def __eq__(self, other):
        # Compare all fields:
        # We check that the pre objs ARE equivalent but that the post objs
        # AREN'T because we explicitly want that: (A->B) || (A->B) = 2(A->B)
        # [`||' indicates ``in parallel'].
        # Additionally, this is a strong requirement for function equivalence -
        # for example `id(lambda x: x) != id(lambda x: x)` - meaning that we
        # probably miss out on a lot of opportunities for sharing connections
        # on ensembles.
        return all([
            np.all(self.transform == other.transform),
            self.function is other.function,
            self.keyspace == other.keyspace,
            self.pre_obj is other.pre_obj,
            self.post_obj is not other.post_obj,
        ])

    def __hash__(self):
        # TODO: Provide a hash for the function to try to avoid replicating
        # connections which are functionally equivalent.  For ensembles this
        # could be hashing the function as evaluated at the given points.

        # Return a hash of the combined data, function and keyspace.  The post
        # object is not included because equivalence in hash(post_obj) actually
        # means the objects are non-equivalent...
        return hash((self.transform.data, self.function, self.keyspace,
                     self.pre_obj))


def get_combined_connections(
        connections,
        reduced_connection_type=TransformFunctionKeyspaceConnection):
    """Return a set of reduced connection types that indicate the unique
    unique connections which are required.

    :param iterable connections: The set of connections to reduce.
    :param type reduced_connection_type: Type which accepts a connection and
        returns a reduced representation of it.
    :returns tuple: A tuple containing an iterable of reduced connection types
        and a dictionary mapping connections to an index into the list of
        reduced connections.
    """
    reduced_connections = list()
    connection_map = dict()

    # TODO: Detect the case where there are two identical connections between
    # the SAME objects and increase the transform accordingly, i.e.
    # (A->B) || (A->B) => 2(A->B).

    for (c, rc) in [(c, reduced_connection_type(c)) for c in connections]:
        if rc not in reduced_connections:
            reduced_connections.append(rc)
        connection_map[c] = reduced_connections.index(rc)

    return reduced_connections, connection_map
