"""Utils for working with connections.
"""

import collections


class TransformFunctionKeyspaceConnection(
    collections.namedtuple('TransformFunctionKeyspace',
                           'transform function keyspace')):
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
            keyspace=ks
        )

    def __eq__(self, other):
        # Use hashing when comparing elements of this type
        return hash(self) == hash(other)

    def __hash__(self):
        # TODO: Provide a hash for the function to try to avoid replicating
        # connections which are functionally equivalent.  For ensembles this
        # could be hashing the function as evaluated at the given points.

        # Return a hash of the combined data, function and keyspace
        return hash((self.transform.data, self.function, self.keyspace))


def get_combined_connections(
        connections,
        reduced_connection_type=TransformFunctionKeyspaceConnection):
    """Return a set of reduced connection types that indicate the unique
    unique connections which are required.

    :param iterable connections: The set of connections to reduce.
    :param type reduced_connection_type: Type which accepts a connection and
        returns a reduced representation of it.
    :returns iterable: A set of reduced connection types.
    """
    reduced_connections = list()
    for c in map(reduced_connection_type, connections):
        if c not in reduced_connections:
            reduced_connections.append(c)
    return reduced_connections
