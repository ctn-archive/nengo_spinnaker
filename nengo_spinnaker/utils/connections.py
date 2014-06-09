"""Manage collections of Connections.
"""

import collections
import numpy as np

from nengo.utils.builder import full_transform


TransformFunctionPair = collections.namedtuple(
    'TransformFunctionPair', ['transform', 'function'])


TransformFunctionWithSolverEvalPoints = collections.namedtuple(
    'TransformFunctionWithSolverEvalPoints',
    ['transform', 'function', 'solver', 'eval_points'])


class Connections(object):
    def __init__(self, connections=[]):
        self._connection_indices = dict()
        self._source = None
        self.transforms_functions = list()

        for connection in connections:
            self.add_connection(connection)

    def add_connection(self, connection):
        # Ensure that this Connection collection is only for connections from
        # the same source object
        if self._source is None:
            self._source = connection.pre
        assert(self._source == connection.pre)

        # Get the index of the connection if the same transform and function
        # have already been added, otherwise add the transform/function pair
        transform = full_transform(connection, allow_scalars=False)
        connection_entry = self._make_connection_entry(connection, transform)
        for (i, tf) in enumerate(self.transforms_functions):
            if self._are_compatible_connections(tf, connection_entry):
                index = i
                break
        else:
            self.transforms_functions.append(connection_entry)
            index = len(self.transforms_functions) - 1

        self._connection_indices[connection] = index

    def contains_compatible_connection(self, connection):
        if self._source is None or self._source != connection.pre:
            return False

        for tf in self.transforms_functions:
            if self._are_compatible_connections(tf, connection):
                return True
        return False

    def _are_compatible_connections(self, c1, c2):
        return (np.all(c1.transform == c2.transform) and
                c1.function == c2.function)

    def _make_connection_entry(self, connection, transform):
        return TransformFunctionPair(transform, connection.function)

    @property
    def width(self):
        return sum([t.transform.shape[0] for t in self.transforms_functions])

    def get_connection_offset(self, connection):
        i = self[connection]
        return sum([t.transform.shape[0] for t in
                    self.transforms_functions[:i]])

    def __len__(self):
        return len(self.transforms_functions)

    def __iter__(self):
        return iter(self._connection_indices)

    def __getitem__(self, connection):
        return self._connection_indices[connection]


class ConnectionsWithSolvers(Connections):
    def _are_compatible_connections(self, c1, c2):
        return (np.all(c1.transform == c2.transform) and
                np.all(c1.eval_points == c2.eval_points) and
                c1.solver == c2.solver and
                c1.function == c2.function)

    def _make_connection_entry(self, connection, transform):
        return TransformFunctionWithSolverEvalPoints(
            transform, connection.function, connection.solver,
            connection.eval_points)


class ConnectionBank(object):
    """Represents a set of Connection collections which need not necessarily
    share the same source object.
    """

    def __init__(self, connections, collector_type=Connections):
        self._connections = collections.defaultdict(collector_type)

        for connection in connections:
            self.add_connection(connection)

    def add_connection(self, connection):
        self._connections[connection.pre].add_connection(connection)

    def get_connection_offset(self, connection):
        offset = 0
        for connections in self._connections.values():
            if connection in connections:
                return offset + connections.get_connection_offset(connection)
            offset += connections.width
        raise KeyError

    def contains_compatible_connection(self, connection):
        return self._connections[connection.pre].\
            contains_compatible_connection(connection)

    @property
    def width(self):
        return sum([c.width for c in self._connections.values()])

    def __getitem__(self, connection):
        index = 0
        for connections in self._connections.values():
            if connection in connections:
                return index + connections[connection]
            index += len(connections)
        raise KeyError(connection)

    def __iter__(self):
        """Iterate through the list of connections."""
        for connections in self._connections.values():
            for c in connections:
                yield c


ConnectionWithFilter = collections.namedtuple(
    'ConnectionWithFilter', ['connection', 'accumulatory'])
FilteredConnection = collections.namedtuple(
    'FilteredConnection', ['time_constant', 'accumulatory'])


class Filters(object):
    def __init__(self, connections_with_filters):
        self._connection_indices = dict()
        self._termination = None
        self.filters = list()

        for connection in connections_with_filters:
            self.add_connection(connection)

    def add_connection(self, connection_with_filter):
        if self._termination is None:
            self._termination = connection_with_filter.connection.post
        assert(self._termination == connection_with_filter.connection.post)

        index = None
        for (i, f) in enumerate(self.filters):
            if (connection_with_filter.accumulatory == f.accumulatory and
                connection_with_filter.connection.synapse == f.time_constant):
                index = i
                break
        else:
            new_f = FilteredConnection(
                connection_with_filter.connection.synapse,
                connection_with_filter.accumulatory
            )
            self.filters.append(new_f)
            index = len(self.filters) - 1

        self._connection_indices[connection_with_filter.connection] = index

    def __getitem__(self, connection):
        return self._connection_indices[connection]

    def __len__(self):
        return len(self.filters)
