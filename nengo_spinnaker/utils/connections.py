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
        for (i, tf) in enumerate(self.transforms_functions):
            if (np.all(tf.transform == transform) and
                    tf.function == connection.function):
                index = i
                break
        else:
            self.transforms_functions.append(
                TransformFunctionPair(transform, connection.function))
            index = len(self.transforms_functions) - 1

        self._connection_indices[connection] = index

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
    def add_connection(self, connection):
        # Ensure that this Connection collection is only for connections from
        # the same source object
        if self._source is None:
            self._source = connection.pre
        assert(self._source == connection.pre)

        # Get the index of the connection if the same transform and function
        # have already been added, otherwise add the transform/function pair
        transform = full_transform(connection, allow_scalars=False)
        for (i, tf) in enumerate(self.transforms_functions):
            if (np.all(tf.transform == transform) and
                    tf.function == connection.function and
                    tf.solver == connection.solver and
                    np.all(tf.eval_points == connection.eval_points)):
                index = i
                break
        else:
            self.transforms_functions.append(
                TransformFunctionWithSolverEvalPoints(transform,
                                                      connection.function,
                                                      connection.solver,
                                                      connection.eval_points))
            index = len(self.transforms_functions) - 1

        self._connection_indices[connection] = index


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

