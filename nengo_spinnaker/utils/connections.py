"""Manage collections of Connections.
"""

import collections
import numpy as np

import nengo


TransformFunctionKeyspace = collections.namedtuple(
    'TransformFunctionPair', ['transform', 'function', 'keyspace'])


TransformFunctionWithSolverEvalPoints = collections.namedtuple(
    'TransformFunctionWithSolverEvalPoints',
    ['transform', 'function', 'solver', 'eval_points', 'keyspace'])


class Connections(object):
    """Generates a list of unique transform, function, keyspace triples.

    Merge together equivalent connections when they share a transform,
    function, source and keyspace.
    """
    def __init__(self, connections=[]):
        self._connection_indices = dict()
        self._source = None
        self.transforms_functions = list()

        for connection in connections:
            # If the connection is a tuple then it's (connection, keyspace)
            if isinstance(connection, tuple) and len(connection) == 2:
                self.add_connection(connection[0], connection[1])
            else:
                self.add_connection(connection)

    def add_connection(self, connection, keyspace=None):
        # Ensure that this Connection collection is only for connections from
        # the same source object
        if self._source is None:
            self._source = connection.pre_obj
        assert(self._source == connection.pre_obj)

        # Get the index of the connection if the same transform and function
        # have already been added, otherwise add the transform/function pair
        connection_entry = self._make_connection_entry(
            connection, connection.transform, connection.keyspace)

        # For each pre_obj-existing unique connection see if this connection
        # matches
        for (i, tf) in enumerate(self.transforms_functions):
            if self._are_compatible_connections(tf, connection_entry):
                # If it does then the index for this connection is the same as
                # that for the unique connection set
                index = i
                break
        else:
            # Otherwise create a new transform/function/keyspace entry and
            # use its index.
            self.transforms_functions.append(connection_entry)
            index = len(self.transforms_functions) - 1

        self._connection_indices[connection] = index

    def contains_compatible_connection(self, connection,
                                       keyspace=None):
        """Does the Connection block already contain an equivalent connection.
        """
        # It doesn't if the connections have different sources
        if self._source is None or self._source != connection.pre_obj:
            return False

        # Simulate an entry for the given connection
        connection_entry = self._make_connection_entry(
            connection, connection.transform, keyspace)

        # For each entry in the Connections block is the given connection
        # compatible?
        for tf in self.transforms_functions:
            if self._are_compatible_connections(tf, connection_entry):
                return True
        return False

    def _are_compatible_connections(self, c1, c2):
        return (np.all(c1.transform == c2.transform) and
                c1.function == c2.function and c1.keyspace == c2.keyspace)

    def _make_connection_entry(self, connection, transform,
                               keyspace=None):
        return TransformFunctionKeyspace(transform, connection.function,
                                         keyspace)

    @property
    def width(self):
        # The total dimensionality of __all__ connections
        return sum([t.transform.shape[0] for t in self.transforms_functions])

    def get_connection_offset(self, connection):
        # Get the offset (width of the connection block up until this
        # connection)
        i = self[connection]
        return sum([t.transform.shape[0] for t in
                    self.transforms_functions[:i]])

    def __len__(self):
        # Number of unique transform/function/keyspaces/...
        return len(self.transforms_functions)

    def __iter__(self):
        return iter(self._connection_indices)

    def __getitem__(self, connection):
        return self._connection_indices[connection]


class OutgoingEnsembleConnections(Connections):
    def _are_compatible_connections(self, c1, c2):
        return (np.all(c1.transform == c2.transform) and
                np.all(c1.eval_points == c2.eval_points) and
                c1.solver == c2.solver and
                c1.function == c2.function and c1.keyspace == c2.keyspace)

    def _make_connection_entry(self, connection, transform,
                               keyspace=None):
        return TransformFunctionWithSolverEvalPoints(
            transform, connection.function, connection.solver,
            connection.eval_points, keyspace)


FilteredConnection = collections.namedtuple(
    'FilteredConnection', ['time_constant', 'is_accumulatory', 'modulatory', 'width'])


class Filters(object):
    def __init__(self, connections_with_filters):
        self._connection_indices = dict()
        self._termination = None
        self.filters = list()

        for connection in connections_with_filters:
            self.add_connection(connection)

    def add_connection(self, connection):
        if self._termination is None:
            self._termination = connection.post_obj
        assert(self._termination == connection.post_obj)

        if (connection.synapse is not None and
                not isinstance(connection.synapse, float) and
                not isinstance(connection.synapse, nengo.synapses.Lowpass)):
            raise NotImplementedError("Currently only support Lowpass "
                                      "synapse model. Not '%s'." %
                                      connection.synapse.__class__.__name__)

        # If this filter isn't modulatory (modulatory signals need to be kept
        # Seperate, if its parameters match existing filter, use its index
        index = None
        for (i, f) in enumerate(self.filters):
            if (connection.modulatory == False and
                connection.is_accumulatory == f.is_accumulatory and
                connection.synapse == f.time_constant):
                index = i
                break
        else:
            if isinstance(connection.synapse, nengo.synapses.Lowpass):
                syn = connection.synapse.tau
            else:
                syn = connection.synapse

            new_f = FilteredConnection(syn, connection.is_accumulatory, 
                connection.modulatory, connection.width)
            self.filters.append(new_f)
            index = len(self.filters) - 1

        self._connection_indices[connection] = index

    def __getitem__(self, connection):
        return self._connection_indices[connection]

    def __len__(self):
        return len(self.filters)


def get_output_keys(connections):
    """Return a list of output keys for the given connection
    block.
    """
    keys = list()
    for tfk in connections.transforms_functions:
        for d in range(tfk.transform.shape[0]):
            keys.append(tfk.keyspace.key(d=d))
    return keys

def get_learning_rules(connection):
    """ Converts all possible forms of connection's learning rule
    Parameters into things that can be iterated.
    """ 
    if nengo.utils.compat.is_iterable(connection.learning_rule):
        return connection.learning_rule
    elif connection.learning_rule is not None:
        return (connection.learning_rule,)
    else:
        return ()