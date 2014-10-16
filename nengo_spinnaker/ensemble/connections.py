"""Connection utilities required solely by Ensembles.
"""

import collections
import nengo
import numpy as np

from nengo.utils.compat import is_iterable

from .. import connection
from ..utils import connections as connection_utils


class TransformFunctionKeyspaceEvalSolverConnection(
    collections.namedtuple('TransformFunctionKeyspace',
                           'transform function keyspace eval_points solver')):
    """Minimum representation of a connection provided to allow trivial
    combination of equivalent connections.
    """
    def __new__(cls, connection):
        # Get the keyspace, or None by default
        ks = getattr(connection, 'keyspace', None)

        # Create a non-writeable copy of the transform applied to this
        # connection, likewise for the eval points (if any).
        transform = connection.transform.copy()
        transform.flags.writeable = False
        eval_points = None

        if connection.eval_points is not None:
            eval_points = connection.eval_points.copy()
            eval_points.flags.writeable = False

        return super(TransformFunctionKeyspaceEvalSolverConnection, cls).\
            __new__(cls, transform=transform, function=connection.function,
                    keyspace=ks, eval_points=eval_points,
                    solver=connection.solver)

    def __eq__(self, other):
        # Use hashing when comparing elements of this type
        return hash(self) == hash(other)

    def __hash__(self):
        # TODO: Provide a hash for the function to try to avoid replicating
        # connections which are functionally equivalent.  For ensembles this
        # could be hashing the function as evaluated at the given points.

        # Return a hash of the combined data, function and keyspace
        return hash((
            self.transform.data, self.function, self.keyspace,
            None if self.eval_points is None else self.eval_points.data,
            self.solver
        ))


def get_combined_outgoing_ensemble_connections(connections):
    return connection_utils.get_combined_connections(
        connections=connections,
        reduced_connection_type=TransformFunctionKeyspaceEvalSolverConnection
    )


def process_global_inhibition_connections(objs, connections, probes):
    """Replace connections which represent globally inhibitive connections.

    A global inhibition connection is one which connects from the decoded
    representation of one ensemble to ALL ensembles in another.  It is
    identifiable because the transform on the connection is [v, v, ..., v].
    """
    new_connections = list()
    for c in connections:
        if (isinstance(c.post_obj, nengo.ensemble.Neurons) and
                np.all([c.transform[0] == t for t in c.transform])):
            # This is a global inhibition connection, swap out
            c = IntermediateGlobalInhibitionConnection.from_connection(c)
        new_connections.append(c)

    return objs, new_connections


class IntermediateGlobalInhibitionConnection(
        connection.IntermediateConnection):
    """Representation of a connection which is a global inhibition connection.
    """
    @classmethod
    def from_connection(cls, c):
        # Assert that the transform is as we'd expect
        assert isinstance(c.post_obj, nengo.ensemble.Neurons)
        assert np.all([c.transform[0] == t for t in c.transform])

        # Compress the transform to have output dimension of 1
        tr = c.transform[0][0]

        # Get the keyspace for the connection
        keyspace = getattr(c, 'keyspace', None)

        # Create a new instance
        return cls(c.pre_obj, c.post_obj.ensemble, c.synapse, c.function, tr,
                   c.solver, c.eval_points, keyspace)


def get_learning_rules(connection):
    if is_iterable(connection.learning_rule):
        return tuple(connection.learning_rule)
    elif connection.learning_rule is not None:
        return (connection.learning_rule,)
    else:
        return ()
