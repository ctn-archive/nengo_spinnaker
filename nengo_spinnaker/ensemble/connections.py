"""Connection utilities required solely by Ensembles.
"""

import nengo
import numpy as np

from nengo.utils.compat import is_iterable

from ..connections.intermediate import IntermediateConnection
from ..connections.reduced import GlobalInhibitionPort


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


class IntermediateGlobalInhibitionConnection(IntermediateConnection):
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

    def get_reduced_outgoing_connection(self):
        """Get the reduced outgoing connection representing this connection.
        """
        # Get the standard reduced outgoing connection
        oc = super(IntermediateGlobalInhibitionConnection, self).\
            get_reduced_outgoing_connection()

        # Swap out the width of the connection to be 1.
        oc.width = 1

        return oc

    def get_reduced_incoming_connection(self):
        """Get the reduced connection representing this connection.
        """
        # Get the standard reduced incoming connections
        ic = super(IntermediateGlobalInhibitionConnection, self).\
            get_reduced_incoming_connection()

        # Swap out the port on the receiving object so that it points at the
        # global inhibition port.
        ic.target.port = GlobalInhibitionPort

        return ic

    def _get_filter(self):
        """Return the filter required by the connection."""
        # Use the given filter but modify the width to match the required port
        # size.
        f = super(IntermediateGlobalInhibitionConnection, self)._get_filter()
        return f


def get_learning_rules(connection):
    if is_iterable(connection.learning_rule_type):
        return tuple(connection.learning_rule_type)
    elif connection.learning_rule_type is not None:
        return (connection.learning_rule_type,)
    else:
        return ()
