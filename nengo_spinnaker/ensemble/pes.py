import collections
import nengo
import numpy as np

from ..builder import Builder
from ..connection import IntermediateConnection
from ..utils.fixpoint import bitsk
from ..spinnaker import regions


PESInstance = collections.namedtuple('PESInstance', 'learning_rate width')


class IntermediatePESConnection(IntermediateConnection):
    """Intermediate connection representing a PES modulatory connection.
    """
    def __init__(self, pre_obj, post_obj, synapse=None, function=None,
                 transform=1., solver=None, eval_points=None, keyspace=None,
                 is_accumulatory=True, learning_rule=None, modulatory=False,
                 pes_instance=None):
        super(IntermediatePESConnection, self).__init__(
            pre_obj, post_obj, synapse=synapse, function=function,
            transform=transform, solver=solver, eval_points=eval_points,
            keyspace=keyspace, is_accumulatory=is_accumulatory,
            learning_rule=learning_rule, modulatory=modulatory
        )
        self.pes_instance = pes_instance

    @classmethod
    def from_connection(cls, c, pes_instance, keyspace=None,
                        is_accumulatory=True):
        if isinstance(c, nengo.Connection):
            # Get the full transform
            tr = nengo.utils.builder.full_transform(c, allow_scalars=False)

            # Return a copy of this connection but with less information and
            # the full transform.
            return cls(c.pre_obj, c.post_obj, synapse=c.synapse,
                       function=c.function, transform=tr, solver=c.solver,
                       eval_points=c.eval_points, keyspace=keyspace,
                       learning_rule=c.learning_rule, modulatory=c.modulatory,
                       is_accumulatory=is_accumulatory,
                       pes_instance=pes_instance)
        return c


class IntermediatePESModulatoryConnection(IntermediatePESConnection):
    """Intermediate connection representing a PES modulatory connection.
    """
    def get_reduced_incoming_connection(self):
        """Get the reduced incoming connection for the modulatory signal.
        """
        # As in the parent, but with the target port set to the PES instance
        ic = super(IntermediatePESModulatoryConnection, self).\
            get_reduced_incoming_connection()
        ic.target.port = self.pes_instance

        # And with the filter width set to the width of the PES connection
        ic.filter_object.width = self.pes_instance.width

        return ic


@Builder.network_transform
def process_pes_connections(objs, conns, probes):
    """Reroute PES modulatory connections.
    """
    new_conns = list()
    replaced_conns = list()

    for c in conns:
        if not isinstance(c.learning_rule, nengo.PES):
            # Defer dealing with connections that don't have a PES learning
            # rule attached.
            continue

        # This connection has a PES learning rule.  Begin by creating a PES
        # instance that PES connections can refer to.
        pes_instance = PESInstance(c.learning_rule.learning_rate,
                                   c.post_obj.size_in)

        # Create a new connection representing the PES connection itself
        pes_connection = IntermediatePESConnection.from_connection(
            c, pes_instance=pes_instance)

        # Now reroute the modulatory connection
        mod_conn = c.learning_rule.error_connection
        new_mod_conn = IntermediatePESModulatoryConnection.from_connection(
            mod_conn, pes_instance=pes_instance)
        new_mod_conn.post_obj = c.pre_obj

        # Add the new connections and mark which connections we've replaced.
        new_conns.append(pes_connection)
        new_conns.append(new_mod_conn)
        replaced_conns.append(c)
        replaced_conns.append(c.learning_rule.error_connection)

    # Loop over all connections and add those which we haven't already replaced
    new_conns.extend(c for c in conns if c not in replaced_conns)

    return objs, new_conns


def make_pes_region(learning_rules, dt, filter_indices):
    """Create a region containing PES data.

    :param iterable learning_rules: Iterable of the learning rules to construct
        the region for (non PES rules will be ignored).
    :param float dt: dt of simulation.
    :param dict filter_indices: Mapping of connection to filter index.
    """
    # Construct the data as a matrix of the appropriate form
    pes_data = list()

    for l in learning_rules:
        # Only deal with PES rules
        if not isinstance(l.rule, nengo.PES):
            pass

        # Add the data for this learning rule
        pes_data.append([bitsk(l.rule.learning_rate * dt),
                         filter_indices[l.rule.error_connection],
                         l.decoder_index])

    # Convert to appropriate Numpy array and make a region with the number of
    # rows as the first word.
    pes_data = np.array(pes_data, dtype=np.uint32)
    return regions.MatrixRegion(
        pes_data, prepends=[regions.MatrixRegionPrepends.N_ROWS])
