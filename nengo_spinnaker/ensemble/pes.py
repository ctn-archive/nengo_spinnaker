import collections
import nengo
import numpy as np
from six import iteritems

from ..connections.intermediate import IntermediateConnection
from ..connections.reduced import OutgoingReducedEnsembleConnection
from ..utils import filters as filter_utils
from ..utils.fixpoint import bitsk
from ..spinnaker import regions


PESInstance = collections.namedtuple('PESInstance', 'learning_rate width')


class _IntermediatePESConnection(IntermediateConnection):
    """Intermediate connection representing a PES modulatory connection.
    """
    def __init__(self, *args, **kwargs):
        self.pes_instance = kwargs.pop('pes_instance', None)
        super(_IntermediatePESConnection, self).__init__(*args, **kwargs)

    @classmethod
    def from_connection(cls, c, pes_instance, keyspace=None,
                        is_accumulatory=True):
        if isinstance(c, nengo.Connection):
            c = super(_IntermediatePESConnection, cls).from_connection(
                c, keyspace=keyspace, is_accumulatory=is_accumulatory)
        c.pes_instance = pes_instance
        return c


class IntermediatePESConnection(_IntermediatePESConnection):
    """Intermediate connection representing a learnt PES connection."""
    def get_reduced_outgoing_connection(self):
        assert isinstance(self.pre_obj, self._expected_ensemble_type)

        # Get the eval points
        eval_points = self._get_eval_points()

        return OutgoingReducedEnsembleConnection(
            self.width, self.transform, self.function, self.pre_slice,
            self.post_slice, self.keyspace, eval_points, self.solver,
            self.pes_instance
        )


class IntermediatePESModulatoryConnection(_IntermediatePESConnection):
    """Intermediate connection representing a PES modulatory connection.
    """
    def get_reduced_outgoing_connection(self):
        """Get the reduced outgoing connection for the modulatory signal.
        """
        # As in the parent, but set the width to the width of the PES signal
        oc = super(IntermediatePESModulatoryConnection, self).\
            get_reduced_outgoing_connection()
        oc.width = self.pes_instance.width

        return oc

    def get_reduced_incoming_connection(self):
        """Get the reduced incoming connection for the modulatory signal.
        """
        # As in the parent, but with the target port set to the PES instance
        ic = super(IntermediatePESModulatoryConnection, self).\
            get_reduced_incoming_connection()
        ic.target.port = self.pes_instance

        return ic


# This is a Builder network transform, but it is called in
# nengo_spinnaker.ensemble.build
def process_pes_connections(objs, conns, probes):
    """Reroute PES modulatory connections.
    """
    new_conns = list()
    replaced_conns = list()

    for c in conns:
        if not isinstance(c.learning_rule_type, nengo.PES):
            # Defer dealing with connections that don't have a PES learning
            # rule attached.
            continue

        # This connection has a PES learning rule.  Begin by creating a PES
        # instance that PES connections can refer to.
        pes_instance = PESInstance(c.learning_rule_type.learning_rate,
                                   c.post_obj.size_in)

        # Create a new connection representing the PES connection itself
        pes_connection = IntermediatePESConnection.from_connection(
            c, pes_instance=pes_instance)

        # Now reroute the modulatory connection
        mod_conn = c.learning_rule_type.error_connection
        new_mod_conn = IntermediatePESModulatoryConnection.from_connection(
            mod_conn, pes_instance=pes_instance)
        new_mod_conn.post_obj = c.pre_obj

        # Add the new connections and mark which connections we've replaced.
        new_conns.append(pes_connection)
        new_conns.append(new_mod_conn)
        replaced_conns.append(c)
        replaced_conns.append(c.learning_rule_type.error_connection)

    # Loop over all connections and add those which we haven't already replaced
    new_conns.extend(c for c in conns if c not in replaced_conns)

    return objs, new_conns


def make_pes_regions(learning_rules, incoming_connections, dt):
    """Create the filter region, routing region and PES regions for PES
    learning connections.

    Parameters
    ----------
    learning_rules : list
        A list of ``IntermediateLearningRule``s.
    incoming_connections : dict
        A mapping of ports to a map of filters to keyspaces.
    dt : float
        The duration of a simulation timestep.

    Returns
    -------
    tuple
        A tuple consisting of the PES region, the PES filter region and the PES
        filter routing region.
    """
    pes_filters = list()
    widths = list()
    filter_indices = dict()

    # Iterate through the PES connections
    for i, l in enumerate([l for l in learning_rules if
                           isinstance(l.rule, PESInstance)]):
        # Get the incoming connections for this PES instance
        conns = incoming_connections[l.rule]  # Get incoming connections
        assert len(conns) == 1  # There *should* only be one mod connection
        filter_indices[l] = i  # Store the index of this filter

        # Add the filter, keyspaces and width
        for (f, keyspaces) in iteritems(conns):
            pes_filters.append((f, keyspaces))
            widths.append(l.rule.width)

    # Make the regions
    pes_region = make_pes_region(learning_rules, dt, filter_indices)
    pes_filters, pes_routing = filter_utils.get_filter_regions(pes_filters, dt,
                                                               widths)
    return pes_region, pes_filters, pes_routing


def make_pes_region(learning_rules, dt, filter_indices):
    """Create a region containing PES data.

    :param iterable learning_rules: Iterable of the learning rules to construct
        the region for (non PES rules will be ignored).
    :param float dt: dt of simulation.
    :param dict filter_indices: Mapping of connection to filter index.
    """
    # Construct the data as a matrix of the appropriate form
    pes_data = list()

    for l in [l for l in learning_rules if isinstance(l.rule, PESInstance)]:
        # Add the data for this learning rule
        pes_data.append([bitsk(l.rule.learning_rate * dt),
                         filter_indices[l],
                         l.decoder_index])

    # Convert to appropriate Numpy array and make a region with the number of
    # rows as the first word.
    pes_data = np.array(pes_data, dtype=np.uint32)
    return regions.MatrixRegion(
        pes_data, prepends=[regions.MatrixRegionPrepends.N_ROWS])
