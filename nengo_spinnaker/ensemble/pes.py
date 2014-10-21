import nengo
import numpy as np

from ..connection import IntermediateConnection
from ..utils.fixpoint import bitsk
from . import connections as ens_conn_utils
from ..spinnaker import regions


def reroute_modulatory_connections(objs, connections, probes):
    new_objs = list(objs)
    new_connections = list()

    # Loop through connections and their associated learning rules
    replaced_connections = list()
    for c in connections:
        intermediate_c = None
        replaced_learning_rules = list()

        for l in ens_conn_utils.get_learning_rules(c):
            # If learning rule is PES
            if isinstance(l, nengo.PES):
                # Create an intermediate connection
                # To replace error connection
                e = IntermediateConnection.from_connection(l.error_connection)

                # Reroute this so it terminates at connection's pre-object
                e.post_obj = c.pre_obj

                # Add original error connection to list of
                # Connections that have been replaced
                replaced_connections.append(l.error_connection)

                # Add error connection to output
                new_connections.append(e)

                # If intermediate version of connection hasn't been created
                # Create one from c and wipe its exiting list of learning rules
                if intermediate_c is None:
                    intermediate_c = IntermediateConnection.from_connection(c)
                    intermediate_c.learning_rule = list()

                # Create new learning rule using intermediate error connection
                intermediate_c.learning_rule.append(
                    nengo.PES(e, l.learning_rate))

                # Add original learning rule to list list
                # Of learning rules that have been replaced
                replaced_learning_rules.append(l)

        # If this connection's been replaced
        if intermediate_c is not None:
            # Add learning rules from original connection that
            # Haven't been replaced to intermediate connection
            intermediate_c.learning_rule.extend(
                [l for l in ens_conn_utils.get_learning_rules(c)
                    if l not in replaced_learning_rules])

            # Add original to list
            replaced_connections.append(c)

            # Add intermediate connection to output
            new_connections.append(intermediate_c)

    # Add connections from original list that
    # Haven't been replaced to output list
    new_connections.extend(
        [c for c in connections if c not in replaced_connections])

    # Return new lists
    return new_objs, new_connections


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
                         p.decoder_index])

    # Convert to appropriate Numpy array and make a region with the number of
    # rows as the first word.
    pes_data = np.array(pes_data, dtype=np.uint32)
    return regions.MatrixRegion(
        pes_data, prepends=[regions.MatrixRegionPrepends.N_ROWS])
