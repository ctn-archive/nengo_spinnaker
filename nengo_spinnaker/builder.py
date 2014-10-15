"""Build models into intermediate representations which can be simply
converted into PACMAN problem specifications.
"""

import math
import numpy as np

import nengo.utils.builder

import connection
import ensemble.intermediate
import ensemble.connections as ens_conn_utils

import utils.builder as builder_utils
import utils.connections as connection_utils
import spinnaker.keyspaces


class Builder(object):
    pre_rpn_transforms = list()  # Network transforms which alter connectivity
    post_rpn_transforms = list()  # Network transforms which alter objects

    @classmethod
    def register_connectivity_transform(cls, func):
        """Add a new network transform to the builder."""
        cls.pre_rpn_transforms.append(func)

    @classmethod
    def register_object_transform(cls, func):
        """Add a new network transform to the builder."""
        cls.post_rpn_transforms.append(func)

    @classmethod
    def build(cls, network, dt, seed):
        """Build an intermediate representation of a Nengo model which can be
        assembled to form a PACMAN problem graph.
        """
        # Flatten the network
        (objs, conns) = nengo.utils.builder.objs_and_connections(network)

        # Generate a RNG
        rng = np.random.RandomState(seed)

        # Apply all network transforms which modify connectivity, they should
        # occur before removing pass through nodes
        for transform in cls.pre_rpn_transforms:
            (objs, conns) = transform(objs, conns, network.probes)

        # Remove pass through nodes
        (objs, conns) = nengo.utils.builder.remove_passthrough_nodes(
            objs, conns, builder_utils.create_replacement_connection)

        # Replace all connections with fully specified equivalents
        new_conns = list()
        for c in conns:
            new_conns.append(
                connection.IntermediateConnection.from_connection(c))
        conns = new_conns

        # Apply all network transforms which modify/replace network objects
        for transform in cls.post_rpn_transforms:
            (objs, conns) = transform(objs, conns, network.probes, dt, rng)

        # Assign an ID to each object
        object_ids = dict([(o, i) for i, o in enumerate(objs)])

        # Create the keyspace for the model
        keyspace = _create_keyspace(conns)

        # Assign the keyspace to the connections, drill down as far as possible
        connection_ids = _get_outgoing_ids(conns)
        for c in conns:
            # Assign the keyspace if one isn't already set
            if c.keyspace is None:
                c.keyspace = keyspace()

            # Set fields within the keyspace
            if not c.keyspace.is_set_i:
                c.keyspace = c.keyspace(o=object_ids[c.pre_obj])
                c.keyspace = c.keyspace(i=connection_ids[c])

        # Build the list of output keyspaces for all of the ensemble objects
        # now that we've assigned IDs and keyspaces.
        for obj in objs:
            if isinstance(obj, ensemble.intermediate.IntermediateEnsemble):
                obj.create_output_keyspaces(object_ids[obj], keyspace)

        # Return list of intermediate representation objects and connections
        return objs, conns, keyspace


def _create_keyspace(connections):
    """Create the minimum keyspace necessary to represent the connection set.
    """
    # Get connection IDs
    max_o = len(set([c.pre_obj for c in connections]))
    max_i = max([i for i in _get_outgoing_ids(connections).values()])
    max_d = max([c.width for c in connections])

    # Get the number of bits necessary for these
    (bits_o, bits_i, bits_d) = [int(math.ceil(math.log(v + 1, 2)))
                                for v in [max_o, max_i, max_d]]
    bits_c = 7  # Allow for 128 splits.  TODO Revise this

    # Ensure that these will fit within a 32-bit key
    padding = 32 - (1 + bits_o + bits_i + bits_d + bits_c)
    assert padding >= 0
    bits_o += padding

    # Create the keyspace
    return spinnaker.keyspaces.create_keyspace(
        'NengoDefault',
        [('x', 1), ('o', bits_o), ('c', bits_c), ('i', bits_i), ('d', bits_d)],
        'xoci', 'xoi'
    )(x=0)


def _get_outgoing_ids(connections):
    """Get the outgoing ID of each connection.

    Returns a dictionary mapping connection to ID.
    """
    connection_ids = dict()

    # Group connections by their originating object
    _, outgoing = nengo.utils.builder.find_all_io(connections)
    for (obj, conns) in outgoing.iteritems():
        # Get the connection indices for this set of connections
        if isinstance(obj, (nengo.Ensemble,
                            ensemble.intermediate.IntermediateEnsemble)):
            _, conn_map = ens_conn_utils.\
                get_combined_outgoing_ensemble_connections(conns)
        else:
            _, conn_map = connection_utils.get_combined_connections(conns)

        # Include these connection indices
        connection_ids.update(conn_map)

    return connection_ids
