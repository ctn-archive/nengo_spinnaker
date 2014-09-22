"""Build models into intermediate representations which can be simply
converted into PACMAN problem specifications.
"""

import math
import numpy as np

import nengo.utils.builder

import connection
import ensemble
import pes
import probe
import utils


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
            objs, conns, utils.builder.create_replacement_connection)

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
            if not c.keyspace.is_set_o:
                c.keyspace = c.keyspace(o=object_ids[c.pre_obj])

            if not c.keyspace.is_set_i:
                c.keyspace = c.keyspace(i=connection_ids[c])

        # Build the list of output keyspaces for all of the ensemble objects
        # now that we've assigned IDs and keyspaces.
        for obj in objs:
            if isinstance(obj, ensemble.IntermediateEnsemble):
                obj.create_output_keyspaces(object_ids[obj], keyspace)
            if hasattr(obj, 'object_id'):
                obj.object_id = object_ids[obj]

        # Return list of intermediate representation objects and connections
        return objs, conns, keyspace

Builder.register_object_transform(ensemble.build_ensembles)
Builder.register_connectivity_transform(probe.insert_decoded_output_probes)
Builder.register_connectivity_transform(pes.reroute_modulatory_connections)


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
    return utils.keyspaces.create_keyspace(
        'NengoDefault',
        [('x', 1), ('o', bits_o), ('c', bits_c), ('i', bits_i), ('d', bits_d)],
        'xoci', 'xoi'
    )(x=0)


def _get_outgoing_ids(connections):
    """Get the outgoing ID of each connection.
    """
    output_blocks = dict()
    connection_ids = dict()

    # Iterate through the connections building connection blocks where
    # necessary.
    for c in connections:
        if c.pre_obj not in output_blocks:
            output_blocks[c.pre_obj] =\
                (utils.connections.Connections() if not
                 isinstance(c.pre_obj, ensemble.IntermediateEnsemble) else
                 utils.connections.OutgoingEnsembleConnections())
        output_blocks[c.pre_obj].add_connection(c)
        connection_ids[c] = output_blocks[c.pre_obj][c]

    return connection_ids
