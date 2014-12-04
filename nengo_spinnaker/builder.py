import logging
import math
import nengo
from nengo.utils.builder import objs_and_connections, remove_passthrough_nodes
import numpy as np

from .connections.intermediate import IntermediateConnection
from .connections.connection_tree import ConnectionTree
from .spinnaker.keyspaces import create_keyspace
from .utils import builder as builder_utils

logger = logging.getLogger(__name__)


class Builder(object):
    network_transforms = list()
    object_transforms = dict()

    @classmethod
    def add_network_transform(cls, transform):
        """Register a new network transform.

        A network transform accepts as parameters a list of objects, a list of
        connections and a list of probes.  It is expected to return a new list
        of objects and a new list of connections.
        """
        cls.network_transforms.append(transform)

    @classmethod
    def add_object_builder(cls, object_type, builder):
        """Register a new object builder.
        """
        cls.object_transforms[object_type] = builder

    @classmethod
    def network_transform(cls, f):
        """Decorator that adds the function as a network transform.
        """
        cls.add_network_transform(f)
        return f

    @classmethod
    def build_obj(cls, obj, connection_trees, config, rngs):
        """Build a given object, or if no builder exists return it unchanged.
        """
        # Work through the MRO to find a build function
        for c in obj.__class__.__mro__:
            if c in cls.object_transforms:
                return cls.object_transforms[c](
                    obj, connection_trees, config, rngs)
        else:
            # Otherwise just return the object unchanged
            return obj

    @classmethod
    def object_builder(cls, object_type):
        """Decorator that adds the function as an object builder.
        """
        def obj_builder(f):
            cls.add_object_builder(object_type, f)
            return f
        return obj_builder

    @classmethod
    def build(cls, network, config=None):
        """Build the network into an intermediate form.

        Parameters
        ----------
        network : nengo.Network
            A network to transform and build into an intermediate form
            (represented through a connection tree).
        config : .config.Config
            Specific Nengo/SpiNNaker configuration options.

        Returns
        -------
        nengo_spinnaker.connections.connection_tree.ConnectionTree
            A ``ConnectionTree`` representing an intermediate form of the
            network.
        dict
            A dictionary mapping original network objects to random number
            generators.

        See Also
        --------
        ConnectionTree : How connection trees represent the objects and
            connectivity of a Nengo model when it is simulated on the SpiNNaker
            platform.
        """
        # Flatten the network
        logger.info("Build step 1/8: Flattening network hierarchy")
        (objs, conns) = objs_and_connections(network)

        # Create seeds for all objects
        rng = np.random.RandomState(_get_seed(network, np.random))
        rngs = {obj: np.random.RandomState(_get_seed(obj, rng)) for obj in
                objs}

        # Apply all connectivity transforms
        logger.info("Build step 2/8: Applying network transforms")
        for transform in cls.network_transforms:
            logger.debug("Applying network transform {}".format(transform))
            (objs, conns) = transform(objs, conns, network.probes, rngs)

        # Convert all remaining Nengo connections into their intermediate
        # forms.  Some Nengo connections may have already been replaced by one
        # of the connectivity transforms.
        logger.info("Build step 3/8: Replacing connections")
        conns = _convert_remaining_connections(conns)

        # Build the connectivity tree, by this point there should be NO
        # instances of Node -> Node connections present in the connection list:
        # if there are, they will be removed now.  Any Node -> Node connections
        # which are required should have been modified during the network
        # transformation stage, it is assumed that any remaining Node -> Node
        # connections may be simulated on host rather than on SpiNNaker.
        logger.info("Build step 4/8: Building connectivity tree")
        c_trees = ConnectionTree.from_intermediate_connections(
            c for c in conns if not (isinstance(c.pre_obj, nengo.Node) and
                                     isinstance(c.post_obj, nengo.Node))
        )

        for c in [c for c in conns if isinstance(c.pre_obj, nengo.Node) and
                  isinstance(c.post_obj, nengo.Node)]:
            logger.info('Connection {} will be simulated on host.'.format(c))

        # From this build the keyspace
        logger.info("Build step 5/8: Generating default keyspace")
        default_keyspace = _build_keyspace(c_trees)

        # Assign this keyspace to all connections which have no keyspace
        logger.info("Build step 6/8: Applying default keyspace")
        c_trees = c_trees.get_new_tree_with_applied_keyspace(default_keyspace)

        # Build all objects
        logger.info("Build step 7/8: Building objects")
        connected_objects = c_trees.get_objects()

        replaced_objects = dict()

        for obj in connected_objects:
            logger.debug("Building {}".format(obj))
            replaced_objects[obj] = cls.build_obj(obj, c_trees, config, rngs)

        # Replace built objects in the connection tree
        logger.info("Build step 8/8: Add built objects to connectivity tree")
        c_trees = c_trees.get_new_tree_with_replaced_objects(replaced_objects)

        # Return the connection tree and the random number generators
        return c_trees, rngs


# Register a new network transform that removes all passthrough Nodes
@Builder.network_transform
def remove_passnodes(objs, conns, probes, rngs):
    return remove_passthrough_nodes(
        objs, conns, builder_utils.create_replacement_connection)


# Utility functions used by the Builder
def _convert_remaining_connections(connections):
    """Replace all Nengo Connections with IntermediateConnections.
    """
    new_conns = list()

    # Replace connections with intermediate connections where appropriate
    for c in connections:
        if isinstance(c, nengo.Connection):
            c = IntermediateConnection.from_connection(c)
        new_conns.append(c)

    return new_conns


def _build_keyspace(connection_tree, subobject_bits=7):
    """Build a keyspace from the given connectivity trees.

    Determines the minimum number of bits required to represent the object,
    connection and dimension fields in the keyspace.  A new keyspace with
    these field widths is returned.

    The new keyspace will take the form of:
        x|o|s|i|d
    Where:
     - `x` indicates that a packet is to be routed off board to a robot or to
       the host via. FPGA.
     - `o` indicates the originating object ID.
     - `s` indicates the index of the partition of the originating object, used
       only in routing.
     - `i` indicates the connection index.
     - `d` indicates the index of the represented component.
    """
    # Get the number of bits required for each field; all these calculations
    # assume that a field cannot have length 0, relax this requirement.
    # TODO Exclude connections that already have keyspaces from these
    # calculations.
    objs = connection_tree.get_objects()
    num_o = 0
    max_i = 0
    max_d = 0

    for o in objs:
        out_conns = connection_tree.get_outgoing_connections(o)
        if len(out_conns) > 0:
            num_o += 1
            max_i = max(max_i, len(out_conns))
            max_d = max(max_d, max(c.width for c in out_conns))

    x_bits = 1
    o_bits = int(math.log(num_o + 1, 2))
    s_bits = subobject_bits
    i_bits = int(math.log(max_i + 1, 2))
    d_bits = int(math.log(max_d + 1, 2))

    padding = 32 - sum([x_bits, o_bits, s_bits, i_bits, d_bits])

    # Create the keyspace
    return create_keyspace(
        'NengoDefault', [('x', x_bits),   # Routed to external device
                         ('o', o_bits),   # Sending object ID
                         ('s', s_bits),   # Sending sub-object ID
                         ('_', padding),  # Padding
                         ('i', i_bits),   # Connection ID
                         ('d', d_bits)],  # Component index (dimension)
        'xosi',          # Fields used in routing
        'xoi'            # Fields used in filter routing
    )


def _get_seed(obj, rng):
    # Copy of function from Nengo reference.
    seed = rng.randint(np.iinfo(np.int32).max)
    return getattr(obj, 'seed', seed)
