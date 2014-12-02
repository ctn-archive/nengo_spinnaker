import logging
import math
import nengo
from nengo.utils.builder import objs_and_connections, remove_passthrough_nodes
import numpy as np

from .connections.intermediate import IntermediateConnection
from .connections.connection_tree import ConnectionTree
from .spinnaker.keyspaces import Keyspace
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
        cls.network_transforms.insert(0, transform)

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
    def build(cls, network, keyspace, config=None):
        """Build the network into an intermediate form.

        Parameters
        ----------
        network : nengo.Network
            A network to transform and build into an intermediate form
            (represented through a connection tree).
        config : .config.Config
            Specific Nengo/SpiNNaker configuration options.
        keyspace : :py:class:`~.spinnaker.keyspaces.Keyspace`
            The Keyspace to use.

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
            c for c in conns if (not isinstance(c.pre_obj, nengo.Node) and
                                 not isinstance(c.post_obj, nengo.Node))
        )

        for c in [c for c in conns if not(isinstance(c.pre_obj, nengo.Node) or
                                          isinstance(c.post_obj, nengo.Node))]:
            logger.info('Connection {} will be simulated on host.'.format(c))

        # From this build the keyspace
        logger.info("Build step 5/8: Adding fields to keyspace")
        _add_nengo_keyspace_fields(keyspace)
        nengo_keyspace = keyspace(n_system=0)

        # Assign this keyspace to all connections which have no keyspace
        logger.info("Build step 6/8: Applying default keyspace")
        c_trees = c_trees.get_new_tree_with_applied_keyspace(nengo_keyspace)

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


def _add_nengo_keyspace_fields(ks):
    """Add the standard fields used by Nengo simulations to the keyspace.

    The following fields (prefixed with `n_` will be added to the Keyspace:
    - `n_system` indicates that the packet is for system control purposes
        - If `n_system` is 0:
            - `n_object` indicates the originating object ID.
            - `n_subobject` indicates the index of the partition of the
              originating object, used only in routing.
            - `n_connection` indicates the connection index.
            - `n_dimension` indicates the index of the represented component.
        - If `n_system` is 1:
            - `n_system_object` indicates the target object ID.
            - `n_system_subobject` indicates the index of the partition of
              the target object, used only in routing.
            - `n_system_command` indicates the command to execute

    The following fields are tagged `n_routing`:
    - `n_object`
    - `n_subobject`
    - `n_connection`
    - `n_system_object`
    - `n_system_subobject`

    The following fields are tagged `n_filter_routing`:
    - `n_object`
    - `n_connection`

    Note that the tags propagate up the keyspace's hierarchy (e.g. `n_system`
    will be tagged as both `n_routing` and `n_filter_routing`).
    """
    ks.add_field("n_external", length=1, start_at=31)
    
    ks_internal = ks(n_external=0)
    ks_internal.add_field("n_system")
    
    ks_nengo = ks_internal(n_system=0)
    ks_nengo.add_field("n_object", tags="n_routing n_filter_routing")
    ks_nengo.add_field("n_subobject", tags="n_routing")
    ks_nengo.add_field("n_connection", tags="n_routing n_filter_routing")
    ks_nengo.add_field("n_dimension")
    
    ks_system = ks_internal(n_system=1)
    ks_system.add_field("n_system_object", tags="n_routing")
    ks_system.add_field("n_system_subobject", tags="n_routing")
    ks_system.add_field("n_system_command")


def _get_seed(obj, rng):
    # Copy of function from Nengo reference.
    seed = rng.randint(np.iinfo(np.int32).max)
    return getattr(obj, 'seed', seed)
