import collections
import copy
import logging
import math
import nengo
from nengo.utils.builder import objs_and_connections, remove_passthrough_nodes

from .connection import (IntermediateConnection, build_connection_trees,
                         get_objects_from_connection_trees, Target)
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
    def build_obj(cls, obj, *args):
        """Build a given object, or if no builder exists return it unchanged.
        """
        # Work through the MRO to find a build function
        for c in obj.__class__.__mro__:
            if c in cls.object_transforms:
                return cls.object_transforms[c](obj, *args)
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
        """
        # Flatten the network
        logger.info("Build step 1/8: Flattening network hierarchy")
        (objs, conns) = objs_and_connections(network)

        # Apply all connectivity transforms
        logger.info("Build step 2/8: Applying network transforms")
        for transform in cls.network_transforms:
            logger.debug("Applying network transform {}".format(transform))
            (objs, conns) = transform(objs, conns, network.probes)

        # Convert all remaining Nengo connections into their intermediate
        # forms.  Some Nengo connections may have already been replaced by one
        # of the connectivity transforms.
        logger.info("Build step 3/8: Replacing connections")
        conns = _convert_remaining_connections(conns)

        # Build the connectivity tree
        logger.info("Build step 4/8: Building connectivity tree")
        c_trees = build_connection_trees(conns)

        # From this build the keyspace
        logger.info("Build step 5/8: Generating default keyspace")
        default_keyspace = _build_keyspace(c_trees)

        # Assign this keyspace to all connections which have no keyspace
        logger.info("Build step 6/8: Applying default keyspace")
        c_trees = _apply_default_keyspace(default_keyspace, c_trees)

        # Build all objects
        logger.info("Build step 7/8: Building objects")
        connected_objects = get_objects_from_connection_trees(c_trees)
        replaced_objects = dict()

        for obj in connected_objects:
            logger.debug("Building {}".format(obj))
            replaced_objects[obj] = cls.build_obj(
                obj, c_trees, network, config)

        # Replace built objects in the connection tree
        logger.info("Build step 8/8: Add built objects to connectivity tree")
        c_trees = _replace_objects_in_connection_trees(replaced_objects,
                                                       c_trees)

        # Return the connection tree
        return c_trees


# Register a new network transform that removes all passthrough Nodes
@Builder.network_transform
def remove_passnodes(objs, conns, probes):
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
    x_bits = 1
    o_bits = int(math.log(len(connection_tree) + 1, 2))
    s_bits = subobject_bits
    i_bits = int(math.log(max(len(v) for v in connection_tree.items()) + 1, 2))

    # TODO Add width to outgoing reduced connections to make this nicer, rather
    # than relying on the filter width.
    d_bits = int(
        math.log(
            1 + max(k.filter_object.width for i in connection_tree.values() for
                    j in i.values() for k in j),
            2)
    )

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


def _apply_default_keyspace(keyspace, connection_tree):
    """Apply a keyspace taken from a given template to all connections without.

    Returns a copy of the given connection tree with all connections that had
    None as their keyspace replaced with new keyspaces derived from the
    default.  This should be the last time that keyspaces are modified.
    Additionally, any keyspaces that had the object field ('o') unset are
    replaced with a keyspace with this field set appropriately.
    """
    # Create a new connection tree
    tree = collections.defaultdict(lambda: collections.defaultdict(list))

    # Iterate over all objects and outgoing connections
    for i, (obj, oconns) in enumerate(connection_tree.items()):
        # Create the keyspace for connections from this object
        core_ks = keyspace(o=i)

        # Apply a unique keyspace to all connections from this core that don't
        # already have assigned keyspaces.
        for j, c in enumerate(o for o in oconns if o.keyspace is None):
            new_c = copy.copy(c)
            c_ks = core_ks(i=j)
            new_c.keyspace = c_ks

            tree[obj][new_c] = copy.deepcopy(connection_tree[obj][c])

        # Modify all connections that already have keyspaces.
        for j, c in enumerate(o for o in oconns if o.keyspace is not None):
            new_c = copy.copy(c)

            # Add the object ID for the connection if this hasn't already been
            # filled in.
            if not c.keyspace.is_set_o:
                new_c.keyspace = c.keyspace(o=i)

            tree[obj][new_c] = copy.deepcopy(connection_tree[obj][c])

    return tree


def _replace_objects_in_connection_trees(replacements, connection_tree):
    """Return a new connection tree with various objects replaced.

    :param dict replacements: A dict mapping objects to replaced objects.
    :return: A new connectivity tree containing the replaced objects.
    """
    # Create a new connection tree
    tree = collections.defaultdict(lambda: collections.defaultdict(list))

    # Loop over all objects and outgoing connections
    for (obj, oconns) in connection_tree.items():
        # Get the replaced originating object
        new_obj = replacements[obj]

        # Loop over all incoming connections
        for (oc, inconns) in oconns.items():
            for ic in inconns:
                # Copy and replace the incoming connection
                new_ic = copy.copy(ic)
                new_ic.target = Target(replacements[ic.target.target_object],
                                       ic.filter_object)

                tree[new_obj][oc].append(new_ic)

    return tree
