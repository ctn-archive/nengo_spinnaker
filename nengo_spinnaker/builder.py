import nengo
from nengo.utils.builder import objs_and_connections

from .connection import IntermediateConnection, build_connection_trees


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
    def build(cls, network, seed):
        """Build the network into an intermediate form.
        """
        # Flatten the network
        (objs, conns) = objs_and_connections(network)

        # Apply all connectivity transforms
        for transform in cls.network_transforms:
            (objs, conns) = transform(objs, conns, network.probes)

        # Convert all remaining Nengo connections into their intermediate
        # forms.  Some Nengo connections may have already been replaced by one
        # of the connectivity transforms.
        conns = _convert_remaining_connections(conns)

        # Build the connectivity tree
        c_trees = build_connection_trees(conns)

        # From this build the keyspace
        default_keyspace = _build_keyspace(c_trees)

        # Assign this keyspace to all connections which have no keyspace
        connection_trees = _apply_default_keyspace(default_keyspace, c_trees)

        # Replace all objects
        connected_objects = get_objects_from_connection_trees(c_trees)
        for obj in connected_objects:
            c_trees = _replace_object_in_connection_trees(
                obj, cls.build_obj(obj, c_trees, rng))

        # Return the connection tree
        return c_trees


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
    raise NotImplementedError


def _apply_default_keyspace(keyspace, connection_tree):
    """Apply a keyspace taken from a given template to all connections without.

    Returns a copy of the given connection tree with all connections that had
    None as their keyspace replaced with new keyspaces derived from the
    default.  This should be the last time that keyspaces are modified.
    """
    raise NotImplementedError
