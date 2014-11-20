"""Communicate with running simulations using Ethernet.
"""
import collections
import nengo

from .sdp_rx_vertex import TransmitObject
from .sdp_tx_vertex import ReceiveObject


class EthernetIO(object):
    """Perform communication with running simulations using Ethernet.
    """
    @staticmethod
    def prepare_connection_tree(connection_tree):
        """Prepare the network for simulation using Ethernet IO.

        :param connection_tree: A connection tree representing the model to
            simulate.
        :type connection_tree:
            :py:class:`~..connections.connection_tree.ConnectionTree`
        :returns: A new connection tree with Nodes replaced with objects to
            transmit and receive packets on behalf of Nodes.
        :rtype:
            :py:class:`~..connections.connection_tree.ConnectionTree`
        """
        # Get all objects, and then filter for ones which are Nodes.
        # It is important that any Nodes which require special treatment are
        # replaced correctly in some earlier network or object transform.
        nodes = [obj for obj in connection_tree.get_objects() if
                 isinstance(obj, nengo.Node)]

        # Create transmit and receive objects for each node
        transmitters = {node: TransmitObject(node) for node in nodes}
        receivers = {node: ReceiveObject(node) for node in nodes}

        # Replace these objects in the connection tree
        connection_tree = connection_tree.get_new_tree_with_replaced_objects(
            transmitters, replace_when_terminating=False)
        connection_tree = connection_tree.get_new_tree_with_replaced_objects(
            receivers, replace_when_originating=False)

        # Return the new connection tree
        return connection_tree

    def __init__(self, placed_vertices, connection_tree):
        """Create a new EthernetIO object to manage communication with a
        simulation running on a SpiNNaker machine using Ethernet.

        :type placed_subvertices: iterable of
            :py:class:`~nengo_spinnaker.spinnaker.vertices.PlacedVertex`
        :type connection_tree:
            :py:class:`~..connections.connection_tree.ConnectionTree`
        """
        # Build a map of Node/Slice to Core
        self.node_slice_to_core_map = get_output_node_slices_to_core_map(
            placed_vertices)

        # Build a map of Core to Node/Slice
        self.core_to_node_slice_map = get_input_core_to_node_slice_map(
            placed_vertices)

        # Build an output buffer for each Node we know something about
        raise NotImplementedError

        # Build an input buffer for each Node we know something about
        raise NotImplementedError

    def start(self):
        """Start communicating with the simulation over ethernet.
        """
        raise NotImplementedError

    def stop(self):
        """Stop communicating with the simulation over ethernet.
        """
        raise NotImplementedError

    def set_node_output(self, node, output):
        """Set the current output value of a Node.

        :type node: :py:class:`~nengo.Node`
        :type output: :py:class:`~numpy.array`
        :raises: :py:exc:`KeyError` if the node is not recognised.
        """
        raise NotImplementedError

    def get_node_input(self, node):
        """Get the current input for a node.

        :type node: :py:class:`~nengo.Node`
        :returns: The current input as a Numpy array, or None if there is no
            complete input for this node.
        :rtype output: :py:class:`~numpy.array` or None
        :raises: :py:exc:`KeyError` if the node is not recognised.
        """
        raise NotImplementedError


PlacedSlice = collections.namedtuple('PlacedSlice', 'slice x y p')


def get_output_node_slices_to_core_map(placed_subvertices):
    """Create a map of Nodes to output slices to cores on a SpiNNaker machine.

    :type placed_subvertices: :py:func:`dict` mapping vertices to
        :py:class:`~nengo_spinnaker.spinnaker.vertices.PlacedVertex`
    :returns: A dictionary, the keys of which are :py:class:`~nengo.Node`, the
        values of which are :py:class:`PlacedSlice`s.
    :rtype: :py:func:`dict`
    """
    raise NotImplementedError


NodeInputSlice = collections.namedtuple('NodeInputSlice', 'node slice')


def get_input_core_to_node_slice_map(placed_subvertices):
    """Create a map of cores to input slices of Nodes.

    :type placed_subvertices: :py:func:`dict` mapping vertices to
        :py:class:`~nengo_spinnaker.spinnaker.vertices.PlacedVertex`
    :returns: A dictionary with core `(x, y, p)` as keys and tuples of
        :py:class:`~nengo.Node`s and :py:func:`slice`s as values.
    :rtype: :py:func:`dict`
    """
    raise NotImplementedError
