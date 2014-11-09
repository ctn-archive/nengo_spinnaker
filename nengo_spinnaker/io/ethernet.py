"""Communicate with running simulations using Ethernet.
"""
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
