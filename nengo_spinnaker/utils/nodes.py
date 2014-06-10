"""Utilities for Nodes and Node simulation.
"""
import logging
import numpy as np

import nengo

from .connections import ConnectionBank


logger = logging.getLogger(__name__)


class NodeIO(object):
    """Manage getting and setting input and output for Nodes where that input
    may come from Nodes simulated on the host, or input from a SpiNNaker board
    handled by some IO communicator.
    """
    def __init__(self, node_node_connections, io):
        """Create a new Node Input/Output Handler for the given set of
        Node->Node connections and an external IO handler.
        """
        self.io = io
        self.internode = OutputBuffer(node_node_connections)

    def set_node_output(self, node, value):
        """Set output for the given Node."""
        if self.io.node_has_output(node):
            self.io.set_node_output(node, value)
        self.internode.set_output(node, value)

    def get_node_input(self, node):
        """Get input for the Node.

        :returns: None if the input is incomplete, or a dictionary with
                  Connections as keys and the current sampled value on those
                  Connections as values.  As a special case (pre-filtered)
                  input from the SpiNNaker machine is indexed with None.
        """
        values = {}

        if self.io.node_has_input(node):
            values[None] = self.io.get_node_input(node)
            if values[None] is None:
                return None
        try:
            i_values = self.internode.get_inputs(node)
            if i_values is None:
                return None
            values.update(i_values)
        except KeyError:
            pass

        if len(values) == 0:
            return None

        return values


def build_output_buffers(connection_bank):
    """Build an output buffer of the correct size for each unique output
    combination."""
    return [np.zeros(w) for w in connection_bank.widths]


def get_incoming_object_connections(connections):
    """Return a dictionary mapping from objects to their incoming connections.
    """
    objs_connections = dict()
    for c in connections:
        if c.post not in objs_connections:
            objs_connections[c.post] = list()
        objs_connections[c.post].append(c)
    return objs_connections


class OutputBuffer(object):
    def __init__(self, connections):
        self.bank = ConnectionBank(connections)
        self.buffers = build_output_buffers(self.bank)
        self.written_to = [False for b in self.buffers]
        self.objs_connections = get_incoming_object_connections(connections)

    def get_input(self, obj):
        outputs = self.get_inputs(obj)
        if outputs is None:
            return None
        return sum(outputs.values())

    def get_inputs(self, obj):
        if obj not in self.objs_connections:
            raise KeyError

        outputs = dict()
        for c in self.objs_connections[obj]:
            if not self._output_written_for_connection(c):
                return
            outputs[c] = self.get_output_for_connection(c)
        return outputs

    def set_output(self, obj, value):
        tfs = self.bank.get_transforms_functions_for_object(obj)
        starting_index = self.bank.get_starting_index_for_object(obj)

        for (i, tf) in enumerate(tfs):
            v = np.dot(tf.transform, value if tf.function is None else
                       tf.function(value))
            self.buffers[i+starting_index][:] = v[:]
            self.written_to[i+starting_index] = True

    def get_output_for_connection(self, connection):
        return self.buffers[self.bank[connection]]

    def _output_written_for_connection(self, connection):
        return self.written_to[self.bank[connection]]


def create_host_network(network, io, config=None):
    """Create a network of Nodes for simulation on the host.

    :returns: A Network with no Ensembles, all Node->Ensemble or Ensemble->Node
              connections replaced with connection to/from Nodes which handle
              IO with the SpiNNaker board.  All custom Nodes (those with a
              `spinnaker_build` method) will have been removed.
    """
    new_network = nengo.Network()

    # Remove custom built Nodes
    (ns, conns) = remove_custom_nodes(network.nodes, network.connections)

    # Replace Node -> Ensemble connections
    (ns, conns) = replace_node_ensemble_connections(conns, io, config)

    # Replace Ensemble -> Node connections
    (ns, conns) = replace_ensemble_node_connections(conns, io)

    # Finish up
    new_network.nodes = get_connected_nodes(conns)
    new_network.connections.extend(conns)
    return new_network


def get_connected_nodes(connections):
    """From the connections return a list of Nodes which are are either at the
    beginning or end of a connection.
    """
    nodes = list()

    for c in connections:
        if c.pre not in nodes and isinstance(c.pre, nengo.Node):
            nodes.append(c.pre)
        if c.post not in nodes and isinstance(c.post, nengo.Node):
            nodes.append(c.post)

    return nodes


def remove_custom_nodes(nodes, connections):
    """Remove Nodes with a `spinnaker_build` method and their associated
    connections.
    """
    removed_nodes = list()
    final_nodes = list()
    final_conns = list()

    for n in nodes:
        if hasattr(n, 'spinnaker_build'):
            removed_nodes.append(n)
        else:
            final_nodes.append(n)

    for c in connections:
        if c.pre not in removed_nodes and c.post not in removed_nodes:
            final_conns.append(c)

    return final_nodes, final_conns


def replace_node_ensemble_connections(connections, io, config=None):
    """Returns a list of new Nodes to add to the model, and the modified list
    of Connections.

    Every Node->Ensemble connection is replaced with a Node->OutputNode where
    appropriate (i.e., output not constant nor function of time.
    """
    new_conns = list()
    new_nodes = list()

    for c in connections:
        if (isinstance(c.pre, nengo.Node) and
                isinstance(c.post, nengo.Ensemble)):
            # Create a new output node if the output is callable and not a
            # function of time (only).
            if callable(c.pre.output) and (config is None or
                                           not config[c.pre].f_of_t):
                n = create_output_node(c.pre, io)

                # Create a new Connection: transforms, functions and filters
                # are handled elsewhere
                c_ = nengo.Connection(c.pre, n, add_to_container=False)

                new_nodes.append(n)
                new_conns.append(c_)
        else:
            new_conns.append(c)

    return (new_nodes, new_conns)


def replace_ensemble_node_connections(connections, io):
    """Returns a list of new Nodes to add to the model, and the modified list
    of Connections.

    Every Ensemble->Node connection is replaced with a InputNode->Node.
    """
    new_conns = list()
    new_nodes = list()

    for c in connections:
        if (isinstance(c.pre, nengo.Ensemble) and
                isinstance(c.post, nengo.Node)):
            # Create a new input node
            n = create_input_node(c.post, io)
            c_ = nengo.Connection(n, c.post, add_to_container=False)

            new_nodes.append(n)
            new_conns.append(c_)
        else:
            new_conns.append(c)

    return (new_nodes, new_conns)


def create_output_node(node, io):
    output_node = nengo.Node(
        OutputToBoard(node, io), size_in=node.size_out, size_out=0,
        add_to_container=False, label='OutputNode:%s' % node
    )
    return output_node


def create_input_node(node, io):
    input_node = nengo.Node(
        InputFromBoard(node, io), size_out=node.size_in, size_in=0,
        add_to_container=False, label='InputNode:%s' % node
    )
    return input_node


class OutputToBoard(object):
    def __init__(self, represent_node, io):
        self.node = represent_node
        self.io = io

    def __call__(self, *vs):
        self.io.set_node_output(self.node, vs[1:])


class InputFromBoard(object):
    def __init__(self, represent_node, io):
        self.node = represent_node
        self.io = io

    def __call__(self, t):
        ins = self.io.get_node_input(self.node)
        if ins is None:
            return np.zeros(self.node.size_in)
        return ins
