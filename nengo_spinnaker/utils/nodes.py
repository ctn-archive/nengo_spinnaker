import numpy as np

import nengo


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

    # Remove Ensemble -> Ensemble connections
    conns = [c for c in conns if (not isinstance(c.pre, nengo.Ensemble) and
                                  not isinstance(c.post, nengo.Ensemble))]

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
    appropriate (i.e., output not constant nor function of time).
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
