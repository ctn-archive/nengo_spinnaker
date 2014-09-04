import numpy as np

import nengo


def create_host_network(connections, io, config=None):
    """Create a network of Nodes for simulation on the host.

    :returns: A Network with nothing but Nodes, all Node->x or x->Node
              connections replaced with connection to/from Nodes which handle
              IO with the SpiNNaker board.
    """
    new_network = nengo.Network()

    # Replace Node -> x connections
    (ns, conns) = replace_node_x_connections(connections, io, config)

    # Replace x -> Node connections
    (ns, conns) = replace_x_node_connections(conns, io)

    # Remove x -> x connections
    conns = [c for c in conns if (isinstance(c.pre_obj, nengo.Node) and
                                  isinstance(c.post_obj, nengo.Node))]

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
        if c.pre_obj not in nodes and isinstance(c.pre_obj, nengo.Node):
            nodes.append(c.pre_obj)
        if c.post_obj not in nodes and isinstance(c.post_obj, nengo.Node):
            nodes.append(c.post_obj)

    return nodes


def replace_node_x_connections(connections, io, config=None):
    """Returns a list of new Nodes to add to the model, and the modified list
    of Connections.

    Every Node->x connection is replaced with a Node->OutputNode where
    appropriate (i.e., output not constant nor function of time).
    """
    new_conns = list()
    new_nodes = list()

    for c in connections:
        if (isinstance(c.pre_obj, nengo.Node) and
                not isinstance(c.post_obj, nengo.Node)):
            # Create a new output node if the output is callable and not a
            # function of time (only).
            if callable(c.pre_obj.output) and (
                    config is None or not config[c.pre_obj].f_of_t):
                n = create_output_node(c.pre_obj, io)

                # Create a new Connection: transforms, functions and filters
                # are handled elsewhere
                c_ = nengo.Connection(c.pre_obj, n, add_to_container=False)

                new_nodes.append(n)
                new_conns.append(c_)
        else:
            new_conns.append(c)

    return (new_nodes, new_conns)


def replace_x_node_connections(connections, io):
    """Returns a list of new Nodes to add to the model, and the modified list
    of Connections.

    Every x->Node connection is replaced with a InputNode->Node.
    """
    new_conns = list()
    new_nodes = list()

    for c in connections:
        if (not isinstance(c.pre_obj, nengo.Node) and
                isinstance(c.post_obj, nengo.Node)):
            # Create a new input node
            n = create_input_node(c.post_obj, io)
            c_ = nengo.Connection(n, c.post_obj, add_to_container=False)

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

    def __call__(self, t, vs):
        self.io.set_node_output(self.node, vs)


class InputFromBoard(object):
    def __init__(self, represent_node, io):
        self.node = represent_node
        self.io = io

    def __call__(self, t):
        ins = self.io.get_node_input(self.node)
        if ins is None:
            return np.zeros(self.node.size_in)
        return ins
