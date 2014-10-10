import mock
import numpy as np

import nengo
from .. import nodes as nodes_utils


def test_output_to_board_node_simple():
    io = mock.Mock()
    m = nengo.Node(output=None, size_out=5, add_to_container=False)
    n = nodes_utils.create_output_node(m, io)

    output = np.random.uniform(-1, 1, 5)
    n.output(0.5, output)

    assert(io.set_node_output.call_count == 1)
    for call in io.set_node_output.call_args_list:
        assert(call[0][0] == m)
        assert(np.all(call[0][1] == output))


def test_input_from_board_node_simple():
    io = mock.Mock()
    io.get_node_input.return_value = np.zeros(5)

    m = nengo.Node(output=None, size_in=5, add_to_container=False)
    n = nodes_utils.create_input_node(m, io)

    input_to_m = n.output(0.5)

    assert(io.get_node_input.call_count == 1)
    io.get_node_input.assert_called_with(m)
    assert(np.all(np.zeros(5) == input_to_m))


def test_input_from_board_node_simple_2():
    io = mock.Mock()
    io.get_node_input.return_value = None

    m = nengo.Node(output=None, size_in=5, add_to_container=False)
    n = nodes_utils.create_input_node(m, io)

    input_to_m = n.output(0.5)

    assert(io.get_node_input.call_count == 1)
    io.get_node_input.assert_called_with(m)
    assert(np.all(np.zeros(5) == input_to_m))


def test_replace_node_x_connections():
    """Checks that connections from Nodes to ANY OTHER objects are replaced
    with a new OutputNode and a connection from the Node to that OutputNode.
    """
    model = nengo.Network()
    with model:
        a = nengo.Node(lambda t: t, size_in=0, size_out=1)
        b = nengo.Ensemble(100, 1)

        c = nengo.Connection(a, b)

    io = mock.Mock()
    (new_nodes, new_conns) = nodes_utils.replace_node_x_connections([c], io)

    # Should be 1 new Node and 1 new Connection
    assert len(new_nodes) == 1
    assert len(new_conns) == 1
    assert isinstance(new_nodes[0].output, nodes_utils.OutputToBoard)
    assert new_nodes[0].output.io is io
    assert new_conns[0].pre_obj is a
    assert new_conns[0].post_obj is new_nodes[0]


def test_replace_node_x_connections_k():
    """Test that constant nodes do not result in new nodes or connections.
    """
    model = nengo.Network()
    with model:
        a = nengo.Node(1.0)
        b = nengo.Ensemble(100, 1)

        c = nengo.Connection(a, b)

    io = mock.Mock()
    (new_nodes, new_conns) = nodes_utils.replace_node_x_connections([c], io)

    # Should be no new Nodes or Connections
    assert len(new_nodes) == 0
    assert len(new_conns) == 0


def test_replace_x_node_connections():
    """Test that connections to Nodes from ANY OTHER objects are replaced with
    a new InputNode and a connection from the InputNode to the Node.
    """
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(100, 1)
        b = nengo.Node(lambda t, x: None, size_in=1, size_out=0)

        c = nengo.Connection(a, b)

    io = mock.Mock()
    (new_nodes, new_conns) = nodes_utils.replace_x_node_connections([c], io)

    # Should be one new nodes and one connection
    assert len(new_nodes) == 1
    assert isinstance(new_nodes[0].output, nodes_utils.InputFromBoard)
    assert new_nodes[0].output.io is io
    assert len(new_conns) == 1
    assert new_conns[0].pre_obj is new_nodes[0]
    assert new_conns[0].post_obj is b
