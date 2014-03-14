"""Test node_bin utilities.
"""

import nengo
import pytest

from .. import node_bin

def test_simple_assigned_node_bin():
    """Test that an AssignedNodeBin correctly keeps track of its remaining
    dimensions when empty.
    """
    nb = node_bin.AssignedNodeBin(64)
    assert(nb.n_assigned_dimensions == 0)
    assert(nb.remaining_space == 64)

def test_assign_node():
    """Test that a Node can be assigned to an AssignedNodeBin.
    """
    model = nengo.Model("Test")

    with model:
        a = nengo.Node(output=[1,2,3])

    nb = node_bin.AssignedNodeBin(64, lambda n: n.size_out)
    nb.append(a)

    assert(nb.n_assigned_dimensions == 3)

def test_assign_and_return_node():
    """Test that a Nodes may be assigned and return from a bin.
    """
    model = nengo.Model("Test")
    nodes = []

    with model:
        for n in range(10):
            nodes.append(nengo.Node(output=[1,2,3]))

    nb = node_bin.AssignedNodeBin(64, lambda n: n.size_out)

    for n in nodes:
        nb.append(n)

    assert(nb.n_assigned_dimensions == 30)

    for (n, an) in zip(nodes, list(nb.nodes)):
        assert(n == an)

def test_over_assign():
    """Test that it is not possible to append too many dimensions to a bin.
    """
    model = nengo.Model("Test")

    with model:
        a = nengo.Node(output=[1,2,3])

    nb = node_bin.AssignedNodeBin(2, lambda n: n.size_out)

    with pytest.raises(ValueError):
        nb.append(a)
