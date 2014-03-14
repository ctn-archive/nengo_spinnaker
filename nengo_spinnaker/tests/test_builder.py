"""Tests for the nengo_spinnaker Builder.
"""

import pytest
import nengo
import numpy as np

from .. import builder, ensemble_vertex

def test_build_fail():
    """Ensure that the Builder raises an error when we try to build something
    that we don't know anything about.
    """
    b = builder.Builder()

    x = nengo.Model("Test")
    x.objs.append(1.0)

    with pytest.raises(TypeError):
        b(x, 0.001)

def test_build_ensemble():
    """Ensure that the Builder returns a DAO containing the given Ensemble.
    """
    # Construct and build the simple model
    model = nengo.Model("Test")
    with model:
        ens = nengo.Ensemble(100, 2)

    b = builder.Builder()
    dao = b(model, dt=0.001)

    # Test that the DAO contains a vertex which represents the given Ensemble
    assert(isinstance(dao.vertices[0], ensemble_vertex.EnsembleVertex))
    assert(dao.ensemble_vertices[ens]._ens == ens)

def test_build_node_ensemble_ensemble_node():
    """Test that the Builder can build a simple network."""
    # Construct and build a simple model
    model = nengo.Model("Test")
    d = 5
    with model:
        def printout(t, x):
            print t, x

        n1 = nengo.Node(np.sin, label='input')
        n2 = nengo.Node(printout, size_in=d, label='output')

        e_a = nengo.Ensemble(90, d)
        e_b = nengo.Ensemble(50, d)

        nengo.Connection(n1, e_a, filter=0.001, transform=[[1]]*d)
        nengo.Connection(e_a, e_b, filter=0.002)
        nengo.Connection(e_b, n2, filter=0.001)

    b_ = builder.Builder()
    dao = b_(model, 0.001)

    # Test that inputs are assigned to Rxs
    assert(dao.rx_assigns)
    rx = dao.rx_assigns[n1]
    assert(n1 in list(rx.nodes))

    # Test that outputs are assigned to Txs
    assert(dao.tx_assigns)
    tx = dao.tx_assigns[n2]
    assert(n2 in list(tx.nodes))

    # Test that two ensembles exist and that they have appropriate connections
    # to the Rx and Tx components
    assert(rx.out_edges)
    assert(
        isinstance(rx.out_edges[0].postvertex, ensemble_vertex.EnsembleVertex)
    )
    ev1 = rx.out_edges[0].postvertex

    assert(
        isinstance(tx.in_edges[0].prevertex, ensemble_vertex.EnsembleVertex)
    )
    ev2 = tx.in_edges[0].prevertex

    assert(ev1.out_edges[0].postvertex == ev2)
