"""Tests for probing utilities.
"""
import mock
import numpy as np
import pytest
import warnings

import nengo

from nengo_spinnaker import utils


def test_probe_nodes():
    """Test that ProbeNodes are added to the output of Nodes (Pass or
    otherwise) which are to be probed.
    """
    model = nengo.Network()
    with model:
        a = nengo.Node(np.sin)
        
        b = nengo.Ensemble(1, 1)
        c = nengo.Ensemble(1, 1)
        d = nengo.Node(output=None, size_in=1, size_out=1)
        e = nengo.Ensemble(1, 1)

        b_d = nengo.Connection(b, d)
        c_d = nengo.Connection(c, d)
        d_e = nengo.Connection(d, e)

        p_a = nengo.Probe(a)
        p_d = nengo.Probe(d)

    (nodes, connections) = utils.probes.get_probe_nodes_connections(
        model.probes)

    # 1 node and connection each for p_a, p_d
    assert(len(nodes) == 2)
    assert(len(connections) == 2)

    for n in nodes:
        assert(isinstance(n, utils.probes.ProbeNode))
        assert(n.probe == p_a or n.probe == p_d)

    for c in connections:
        if c.pre == a: assert(c.post.probe == p_a)
        if c.pre == d: assert(c.post.probe == p_d)


def test_probenode_spinnaker_build():
    """Tests that a ProbeNode, when built, adds a vertex and a new Probe to the
    Builder.
    """
    builder = mock.Mock()
    builder.probes = list()

    model = nengo.Network()
    with model:
        n = nengo.Node(np.sin)
        p = nengo.Probe(n)

    pn = utils.probes.ProbeNode(p, add_to_container=False)

    # Call the SpiNNaker build function
    pn.spinnaker_build(builder)
    
    assert(builder.add_vertex.call_count == 1)
    v = builder.add_vertex.call_args[0][0]
    assert(isinstance(v, utils.probes.ValueSinkVertex))
    assert(len(builder.probes) == 1)
    assert(builder.probes[0].probe == p)
    assert(builder.probes[0].recording_vertex == v)
    assert(pn.vertex == v)


def test_passnode_modify():
    """Returns a new set of probes where the synapse value of Probes probing
    PassNodes which have synapses on their inputs is set to None.  Additionally,
    functions and transforms on Probe as removed.
    """
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(1, 1)
        pn = nengo.Node(None, size_in=1, size_out=1)
        b = nengo.Ensemble(1, 1)

        nengo.Connection(a, pn, synapse=0.01)
        probe = nengo.Probe(pn, synapse=0.01)
        p2 = nengo.Probe(a, synapse=0.05)

    with warnings.catch_warnings(record=True) as w:
        (objs, conns) = nengo.utils.builder.objs_and_connections(model)
        new_objs = utils.probes.get_corrected_probes(model.probes, conns)

        assert len(w) == 1
        assert(issubclass(w[-1].category, RuntimeWarning))

    assert(len(new_objs) == 2)
    assert(p2 in new_objs)

    for obj in new_objs:
        if isinstance(obj, nengo.Probe) and obj.target == pn:
            assert(obj.conn_args.get('synapse', None) is None)
