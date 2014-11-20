"""Tests for Ensemble build utilities.
"""

import mock
import nengo
import numpy as np

from .. import connections as ensemble_connection_utils
from ...connections.reduced import LowpassFilterParameter, GlobalInhibitionPort


def test_intermediate_global_inhibition_connection():
    """Test that filters are returned correctly with width=1, and that the port
    is the global inhibition port.
    """
    pre_obj = mock.Mock(spec_set=[])
    post_obj = mock.Mock(spec_set=['size_in'])
    post_obj.size_in = 5

    ic = ensemble_connection_utils.IntermediateGlobalInhibitionConnection(
        pre_obj, post_obj, slice(None), slice(None), nengo.Lowpass(0.05))

    f = ic._get_filter()
    assert isinstance(f, LowpassFilterParameter)
    assert f.tau == ic.synapse.tau

    ir = ic.get_reduced_incoming_connection()
    assert (ir.target.port is GlobalInhibitionPort)
    assert ir.filter_object == f

    orc = ic.get_reduced_outgoing_connection()
    assert orc.width == 1


def test_process_global_inhibition_connections():
    """Test that only global inhibition connections are processed."""
    model = nengo.Network()
    with model:
        a = nengo.Node(lambda t: t, size_in=0, size_out=1)
        ens1 = nengo.Ensemble(100, 1)
        ens2 = nengo.Ensemble(100, 1)

        # Create a connection a->e1 and a global inhibition connection e1->e2
        c1 = nengo.Connection(a, ens1)
        c2 = nengo.Connection(ens1, ens2.neurons, transform=[[-10.]]*100)

    # Process these connections and ensure that c1 is left alone and that c2 is
    # replaced
    _, new_conns = ensemble_connection_utils.\
        process_global_inhibition_connections([], [c1, c2], [])

    assert c1 in new_conns
    assert len(new_conns) == 2
    assert c2 not in new_conns
    assert new_conns[-1].pre_obj is ens1
    assert new_conns[-1].post_obj is ens2
    assert np.all(new_conns[-1].transform == c2.transform)
