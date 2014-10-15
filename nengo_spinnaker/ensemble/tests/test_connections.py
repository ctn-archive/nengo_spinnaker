"""Tests for Ensemble build utilities.
"""

import nengo
import numpy as np

from .. import connections as ensemble_connection_utils


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


def test_combine_outgoing_ensemble_connections():
    """Test that outgoing ensemble connections are merged correctly."""
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(100, 2)
        b = nengo.Ensemble(100, 2)
        c = nengo.Ensemble(100, 2)

        # Create connections from a
        cs = [
            nengo.Connection(a, b, transform=0.5),
            nengo.Connection(a, c, transform=0.5),
            nengo.Connection(a, b, solver=nengo.solvers.Lstsq()),
            nengo.Connection(a, c, solver=nengo.solvers.LstsqNoise()),
            nengo.Connection(a, b, eval_points=np.zeros((10, 2))),
            nengo.Connection(a, c, eval_points=np.zeros((10, 2))),
            nengo.Connection(
                a, b, eval_points=np.random.uniform(size=(10, 2))),
            nengo.Connection(
                a, c, eval_points=np.random.uniform(size=(10, 2))),
        ]

        # Connections 1 & 2 and 5 & 6 should be shared, all others should be
        # separate.
        combined_connections, connection_map = ensemble_connection_utils.\
            get_combined_outgoing_ensemble_connections(cs)

        assert len(combined_connections) == len(cs) - 2

        # Check 1st and 2nd shared connection
        assert (combined_connections[0].function is
                cs[0].function is
                cs[1].function)
        assert connection_map[cs[0]] == connection_map[cs[1]] == 0
        assert np.all(combined_connections[0].transform == cs[0].transform)
        assert combined_connections[0].solver is cs[0].solver is cs[1].solver

        # Check 3rd, 4th connections
        assert combined_connections[1].solver is cs[2].solver
        assert combined_connections[2].solver is cs[3].solver

        # Check shared 5th and 6th connections
        assert connection_map[cs[4]] == connection_map[cs[5]] == 3
        assert np.all(combined_connections[3].eval_points == cs[4].eval_points)
        assert np.all(combined_connections[3].eval_points == cs[5].eval_points)

        # Check 7th and 8th connections
        assert np.all(combined_connections[4].eval_points == cs[6].eval_points)
        assert np.all(combined_connections[5].eval_points == cs[7].eval_points)
