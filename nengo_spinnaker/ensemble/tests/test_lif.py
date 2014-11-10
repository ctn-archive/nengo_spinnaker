"""LIF related Ensembles.
"""

import mock
import nengo
import numpy as np
import random

from ..placeholder import PlaceholderEnsemble
from .. import lif
from ...connections.connection_tree import ConnectionTree
from ...connections.intermediate import IntermediateConnection
from ...utils.fixpoint import bitsk


def test_build():
    # Build a simple network
    model = nengo.Network()
    with model:
        ens = nengo.Ensemble(100, 3, label="Ensemble")
        ens.eval_points = np.random.uniform(size=(100, 3))
        a = nengo.Ensemble(100, 3, label="A")
        a.eval_points = np.random.uniform(size=(100, 3))

        cs = [
            nengo.Connection(ens[1:], a[1:]),
            nengo.Connection(ens[0], a[0], function=lambda x: x**2),
        ]

    # Create the connection tree for this network
    ks = mock.Mock()
    new_cs = [IntermediateConnection.from_connection(c, keyspace=ks) for c in
              cs]

    p_ens = PlaceholderEnsemble(ens)
    p_a = PlaceholderEnsemble(a)

    for i in range(len(new_cs)):
        new_cs[i].pre_obj = p_ens
        new_cs[i].post_obj = p_a

    ctree = ConnectionTree.from_intermediate_connections(new_cs)

    config = mock.Mock()

    # Now create a new LIF intermediate
    rng = np.random.RandomState(1105)
    ilif = lif.IntermediateLIF.build(
        p_ens, connection_trees=ctree, config=config, rng=rng)

    # Make some assertions about the new Intermediate LIF object
    assert ilif.n_neurons == ens.n_neurons
    assert ilif.tau_rc == ens.neuron_type.tau_rc
    assert ilif.tau_ref == ens.neuron_type.tau_ref
    assert len(ilif.decoder_headers) == 3
    assert ilif.encoders.shape == (100, 3)
    assert ilif.decoders.shape == (100, 3)


def test_lif_system_region():
    """Create a system region for an LIF ensemble.
    """
    # Create the regions
    region1 = lif.SystemRegion(n_input_dimensions=10,  # n cols of encoder
                               n_output_dimensions=5,  # n cols of decoders
                               machine_timestep=1000,  # should be in app_ptr
                               t_ref=0.01,             # specific to LIF
                               dt_over_t_rc=0.05,      # specific to LIF
                               record_spikes=False
                               )
    region2 = lif.SystemRegion(n_input_dimensions=3,   # n cols of encoder
                               n_output_dimensions=1,  # n cols of decoders
                               machine_timestep=1000,  # should be in app_ptr
                               t_ref=0.02,             # specific to LIF
                               dt_over_t_rc=0.07,      # specific to LIF
                               record_spikes=True
                               )

    # Assert the size is sensible
    assert region1.sizeof(slice(0, 1000)) == 8

    # Assert that subregions are formatted correctly
    sr1 = region1.create_subregion(slice(0, 10), 1)
    data = np.frombuffer(sr1.data, dtype=np.uint32)
    assert data[0] == 10  # input dims
    assert data[1] == 5  # output dims
    assert data[2] == 10  # number of atoms
    assert data[3] == 1000  # timestep
    assert data[4] == int(0.01 / (1000 * 10**-6))  # t_ref in ticks
    assert data[5] == bitsk(0.05)  # dt over t_rc
    assert data[6] == 0x0  # not recording spikes
    assert data[7] == 1  # number of inhibitory dimensions

    for i in range(100):
        low_val = random.randint(0, 1000)
        high_val = random.randint(low_val+1, low_val+1000)
        n_atoms = high_val - low_val

        sr = region1.create_subregion(slice(low_val, high_val), i)
        data = np.frombuffer(sr.data, dtype=np.uint32)
        assert data[2] == n_atoms

    sr1 = region2.create_subregion(slice(0, 10), 1)
    data = np.frombuffer(sr1.data, dtype=np.uint32)
    assert data[0] == 3  # input dims
    assert data[1] == 1  # output dims
    assert data[2] == 10  # number of atoms
    assert data[3] == 1000  # timestep
    assert data[4] == int(0.02 / (1000 * 10**-6))  # t_ref in ticks
    assert data[5] == bitsk(0.07)  # dt over t_rc
    assert data[6] == 0x1  # not recording spikes
    assert data[7] == 1  # number of inhibitory dimensions


def test_lif_assemble_from_intermediate():
    # Create a small example, build the intermediate representation and then
    # assemble.
    raise NotImplementedError
