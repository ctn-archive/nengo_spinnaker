"""LIF related Ensembles.
"""

import mock
import nengo
import numpy as np
import pytest
import random

from .. import lif
from ...utils.fixpoint import bitsk
from ... import connection


class TestIntermediateLIF(object):
    """Ensure that an Intermediate LIF can correctly be constructed from a
    pre-existing Nengo Ensemble.
    """
    def test_from_object_fail(self):
        model = nengo.Network()
        with model:
            a = nengo.Ensemble(100, 1, neuron_type=nengo.neurons.Direct())

        # Incorrect Neuron type
        with pytest.raises(AssertionError):
            lif.IntermediateLIF.from_object(a, list(), 0.001, mock.Mock())


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


def test_vertex_from_intermediate():
    # Build a simple network
    model = nengo.Network()
    with model:
        ens = nengo.Ensemble(100, 3)
        a = nengo.Node(lambda t, x: None, size_in=3, size_out=0)

        cs = [
            nengo.Connection(ens[1:], a[1:]),
            nengo.Connection(ens[0], a[0], function=lambda x: x**2),
        ]

    # Create an intermediate ensemble
    ilif = lif.IntermediateLIF.from_object(ens, cs, 0.001, mock.Mock())

    # Convert the connections to IntermediateConnections
    conns = [connection.IntermediateConnection.from_connection(c) for c in cs]

    # Assemble the intermediate representation into a vertex
    asmblr = mock.Mock()
    lifvertex = lif.EnsembleLIF.assemble_from_intermediate(ilif, asmblr)

    # Ug, test things are correct...
    raise NotImplementedError
