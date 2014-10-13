"""LIF related Ensembles.
"""

import nengo
import pytest

from .. import lif


class TestIntermediateLIF(object):
    """Ensure that an Intermediate LIF can correctly be constructed from a
    pre-existing Nengo Ensemble.
    """
    def test_from_object_fail(self):
        model = nengo.Network()
        with model:
            a = nengo.Ensemble(100, 1, neuron_type=nengo.neurons.Direct)

        # Incorrect Neuron type
        with pytest.raises(TypeError):
            lif.IntermediateLIF.from_object(a, list(), 0.001, mock.Mock())
