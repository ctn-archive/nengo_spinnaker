"""Tests for Ensemble build functionality.
"""
import collections
import mock
import nengo
import numpy as np
import pytest

from .. import build as ensemble_build
from ...connection import IntermediateConnection
from ..intermediate import IntermediateEnsemble


def test_split_out_ensembles():
    # Create a selection of Nengo objects
    objs = [
        nengo.Node(None, add_to_container=False),
        nengo.Ensemble(100, 1, add_to_container=False),
    ]

    ensembles, not_ensembles = ensemble_build.split_out_ensembles(objs)
    assert len(ensembles) == 1 and objs[1] in ensembles
    assert len(not_ensembles) == 1 and objs[0] in not_ensembles


def test_replaced_ensembles():
    """Test that Ensembles are correctly replaced and that errors are raised
    for unknown neuron types.
    """
    class SillyNeuronType(nengo.neurons.NeuronType):
        pass

    ens1 = [
        nengo.Ensemble(100, 1, add_to_container=False),
    ]
    ens2 = [
        nengo.Ensemble(100, 1, neuron_type=SillyNeuronType(),
                       add_to_container=False),
    ]

    rep1 = ensemble_build.replace_ensembles(
        ens1, [], None, rng=np.random.RandomState(110591))
    assert len(rep1) == len(ens1)
    assert rep1.keys() == ens1

    with pytest.raises(NotImplementedError):
        ensemble_build.replace_ensembles(ens2, [], None, None)


def test_apply_probing():
    FakeProbe = collections.namedtuple('FakeProbe', 'target attr')
    
    class FakeIntermediateEnsemble(object):
        def __init__(self):
            self.record_spikes = False
            self.record_voltage = False
            self.probes = list()

    # Construct some mock objects
    ens1 = mock.Mock()
    ens2 = mock.Mock()
    ie1 = FakeIntermediateEnsemble()
    ie2 = FakeIntermediateEnsemble()
    replaced_ensembles = {ens1: ie1, ens2: ie2}

    # Test Apply Probing for spike probing
    probes = [FakeProbe(ens, 'spikes') for ens in [ens1, ens2]]
    probes.append(FakeProbe(mock.Mock(), 'voltage'))  # Not an ensemble
    ensemble_build.apply_probing(replaced_ensembles, probes)
    assert len(ie1.probes) == 1 and ie1.probes[0] == probes[0]
    assert len(ie2.probes) == 1 and ie2.probes[0] == probes[1]
    assert ie1.record_spikes and ie2.record_spikes

    # Test apply probing for mixed probing
    ie1 = FakeIntermediateEnsemble()
    ie2 = FakeIntermediateEnsemble()
    replaced_ensembles = {ens1: ie1, ens2: ie2}
    probes = [FakeProbe(ens1, 'spikes'), FakeProbe(ens1, 'voltage')]
    with pytest.raises(NotImplementedError):
        ensemble_build.apply_probing(replaced_ensembles, probes)


def test_include_constant_inputs():
    # Create some Nodes
    a = nengo.Node(0.5, add_to_container=False)
    b = nengo.Node(lambda t: t, size_in=0, size_out=1, add_to_container=False)
    c = nengo.Node(lambda t, x: None, size_in=1, size_out=0,
                   add_to_container=False)

    # Create an Intermediate ensemble
    ens = IntermediateEnsemble(100, np.zeros(100), np.zeros(100),
                               np.zeros((100, 1)), np.zeros(100),
                               np.zeros(100), list(), list())

    # Create connections, 1st one to be removed
    cs = [
        IntermediateConnection(a, ens, transform=0.5, function=np.sin), 
        IntermediateConnection(b, ens),
        IntermediateConnection(a, c),
        IntermediateConnection(ens, c),
    ]

    # Run the transform function
    remove_connections = ensemble_build.include_constant_inputs(cs)

    assert len(remove_connections) == 1 and remove_connections[0] is cs[0]
    assert ens.direct_input == cs[0].function(a.output) * cs[0].transform
