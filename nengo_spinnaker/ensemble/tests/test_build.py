"""Tests for Ensemble build functionality.
"""
import collections
import mock
import nengo
import numpy as np
import pytest

from ..connections import IntermediateGlobalInhibitionConnection

from .. import build as ensemble_build
from ...connection import IntermediateConnection


def test_split_out_ensembles():
    # Create a selection of Nengo objects
    objs = [
        nengo.Node(None, add_to_container=False),
        nengo.Ensemble(100, 1, add_to_container=False),
    ]

    ensembles, not_ensembles = ensemble_build.split_out_ensembles(objs)
    assert len(ensembles) == 1 and objs[1] in ensembles
    assert len(not_ensembles) == 1 and objs[0] in not_ensembles


def test_create_placeholder_ensembles():
    with nengo.Network():
        a = nengo.Ensemble(100, 5)

    reps = ensemble_build.create_placeholder_ensembles([a])
    assert len(reps) == 1
    assert reps[a].ens is a
    assert reps[a].record_spikes is False
    assert np.all(reps[a].direct_input == np.zeros(a.size_in))
    assert isinstance(reps[a], ensemble_build.PlaceholderEnsemble)


def test_apply_probing():
    FakeProbe = collections.namedtuple('FakeProbe', 'target attr')

    # Construct some mock objects
    ens1 = mock.Mock(spec_set=['neurons', 'size_in'])
    ens2 = mock.Mock(spec_set=['neurons', 'size_in'])
    ens1.size_in = ens2.size_in = 3

    ie1 = ensemble_build.PlaceholderEnsemble(ens1)
    ie2 = ensemble_build.PlaceholderEnsemble(ens2)

    replaced_ensembles = {ens1: ie1, ens2: ie2}

    # Test Apply Probing for spike probing
    probes = [FakeProbe(ens.neurons, 'output') for ens in [ens1, ens2]]
    probes.append(FakeProbe(mock.Mock(), 'voltage'))  # Not an ensemble
    ensemble_build.apply_probing(replaced_ensembles, probes)
    assert len(ie1.probes) == 1 and ie1.probes[0] == probes[0]
    assert len(ie2.probes) == 1 and ie2.probes[0] == probes[1]
    assert ie1.record_spikes and ie2.record_spikes

    # Test apply probing for mixed probing
    ie1 = ensemble_build.PlaceholderEnsemble(ens1)
    ie2 = ensemble_build.PlaceholderEnsemble(ens2)
    replaced_ensembles = {ens1: ie1, ens2: ie2}

    probes = [FakeProbe(ens1.neurons, 'output'), FakeProbe(ens1, 'voltage')]
    with pytest.raises(NotImplementedError) as excinfo:
        ensemble_build.apply_probing(replaced_ensembles, probes)
        assert "voltage" in excinfo

    probes = [FakeProbe(ens1.neurons, 'input'), FakeProbe(ens1, 'voltage')]
    with pytest.raises(NotImplementedError):
        ensemble_build.apply_probing(replaced_ensembles, probes)
        assert "input" in excinfo


def test_include_constant_inputs():
    # Create some Nodes
    a = nengo.Node(0.5, add_to_container=False)
    b = nengo.Node(lambda t: t, size_in=0, size_out=1, add_to_container=False)
    c = nengo.Node(lambda t, x: None, size_in=1, size_out=0,
                   add_to_container=False)
    d = nengo.Ensemble(100, 1, add_to_container=False)

    # Create an Intermediate ensemble
    ens = ensemble_build.create_placeholder_ensembles([d])[d]

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


def test_build_placeholder():
    # Create two fake neuron types and add a builder for one of them
    class FakeNeuronType1(object):
        pass

    class FakeNeuronType2(object):
        pass

    ensemble_build.ensemble_build_fns[FakeNeuronType1] = mock.Mock()

    # Try to build ensembles with each neuron type, assert that the first
    # successfully calls the build function and that the second fails with a
    # NotImplementedError.
    ens1 = mock.Mock(spec_set=['neuron_type', 'size_in'])
    ens2 = mock.Mock(spec_set=['neuron_type', 'size_in'])
    ens1.neuron_type = FakeNeuronType1()
    ens1.size_in = 3
    ens2.neuron_type = FakeNeuronType2()
    ens2.size_in = 5

    ctree = mock.Mock()
    config = mock.Mock()
    seed = 0xBEEEFFFF

    placeholder1 = ensemble_build.PlaceholderEnsemble(ens1, True)
    placeholder2 = ensemble_build.PlaceholderEnsemble(ens2, False)

    ensemble_build.build_ensemble(placeholder1, ctree, config, seed)
    ensemble_build.ensemble_build_fns[FakeNeuronType1].assert_called_once_with(
        ens1, ctree, config, seed, placeholder1.direct_input, True)

    with pytest.raises(NotImplementedError) as excinfo:
        ensemble_build.build_ensemble(placeholder2, ctree, config, seed)

        assert FakeNeuronType2.__name__ in str(excinfo.value)


def test_build_ensembles():
    """Test that all transforms are correctly applied and that the result is
    sensible.

    TODO: Ensure that PES connections are dealt with correctly.
    """
    with nengo.Network():
        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(100, 2)

        c = nengo.Node([0.5, 0.3])
        p = nengo.Probe(b.neurons)

        cs = [
            nengo.Connection(a, b.neurons, transform=[[-10]]*100),
            nengo.Connection(c, b),
            nengo.Connection(b[0], a),
        ]

    # Call build ensembles on this model
    rngs = {a: np.random.RandomState(123), b: np.random.RandomState(456)}
    (objs, conns) = ensemble_build.build_ensembles([a, b, c], cs, [p], rngs)

    # Assert the objects are the same, and that eval_points have been created
    # for the ensembles.
    assert len(objs) == 3
    for o in objs:
        if isinstance(o, ensemble_build.PlaceholderEnsemble):
            if o.ens is a:
                assert np.all(o.direct_input == np.zeros(1))
                assert o.record_spikes is False
                assert o.ens.eval_points.shape[1] == o.ens.size_out
            elif o.ens is b:
                assert np.all(o.direct_input == [0.5, 0.3])
                assert o.record_spikes is True
                assert o.ens.eval_points.shape[1] == o.ens.size_out
            else:
                assert False, "Unknown object appeared."
        else:
            assert o is c

    # Assert we have one less connection and that one connection is a global
    # inhibition connection.
    print conns
    assert len(conns) == 2
    for c in conns:
        if c.pre_obj.ens is a:
            assert isinstance(c, IntermediateGlobalInhibitionConnection)
            assert c.post_obj.ens is b
        elif c.pre_obj.ens is b:
            assert c.post_obj.ens is a
        else:
            assert False, "Unknown connection appeared."
