"""LIF related Ensembles.
"""

import mock
import nengo
import numpy as np
import random

from ..placeholder import PlaceholderEnsemble
from .. import lif, pes
from ...connections.connection_tree import ConnectionTree
from ...connections.intermediate import IntermediateConnection
from ...utils.fixpoint import bitsk
from nengo_spinnaker.utils import regions as region_utils
from nengo_spinnaker.spinnaker import regions


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


def test_lif_with_encoders_from_numpy_array():
    # Create an Ensemble with no outgoing connections
    with nengo.Network():
        ens = nengo.Ensemble(100, 1)
        ens.encoders = np.random.choice([-1., 1.], size=(100, 1))

    # Create a connection tree
    ctree = ConnectionTree.from_intermediate_connections([])

    # Create a Placeholder for the Ensemble
    p_ens = PlaceholderEnsemble(ens)

    # Build the intermediate representation
    config = mock.Mock()
    rng = np.random.RandomState(1111)
    ilif = lif.IntermediateLIF.build(p_ens, ctree, config, rng)

    # Check the encoders are taken from that selection
    assert np.all(ilif.encoders == ens.encoders[:])


def test_lif_with_encoders_from_numpy_array_normalised():
    # Create an Ensemble with no outgoing connections
    with nengo.Network():
        ens = nengo.Ensemble(100, 3)
        ens.encoders = np.random.uniform(size=(100, 3))

    # Create a connection tree
    ctree = ConnectionTree.from_intermediate_connections([])

    # Create a Placeholder for the Ensemble
    p_ens = PlaceholderEnsemble(ens)

    # Build the intermediate representation
    config = mock.Mock()
    rng = np.random.RandomState(1111)
    ilif = lif.IntermediateLIF.build(p_ens, ctree, config, rng)

    # Check the encoders are normalised
    encoder_mag = np.sqrt(np.sum(ilif.encoders[:]**2, axis=1))
    desired_mag = np.array([1.] * 100)
    assert (np.all(encoder_mag < 1.1 * desired_mag) and
            np.all(encoder_mag > 0.9 * desired_mag))


def test_lif_with_pes_connections():
    """Test building LIF ensembles when PES connections are present.
    """
    with nengo.Network() as model:
        a = nengo.Node(lambda t: t, size_in=0, size_out=1)

        b = nengo.Ensemble(200, 1)
        b.eval_points = np.random.uniform(-1., 1., (1000, 1))

        c = nengo.Node(lambda t, x: None, size_in=1, size_out=0)

        mod_conn = nengo.Connection(a, b, modulatory=True)
        nengo.Connection(b, c, learning_rule_type=nengo.PES(mod_conn))

    # Process the PES connections
    (objs, conns) = pes.process_pes_connections(
        model.all_objects, model.all_connections, model.all_probes)

    # Build the connection tree
    ctree = ConnectionTree.from_intermediate_connections(conns)

    # Create a placeholder
    p_ens = PlaceholderEnsemble(b)

    # Modify the ctree to include the placeholder and a keyspace
    ks = mock.Mock()
    ctree = ctree.get_new_tree_with_replaced_objects({b: p_ens})
    ctree = ctree.get_new_tree_with_applied_keyspace(ks)

    # Build
    rng = np.random.RandomState(1114)
    ilif = lif.IntermediateLIF.build(p_ens, ctree, None, rng)

    # Assert the learning rules are correct
    assert len(ilif.learning_rules) == 1
    assert isinstance(ilif.learning_rules[0].rule, pes.PESInstance)
    assert ilif.learning_rules[0].decoder_index == 0


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
    # Create an intermediate LIF object, then assemble
    ks1 = mock.Mock(name="Keyspace d=1")
    ks2 = mock.Mock(name="Keyspace d=2")
    tau_ref = 0.5
    tau_ref_in_ticks = 0.5 * 10**6 / 10**3
    tau_rc = 0.01
    ilif = lif.IntermediateLIF(
        n_neurons=1, gains=np.array([0.5]), bias=np.array([-1.]),
        encoders=np.array([[1.]]), decoders=np.array([[.2, .3]]),
        tau_ref=tau_ref, tau_rc=tau_rc, decoder_headers=[ks1, ks2],
        learning_rules=list(), direct_input=np.zeros((1, 1))
    )

    # No incoming connections.
    ctree = ConnectionTree.from_intermediate_connections([])

    # Construct
    runtime = 10.
    dt = 0.001
    machine_timestep = 1000
    lif_vertex = lif.assemble_lif_vertex_from_intermediate(
        ilif, ctree, None, {}, runtime, dt, machine_timestep)

    # Now assert that the regions are sensible
    (sys_region, bias_region, encoders_region, decoders_region, outkeys_region,
     input_filters, input_routing, inhib_filters, inhib_routing, gain_region,
     pes_filters, pes_routing, pes_region, blank_region, spikes_region) = \
        lif_vertex.regions

    # System region
    assert isinstance(sys_region, lif.SystemRegion)
    assert sys_region.n_input_dimensions == 1
    assert sys_region.n_output_dimensions == 2
    assert sys_region.machine_timestep == machine_timestep
    assert sys_region.t_ref_in_ticks == tau_ref_in_ticks
    assert sys_region.dt_over_t_rc == bitsk(dt / tau_rc)
    assert sys_region.record_flags == 0x0

    # Bias region
    assert isinstance(bias_region, regions.MatrixRegionPartitionedByRows)
    assert np.all(bias_region.matrix == ilif.bias)
    assert bias_region.formatter is bitsk

    # Encoders region
    assert isinstance(encoders_region, regions.MatrixRegionPartitionedByRows)
    assert np.all(encoders_region.matrix ==
                  np.dot(ilif.encoders, ilif.gains[:, np.newaxis]))
    assert encoders_region.formatter is bitsk

    # Decoders region
    assert isinstance(decoders_region, regions.MatrixRegionPartitionedByRows)
    assert np.all(decoders_region.matrix == ilif.decoders)
    assert decoders_region.formatter is bitsk

    # Output keys
    assert isinstance(outkeys_region, regions.KeysRegion)
    assert not outkeys_region.prepend_n_keys
    assert not outkeys_region.partitioned
    assert outkeys_region.keys == [ks1, ks2]

    # Input filters, inhib filters, PES filters
    for (filter_region, filter_routing) in [(input_filters, input_routing),
                                            (inhib_filters, inhib_routing),
                                            (pes_filters, pes_routing)]:
        # The filter region itself (no incoming connections)
        assert isinstance(filter_region, regions.MatrixRegion)
        assert filter_region.shape == (0, 4)

        # The routing region -- loose test (no incoming connections)
        assert isinstance(filter_routing, regions.KeysRegion)
        assert len(filter_routing.fields) == 4
        assert filter_routing.keys == list()

    # Gain
    assert isinstance(gain_region, regions.MatrixRegionPartitionedByRows)
    assert np.all(gain_region.matrix == ilif.gains)

    # PES
    assert isinstance(pes_region, regions.MatrixRegion)
    assert pes_region.shape == (0,)

    # BLANK
    assert blank_region is None

    # Spikes
    assert isinstance(spikes_region, region_utils.BitfieldBasedRecordingRegion)
