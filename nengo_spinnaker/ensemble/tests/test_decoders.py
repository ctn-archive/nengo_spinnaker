import mock
import nengo
import numpy as np
import pytest

from .. import decoders as decoder_utils
from ...connections.intermediate import IntermediateConnection
from ...connections.reduced import OutgoingReducedEnsembleConnection


class TestBuildDecoders(object):
    """Assert that decoders and decoder headers can be built correctly."""
    def test_single_decoder(self):
        """Assert that the correct function calls are made to build decoders.
        """
        # Create some helper items
        n_neurons = 100
        n_dims = 3

        # A solver and wrapper
        _solver = nengo.solvers.LstsqL2()
        solver = mock.Mock(wraps=_solver)
        solver.weights = False

        # Connection function and evaluation points
        transfer_fn = mock.Mock(wraps=lambda x: x**2)
        eval_points = np.random.uniform(-1., 1., size=(100, n_dims))

        # A mock keyspace
        keyspace = mock.Mock(name='Keyspace')
        keyspaces = {d: mock.Mock(name='Keyspace d={}'.format(d)) for d in
                     range(n_dims)}
        keyspace.side_effect = lambda n_dimension: keyspaces[n_dimension]

        # Neuron type and a wrapper for the rates function
        neuron_type = nengo.LIF()
        rates = mock.Mock(wraps=neuron_type.rates)

        # Some (silly) encoders, gains and biases
        encoders = np.random.uniform(-1., 1., size=(n_neurons, n_dims))
        gain = np.linspace(-1., 1., n_neurons)  # Silly
        bias = np.linspace(-1., 1., n_neurons)

        # Create a reduced connection to build a decoder for
        oc = OutgoingReducedEnsembleConnection(
            n_dims, 1., transfer_fn, slice(0, 2), slice(1, 3),
            keyspace=keyspace, eval_points=eval_points,
            solver=solver
        )

        # Build the decoder solver wrapper
        decoder_builder = decoder_utils.create_decoder_builder(
            encoders=encoders, radius=1., gain=gain, bias=bias,
            rates=rates, rng=np.random,
        )

        # Check that building with weights fails
        oc.solver = nengo.solvers.LstsqL2(weights=True)
        with pytest.raises(NotImplementedError) as excinfo:
            decoder_builder(oc)
            assert "weights" in excinfo.value and "support" in excinfo.value

        # Build a decoder from the outgoing connection
        oc.solver = solver
        (decoder, solver_info, decoder_headers) = decoder_builder(oc)

        # Test that the correct calls were made to supporting objects
        # Rates from the neuron type
        assert rates.called
        assert np.all(rates.call_args[0][1] == gain)
        assert np.all(rates.call_args[0][2] == bias)

        # The first call to the function was 0s
        assert np.all(transfer_fn.call_args_list[0][0] == np.zeros(2))

        # Solver is called
        assert solver.called

        # Test that the decoder is sane
        assert decoder.shape == (n_neurons, 2)

        # Test that the decoder headers are sane
        assert decoder_headers == [keyspaces[1], keyspaces[2]]

    def test_transformed_decoder(self):
        """Test that transforms are correctly applied to decoders.
        Also takes the branch for no function.
        """
        with nengo.Network():
            a = nengo.Ensemble(100, 3)
            b = nengo.Ensemble(100, 3)

            c = nengo.Connection(a[0:2], b, transform=np.zeros((3, 2)))

            a.encoders = np.random.uniform(size=(a.n_neurons, a.size_in))
            a.eval_points = np.random.normal(size=(1000, a.size_in))
            a.gain = np.random.normal(size=a.n_neurons)
            a.bias = np.linspace(-1., 1., a.n_neurons)
            a.solver = nengo.solvers.LstsqL2()

        # Create an intermediate connection and outgoing reduced connection
        ic = IntermediateConnection.from_connection(c, keyspace=mock.Mock())
        oc = ic.get_reduced_outgoing_connection()

        # Create the decoder builder
        decoder_builder = decoder_utils.create_decoder_builder(
            encoders=a.encoders, radius=a.radius, gain=a.gain, bias=a.bias,
            rates=a.neuron_type.rates, rng=np.random
        )

        # Build the decoder
        decoder, solver_info, headers = decoder_builder(oc)
        assert np.all(decoder == np.zeros((a.n_neurons, 1))), "Not transformed"


def test_get_compressed_decoder():
    """Compressing decoders removes columns from the decoder where all the
    values are 0.0 (and hence no neuron ever affects them!).  The new decoder
    and the indices of the remaining dimensions should be returned.
    """
    # Generate a new random decoder, and zero out columns 1, 5, 6
    dec = np.random.normal(size=(100, 7))
    for d in [0, 4, 5]:
        for n in range(100):
            dec[n][d] = 0.

    # Get a compressed version of the decoder, and a list of used dimensions
    (dims, cdec) = decoder_utils.get_compressed_decoder(dec)

    assert(dims == [1, 2, 3, 6])
    assert(cdec.shape == (100, 4))
    assert(cdec[0][0] == dec[0][1])
    assert(cdec[0][3] == dec[0][6])


def test_get_compressed_decoders():
    """Should take a list of decoders and a list of headers, compress the
    decoders and return a reduced list of headers.  Attachments are for things
    like KeySpaces - basically this is compress and zip.
    """
    dec1 = np.array([[1.]*8]*100)
    dec2 = np.array([[2.]*10]*100)

    # Select which columns to zero out
    rdims_1 = set(np.random.randint(7, size=3).tolist())
    rdims_2 = set(np.random.randint(8, size=4).tolist())

    dims1 = [n for n in range(8) if n not in rdims_1]
    dims2 = [n for n in range(10) if n not in rdims_2]
    final_length = len(dims1) + len(dims2)

    # Zero out those columns
    for n in range(100):
        for d in rdims_1:
            dec1[n][d] = 0.
        for d in rdims_2:
            dec2[n][d] = 0.

    # Construct headers
    old_headers = range(8) + range(10)

    # Construct the compressed decoder
    (headers, cdec) = decoder_utils.get_combined_compressed_decoders(
        [dec1, dec2], old_headers)
    assert(cdec.shape == (100, final_length))

    # Assert the missing dimensions are missing and the decoders were indexed
    ids = [d for d in dims1]
    ids.extend(d for d in dims2)
    assert(ids == headers)


def test_get_compressed_and_uncompressed_decoders():
    # Construct 3 decoders
    d1 = np.random.uniform(.1, 1., (100, 5))
    d2 = np.zeros((100, 9))
    d3 = np.random.uniform(.1, 1., (100, 7))

    # Zero some of the elements of d3
    for d in [0, 5]:
        for n in range(100):
            d3[n][d] = 0.

    # Compress the decoders
    headers, cdec = decoder_utils.get_combined_compressed_decoders(
        [d1, d2, d3], compress=[True, False, True])

    # d2 should not have been compressed, d3 should have been
    assert(cdec.shape[1] == 5 + 9 + 5)


def test_null_decoders():
    headers, cdec = decoder_utils.get_combined_compressed_decoders(
        [], headers=[])
