import mock
import nengo
import numpy as np
import pytest

from nengo_spinnaker import utils


def test_decoder_generation():
    """Ensure that Decoders are only generated when absolutely necessary!"""
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(100, 3)
        b = nengo.Node(lambda t, v: None, size_in=3, size_out=0)
        c = nengo.Node(lambda t, v: None, size_in=3, size_out=0)

        # Create a series of connections, some identical, some otherwise
        c1 = nengo.Connection(a, b)
        c2 = nengo.Connection(a, c)  # Should share decoder with c1

        f = lambda v: v**2
        c3 = nengo.Connection(a, b, function=f)  # Share ()
        c4 = nengo.Connection(a, c, function=f,
                              transform=np.random.normal(size=(3, 3)))  # Share (c3)

        c5 = nengo.Connection(a, c, solver=nengo.decoders.LstsqNoise)  # Share ()
        c6 = nengo.Connection(a, c, eval_points=np.random.normal(100)) # !Share

        c7 = nengo.Connection(a, c, transform=3)

    # Build the decoders in order
    decoder_build_func = mock.Mock()
    decoder_build_func.return_value = 0.
    decoder_builder = utils.decoders.DecoderBuilder(decoder_build_func)

    # Build C1, should cause the decoder build func to be called
    decoder_builder.get_transformed_decoder(c1.function, c1.transform,
                                            c1.eval_points, c1.solver)
    decoder_build_func.assert_called_with(
        c1.function, c1.eval_points, c1.solver)

    # Build C2, should NOT cause the decoder build func to be called
    decoder_build_func.reset_mock()
    decoder_builder.get_transformed_decoder(c2.function, c2.transform,
                                            c2.eval_points, c2.solver)
    assert(not decoder_build_func.called)

    # Build C3, should cause the decoder build func to be called
    decoder_build_func.reset_mock()
    decoder_builder.get_transformed_decoder(c3.function, c3.transform,
                                            c3.eval_points, c3.solver)
    decoder_build_func.assert_called_with(
        c3.function, c3.eval_points, c3.solver)

    # Build C4, should NOT ...
    decoder_build_func.reset_mock()
    decoder_builder.get_transformed_decoder(c4.function, c4.transform,
                                            c4.eval_points, c4.solver)
    assert(not decoder_build_func.called)

    # Build C5, should ...
    decoder_build_func.reset_mock()
    decoder_builder.get_transformed_decoder(c5.function, c5.transform,
                                            c5.eval_points, c5.solver)
    decoder_build_func.assert_called_with(
        c5.function, c5.eval_points, c5.solver)

    # Build C6, should ...
    decoder_build_func.reset_mock()
    decoder_builder.get_transformed_decoder(c6.function, c6.transform,
                                            c6.eval_points, c6.solver)
    decoder_build_func.assert_called_with(
        c6.function, c6.eval_points, c6.solver)

    # Check that the decoder is transformed
    dec = np.random.uniform((3, 100))
    decoder_builder = utils.decoders.DecoderBuilder(lambda f, e, s: dec)
    tdec = decoder_builder.get_transformed_decoder(c7.function, c7.transform,
                                                   c7.eval_points, c7.solver)
    assert(np.all(tdec == np.dot(dec, 3)))
