"""Tools for regularising the building and compression of decoders.
"""
import nengo.solvers
import numpy as np

from ..utils import connections as connection_utils


def create_decoder_builder(encoders, radius, gain, bias, rates, rng):
    """Make a function to build the decoder for a reduced outgoing connection.
    """
    # Precompute the encoders / radius
    encoders_over_radius = encoders.T / radius

    # Create the builder function
    def decoder_builder(outgoing_connection):
        """Build the decoder and decoder headers for the outgoing connection.
        """
        # Get the solver
        solver = (outgoing_connection.solver if outgoing_connection.solver
                  is not None else nengo.solvers.LstsqL2())

        # Fail if this is a weight-matrix connection
        if solver.weights:
            raise NotImplementedError(
                "Nengo/SpiNNaker doesn't currently support weight matrices.")

        # Determine the rates of the neurons at each eval point
        x = np.dot(outgoing_connection.eval_points, encoders_over_radius)
        activities = rates(x, gain, bias)

        # Get the targets for the system
        targets = outgoing_connection.get_targets()

        # Build the decoders
        decoder, solver_info = solver(activities, targets, rng=rng)

        # Transform the decoder
        decoder = outgoing_connection.transform.dot(decoder.T).T

        # Create appropriate decoder headers
        decoder_headers = connection_utils.get_keyspaces_with_dimensions([
            outgoing_connection])

        # Check there is one decoder header per column of the decoder block
        assert len(decoder_headers) == decoder.shape[1]

        # Return the decoder, the solver information and the decoder headers
        return (decoder, solver_info, decoder_headers)

    # Return the builder function
    return decoder_builder


def get_compressed_decoder(decoder, threshold=0.):
    """Get a list of used dimensions and a compressed version of the decoder
    with zeroed columns removed.
    """
    # Used dimensions (columns where there is at least one non zero value)
    dims = np.where(np.any(np.abs(decoder) > threshold, axis=0))[0].tolist()

    # Compress the decoder (remove zeroed columns)
    cdec = (decoder.T[dims]).T

    return dims, cdec


def get_combined_compressed_decoders(decoders, headers=None, threshold=0.,
                                     compress=True):
    """Create a compressed decoder block, and return a list of tuples
    samples from headers, indices and dimension numbers.

    :param decoders: A list of decoders to compress.
    :param headers: A list of headers for the decoders (might be KeySpaces).
    :param threshold: Any columns containing a value greater than the threshold
                      will be retained.
    :param compress: A boolean or list of booleans describing whether to
                     compress or otherwise a decoders.
    :returns: A list of tuples of (header, index, dimension) and a combined
              compressed decoder block (NxD).
    """
    if isinstance(compress, bool):
        compress = [compress] * len(decoders)

    assert(headers is None or (sum(d.shape[1] for d in decoders) ==
                               len(headers)))
    assert(len(compress) == len(decoders))
    assert(np.all(decoders[0].shape[0] == d.shape[0] for d in decoders))

    # Compress all of the decoders, with threshold -1. for those we don't want
    # to compress
    cdecoders = list()
    header_indices = list()
    offset = 0
    for (d, c) in zip(decoders, compress):
        # Compress the decoders and get the indices of remaining columns
        dims, cdecoder = get_compressed_decoder(d, threshold if c else -1.)

        # Keep the compressed decoder
        cdecoders.append(cdecoder)

        # Add the indices of remaining columns, offsetting by total width of
        # decoders up to this point.
        header_indices.extend(d + offset for d in dims)

        # Update the offset
        offset += d.shape[1]

    # Combine the final decoder
    if len(cdecoders) > 0:
        decoder = np.hstack(cdecoders)
    else:
        decoder = np.array([])

    # Generate the headers
    final_headers = list()
    if headers is not None:
        final_headers = [headers[d] for d in header_indices]

    return final_headers, decoder
