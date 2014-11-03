"""Tools for regularising the building and compression of decoders.
"""
import numpy as np
from nengo.utils.stdlib import checked_call


def create_decoder_builder(encoders, radius, gain, bias, rates, rng):
    """Make a function to build the decoder for a reduced outgoing connection.
    """
    # Precompute the encoders / radius
    encoders_over_radius = encoders.T / radius

    # Create the builder function
    def decoder_builder(outgoing_connection):
        """Build the decoder and decoder headers for the outgoing connection.
        """
        # Fail if this is a weight-matrix connection
        if outgoing_connection.solver.weights:
            raise NotImplementedError(
                "Nengo/SpiNNaker doesn't currently support weight matrices.")

        # Determine the rates of the neurons at each eval point
        x = np.dot(outgoing_connection.eval_points, encoders_over_radius)
        activities = rates(x, gain, bias)

        evals = outgoing_connection.eval_points

        # Get the targets for the system
        if outgoing_connection.function is None:
            targets = evals[:, outgoing_connection.pre_slice]
        else:
            # Get the output size of the function
            in_size = (outgoing_connection.pre_slice.stop -
                       outgoing_connection.pre_slice.start)
            function_size = np.asarray(
                checked_call(outgoing_connection.function,
                             np.zeros(in_size))[0]
            ).size

            # Build up the targets for the connection
            targets = np.zeros((evals.shape[0], function_size))
            for i, ep in enumerate(evals):
                targets[i] = outgoing_connection.function(
                    ep[outgoing_connection.pre_slice])

        # Build the decoders
        decoder, solver_info = outgoing_connection.solver(activities, targets,
                                                          rng=rng)

        # Transform the decoder
        decoder = outgoing_connection.transform.dot(decoder.T).T

        # Create appropriate decoder headers
        # For the moment we construct these from the post-slice, later we will
        # just enumerate the columns of the decoder and perform dimension ID
        # mapping on receiving cores.
        decoder_headers = [
            outgoing_connection.keyspace(d=d) for d in
            range(outgoing_connection.width)[outgoing_connection.post_slice]
        ]

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
