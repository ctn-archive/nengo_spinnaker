import numpy as np

from .. import decoders as decoder_utils


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
    """Should take a list of decoders (and optionally a list of indices and
    other attachments), compress the decoders and return a list of
    (attachment, index, dimension) tuples.
    Attachments are for things like KeySpaces - basically this is compress and
    zip.
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

    # Construct the compressed decoder
    (headers, cdec) = decoder_utils.get_combined_compressed_decoders(
        [dec1, dec2])
    assert(cdec.shape == (100, final_length))

    # Assert the missing dimensions are missing and the decoders were indexed
    ids = [(None, 0, d) for d in dims1]
    ids.extend([(None, 1, d) for d in dims2])
    assert(ids == headers)


def test_get_compressed_decoders_with_indices():
    # Construct a set of 7 decoders
    n_neurons = 500
    decoders = []
    dimensions = []

    for i in range(7):
        n_dims = np.random.randint(3, 10)
        dec = np.random.uniform(0.1, 1, size=(n_neurons, n_dims))

        # Construct a set of missing dimensions
        missing_dims = set(np.random.randint(n_dims,
                                             size=np.random.randint(n_dims/3)))

        # Construct the set of present dimensions
        dims = [n for n in range(n_dims) if n not in missing_dims]

        # Zero the missing dimensions
        for n in range(n_neurons):
            for d in missing_dims:
                dec[n][d] = 0.

        decoders.append(dec)
        dimensions.append(dims)

    # Construct what we expect the header to look like
    indices = [8, 7, 6, 5, 4, 3, 9]
    expected_headers = []
    for (i, ds) in zip(indices, dimensions):
        expected_headers.extend([(None, i, d) for d in ds])

    # Get the combined compressed decoders and check everything is as expected
    headers, cdec = decoder_utils.get_combined_compressed_decoders(
        decoders, indices)

    assert(cdec.shape == (n_neurons, len(expected_headers)))
    assert(headers == expected_headers)


def test_get_compressed_decoders_with_headers():
    # Construct a set of 7 decoders
    n_neurons = 500
    decoders = []
    dimensions = []

    for i in range(7):
        n_dims = np.random.randint(3, 10)
        dec = np.random.uniform(0.1, 1, size=(n_neurons, n_dims))

        # Construct a set of missing dimensions
        missing_dims = set(np.random.randint(n_dims,
                                             size=np.random.randint(n_dims/3)))

        # Construct the set of present dimensions
        dims = [n for n in range(n_dims) if n not in missing_dims]

        # Zero the missing dimensions
        for n in range(n_neurons):
            for d in missing_dims:
                dec[n][d] = 0.

        decoders.append(dec)
        dimensions.append(dims)

    # Construct what we expect the header to look like
    headers = "ABCDEFG"
    indices = range(7)
    expected_headers = []
    for (h, i, ds) in zip(headers, indices, dimensions):
        expected_headers.extend([(h, i, d) for d in ds])

    # Get the combined compressed decoders and check everything is as expected
    headers, cdec = decoder_utils.get_combined_compressed_decoders(
        decoders, indices, headers)

    assert(cdec.shape == (n_neurons, len(expected_headers)))
    assert(headers == expected_headers)


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
