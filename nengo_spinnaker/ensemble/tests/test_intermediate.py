"""Tests for base class for Intermediate Ensemble representations.
"""

import mock
import numpy as np
import pytest

from .. import intermediate


def test_init_intermediate_ensemble():
    # Mismatched gains size
    with pytest.raises(AssertionError):
        intermediate.IntermediateEnsemble(
            n_neurons=100,
            gains=np.zeros(1),
            bias=np.zeros(100),
            encoders=np.zeros((100, 2)),
            decoders=np.zeros((2, 100)),
            decoder_headers=list(),
            learning_rules=list()
        )

    # Mismatched bias size
    with pytest.raises(AssertionError):
        intermediate.IntermediateEnsemble(
            n_neurons=100,
            gains=np.zeros(100),
            bias=np.zeros(1),
            encoders=np.zeros((100, 2)),
            decoders=np.zeros((2, 100)),
            decoder_headers=list(),
            learning_rules=list()
        )

    # Mismatched encoder size
    with pytest.raises(AssertionError):
        intermediate.IntermediateEnsemble(
            n_neurons=100,
            gains=np.zeros(100),
            bias=np.zeros(100),
            encoders=np.zeros((1, 2)),
            decoders=np.zeros((2, 100)),
            decoder_headers=list(),
            learning_rules=list()
        )

    # No issues
    ine = intermediate.IntermediateEnsemble(
        n_neurons=100,
        gains=np.zeros(100),
        bias=np.zeros(100),
        encoders=np.zeros((100, 2)),
        decoders=np.zeros((2, 100)),
        decoder_headers=list(),
        learning_rules=list()
    )

    assert ine.size_in == 2
    assert np.all(ine.direct_input == np.zeros(2))

    # Test that this function exists and doesn't immediately fall over, bad?
    ine.create_output_keyspaces(5, mock.Mock())


def test_create_output_keyspaces():
    decoder_headers = list()
    decoder_headers.extend([(None, 0, n) for n in [1, 2, 5, 6]])
    decoder_headers.append((None, 1, 1))

    ks_1 = mock.Mock()
    decoder_headers.append((ks_1, 2, 0))

    ks = mock.Mock()

    # Check that intermediate keyspaces are correctly created
    kss = intermediate._create_output_keyspaces(decoder_headers, 5, ks)

    # Ensure that the keyspace object is called correctly
    for (k, i, d) in decoder_headers:
        if k is None:
            assert ks.has_call(o=5, i=i, d=d)
        else:
            assert k.has_call(o=5, i=i, d=d)

    assert len(kss) == len(decoder_headers)
