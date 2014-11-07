"""Tests for base class for Intermediate Ensemble representations.
"""

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
